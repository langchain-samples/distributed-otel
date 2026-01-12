"""
LangGraph Orchestrator with OpenTelemetry Distributed Tracing

Demonstrates distributed tracing between LangGraph and CrewAI MCP services using
OpenTelemetry context propagation (inject/extract pattern).

get_current_span() must be used instead of the standard
trace.get_current_span() because LangChain's callback-based architecture loses
context during queued execution. See: https://github.com/Arize-ai/openinference/issues/1103
"""
import os
import sys
import asyncio
from pathlib import Path
from typing import Literal, Callable, Awaitable
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from otel import setup_otel

tracer = setup_otel(service_name="langgraph-orchestrator")

from openinference.instrumentation.langchain import LangChainInstrumentor, get_current_span as oi_get_current_span
LangChainInstrumentor().instrument()

# 3. Import LangChain after instrumentation
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

# MCP imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import ToolCallInterceptor, MCPToolCallRequest, MCPToolCallResult

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.propagate import inject


class OTELContextPropagationInterceptor(ToolCallInterceptor):
    """Intercepts MCP tool calls to inject OpenTelemetry trace context for distributed tracing."""

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        # Use OpenInference's get_current_span (not trace.get_current_span)
        # LangChain's queued execution loses standard OTEL context
        parent_span = oi_get_current_span()
        parent_context = trace.set_span_in_context(parent_span) if parent_span else None

        with tracer.start_as_current_span(f"mcp.{request.name}", context=parent_context) as span:
            span.set_attribute("tool.name", request.name)

            headers = dict(request.headers or {})
            inject(headers)
            modified_request = request.override(headers=headers)
            return await handler(modified_request)


# Configuration
CREWAI_MCP_URL = os.getenv("CREWAI_MCP_URL", "http://localhost:8000/mcp")
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0


class OrchestratorState(MessagesState):
    pass


def create_clean_mcp_tools(tools):
    """Creates a tool node that strips the runtime parameter from tool calls."""

    async def clean_mcp_tools(state: OrchestratorState, config: RunnableConfig):
        messages = state["messages"]
        tool_calls = messages[-1].tool_calls

        results = []
        for tc in tool_calls:
            clean_args = {k: v for k, v in tc["args"].items() if k != "runtime"}
            tool = next((t for t in tools if t.name == tc["name"]), None)
            if tool:
                tool_result = await tool.ainvoke(clean_args, config=config)
                results.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tc["id"]
                ))

        return {"messages": results}

    return clean_mcp_tools


def should_continue(state: OrchestratorState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


SYSTEM_PROMPT = """You are a research and writing assistant with access to specialized AI agents through CrewAI.

Available tools:
- research_topic: Call a research agent to thoroughly research a topic
- write_content: Call a writing agent to create professional content

Your job is to:
1. Understand the user's request
2. Decide which tools to call and in what order (you can call multiple tools)
3. Use the results to provide a comprehensive response
4. Synthesize all information into a clear, final answer

If a task requires both research and writing, call research_topic first, then write_content with the research results.
"""


def create_workflow(mcp_tools, llm_with_tools):
    """Create and compile the ReAct workflow."""

    def create_agent_node(llm_with_tools):
        async def agent_node(state: OrchestratorState):
            messages = state["messages"]
            prompt = [SystemMessage(content=SYSTEM_PROMPT)] + messages
            response = await llm_with_tools.ainvoke(prompt)
            return {"messages": [response]}

        return agent_node

    workflow = StateGraph(OrchestratorState)
    workflow.add_node("agent", create_agent_node(llm_with_tools))
    workflow.add_node("tools", create_clean_mcp_tools(mcp_tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile()


async def run_orchestrator(task: str):
    """Run the orchestrator with a given task."""
    print(f"\n{'='*60}")
    print(f"Starting LangGraph ReAct Orchestrator")
    print(f"Task: {task}")
    print(f"Connecting to CrewAI MCP server at {CREWAI_MCP_URL}")
    print(f"{'='*60}\n")

    with tracer.start_as_current_span("langgraph-orchestrator") as root_span:
        root_span.set_attribute("task", task)

        client = MultiServerMCPClient(
            {
                "crewai": {
                    "transport": "streamable_http",
                    "url": CREWAI_MCP_URL
                }
            },
            tool_interceptors=[OTELContextPropagationInterceptor()]
        )

        mcp_tools = await client.get_tools()
        print(f"Loaded {len(mcp_tools)} tools from CrewAI MCP server\n")

        llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        llm_with_tools = llm.bind_tools(mcp_tools)

        orchestrator = create_workflow(mcp_tools, llm_with_tools)

        result = await orchestrator.ainvoke(
            {"messages": [HumanMessage(content=task)]},
            config={"run_name": "LangGraph"}
        )

        print(f"\n{'='*60}")
        print("Orchestrator Execution Complete")
        print(f"{'='*60}\n")

        final_response = result["messages"][-1].content
        print(final_response)

        return result


async def main():
    task = "Research the impact of artificial intelligence on healthcare and write a brief summary"
    await run_orchestrator(task)


if __name__ == "__main__":
    asyncio.run(main())
