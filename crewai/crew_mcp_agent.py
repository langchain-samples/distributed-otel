"""
CrewAI MCP Server with OpenTelemetry Distributed Tracing

This MCP server exposes CrewAI crews as tools via HTTP with distributed tracing support.
Uses standard OpenTelemetry context propagation (extract) to continue traces from orchestrator.

Based on: https://docs.langchain.com/langsmith/trace-with-opentelemetry
"""
import asyncio
import sys
import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent

# OpenTelemetry for distributed tracing
from opentelemetry import trace, context as otel_context
from opentelemetry.propagate import extract

# Add parent directory to path for shared otel_setup
sys.path.insert(0, str(Path(__file__).parent.parent))
from otel import setup_otel

load_dotenv()

# Set up OTEL TracerProvider + LangSmith exporter BEFORE importing CrewAI
tracer = setup_otel(service_name="crewai-mcp-server")

# Instrument CrewAI with OpenInference
from openinference.instrumentation.crewai import CrewAIInstrumentor
CrewAIInstrumentor().instrument()

# NOW import CrewAI after OTEL is configured
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI

# Configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
SERVER_PORT = 8000

# Initialize FastMCP server
mcp = FastMCP("crewai-mcp-server")

# LLM instance
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)


# Simple distributed tracing following the docs example
async def run_crew_with_tracing(
    crew_name: str,
    tool_name: str,
    inputs: dict,
    agent: Agent,
    task: Task,
    ctx: Context
) -> str:
    """Execute CrewAI crew with distributed tracing.

    Follows the Service B pattern from docs:
    1. Extract context from headers
    2. Attach context to make it active (critical for child spans)
    3. Execute work inside the span
    """
    # Extract parent context from headers (Service B pattern)
    request = ctx.request_context.request
    print(f"Headers: {dict(request.headers)}", file=sys.stderr)
    parent_context = extract(dict(request.headers)) if request else None
    

    token = otel_context.attach(parent_context) if parent_context else None

    try:
        with tracer.start_as_current_span(f"crewai.{tool_name}") as span:
            span.set_attribute("crew.name", crew_name)

            # Build and execute crew - CrewAI instrumentation will create child spans
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
                name=crew_name
            )

            result = crew.kickoff(inputs=inputs)
            return result.raw if hasattr(result, 'raw') else str(result)
    finally:
        if token:
            otel_context.detach(token)


# Define CrewAI tool for tracking agent thinking
@tool("Think")
def think(thought: str) -> str:
    """Records a thought, plan, or observation during task execution.

    IMPORTANT: Use this tool sparingly - MAXIMUM 1-2 times per task.
    Only use when you have a critical insight or need to note a major decision.

    Good use cases:
    - Recording your overall strategy at the start (1 time)
    - Noting a critical insight or decision point (1 time max)

    Do NOT use for:
    - Every small step or observation
    - Routine information gathering
    - Minor details or facts

    Args:
        thought: Your thought, plan, observation, or note about the current task

    Returns:
        Confirmation that the thought was recorded
    """
    return f"Thought recorded: {thought}"


@mcp.tool()
async def research_topic(topic: str, ctx: Context) -> str:
    """Research a given topic thoroughly and provide detailed analysis.

    Args:
        topic: The topic to research
        ctx: MCP context with request headers for distributed tracing
    """
    # Configure researcher agent
    researcher = Agent(
        role="Research Analyst",
        goal="Conduct thorough research on given topics",
        backstory="You are an expert research analyst with deep knowledge across domains.",
        llm=llm,
        verbose=True,
        tools=[think]
    )

    task = Task(
        description=f"Research the following topic thoroughly: {topic}",
        expected_output="A detailed research report with key findings",
        agent=researcher
    )

    # Execute crew with distributed tracing (Service B pattern)
    return await run_crew_with_tracing(
        crew_name="Research Crew",
        tool_name="research_topic",
        inputs={"topic": topic},
        agent=researcher,
        task=task,
        ctx=ctx
    )


@mcp.tool()
async def write_content(subject: str, ctx: Context, style: str = "professional") -> str:
    """Write engaging content on a given subject.

    Args:
        subject: The subject to write about
        ctx: MCP context with request headers for distributed tracing
        style: Writing style (e.g., formal, casual, technical)
    """
    # Configure writer agent
    writer = Agent(
        role="Content Writer",
        goal="Create engaging and informative content",
        backstory="You are a skilled writer who creates clear, compelling content.",
        llm=llm,
        verbose=True,
        tools=[think]
    )

    task = Task(
        description=f"Write {style} content about: {subject}",
        expected_output=f"Well-written {style} content",
        agent=writer
    )

    # Execute crew with distributed tracing (Service B pattern)
    return await run_crew_with_tracing(
        crew_name="Writing Crew",
        tool_name="write_content",
        inputs={"subject": subject, "style": style},
        agent=writer,
        task=task,
        ctx=ctx
    )


if __name__ == "__main__":
    # Run as HTTP server using streamable HTTP transport
    print(f"✓ Starting CrewAI MCP HTTP Server on http://localhost:{SERVER_PORT}", file=sys.stderr)
    mcp.run(transport="streamable-http")