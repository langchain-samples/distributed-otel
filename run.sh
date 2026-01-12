#!/bin/bash
# Main entry point for the multi-agent system
#
# Starts:
# 1. CrewAI MCP Server (HTTP) on localhost:8000
# 2. LangGraph Orchestrator with BOTH native research + MCP tools

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(cat "$PROJECT_ROOT/.env" | grep -v '^#' | xargs)
fi

CREWAI_DIR="$PROJECT_ROOT/crewai"
ORCHESTRATOR_DIR="$PROJECT_ROOT/langgraph"

# Python executables from venvs
CREWAI_PYTHON="$CREWAI_DIR/.venv/bin/python"
ORCHESTRATOR_PYTHON="$ORCHESTRATOR_DIR/.venv/bin/python"

# Scripts
CREWAI_SCRIPT="$CREWAI_DIR/crew_mcp_agent.py"
ORCHESTRATOR_SCRIPT="$ORCHESTRATOR_DIR/orchestrator.py"

echo "============================================================"
echo "Multi-Agent System: Native Research + MCP CrewAI Tools"
echo "============================================================"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping CrewAI MCP Server..."
    if [ -n "$MCP_PID" ]; then
        kill $MCP_PID 2>/dev/null || true
        wait $MCP_PID 2>/dev/null || true
    fi
    echo "✓ Cleanup complete"
}

trap cleanup EXIT INT TERM

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_ROOT/logs"

# Start MCP server in background
echo "🚀 Starting CrewAI MCP Server..."
echo "   Logs: $PROJECT_ROOT/logs/crewai_mcp.log"
$CREWAI_PYTHON "$CREWAI_SCRIPT" > "$PROJECT_ROOT/logs/crewai_mcp.log" 2>&1 &
MCP_PID=$!

# Wait for server to be ready
echo "⏳ Waiting for MCP server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/ > /dev/null 2>&1; then
        echo "✓ MCP server is ready"
        break
    fi
    if ! kill -0 $MCP_PID 2>/dev/null; then
        echo "✗ MCP server failed to start"
        cat "$PROJECT_ROOT/logs/crewai_mcp.log"
        exit 1
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo "✗ Timeout waiting for MCP server"
        cat "$PROJECT_ROOT/logs/crewai_mcp.log"
        exit 1
    fi
done

# Run orchestrator
echo ""
echo "🎯 Starting LangGraph Orchestrator..."
echo "   Logs: $PROJECT_ROOT/logs/orchestrator.log"
$ORCHESTRATOR_PYTHON "$ORCHESTRATOR_SCRIPT" 2>&1 | tee "$PROJECT_ROOT/logs/orchestrator.log"

exit_code=$?
exit $exit_code
