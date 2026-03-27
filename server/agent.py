#!/usr/bin/env python3
"""
Multi-Model Anomaly Investigation Agent
========================================

A single LangChain agent with dynamic model switching via OpenRouter.
Uses Claude Sonnet for routine status checks and Claude Opus for
complex failure investigation and cross-agent correlation analysis.

Two operating modes:
  1. EMBEDDED — imported by api.py, tools call the API's in-memory data
     directly through accessor functions.  The API exposes a streaming
     SSE endpoint at POST /api/agent/chat.
  2. STANDALONE — run as a CLI, tools call the API over HTTP.

Usage (standalone):
    python agent.py
    python agent.py --query "What failures have occurred recently?"
    python agent.py --api-url http://localhost:8080

Requirements:
    pip install langchain langchain-openrouter langgraph httpx
"""

import os
import json
import argparse
from typing import Callable, Generator

import httpx
from langchain.tools import tool
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.messages import AIMessageChunk, AIMessage, ToolMessage as LCToolMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000")

FAST_MODEL  = "anthropic/claude-sonnet-4.6"
POWER_MODEL = "anthropic/claude-opus-4.6"

COMPLEX_TOOLS = {
    "investigate_failures",
    "cross_agent_correlation",
    "compare_nodes",
    "search_readings",
}


# ─────────────────────────────────────────────
# Data access layer
#
# When embedded in the API process, api.py sets these to functions
# that read the in-memory stores directly (no HTTP round-trip).
# When running standalone CLI, they fall back to HTTP calls.
# ─────────────────────────────────────────────

_data_accessors: dict[str, Callable] = {}


def set_data_accessors(accessors: dict[str, Callable]):
    """Called by api.py to inject direct data-access functions.

    Expected keys:
        "system_status", "node_detail", "failures", "sensor_summary",
        "compare_nodes", "correlation", "recent_activity", "search_readings"

    Each function must accept the same kwargs as the corresponding tool
    and return a dict (the JSON-serialisable response body).
    """
    _data_accessors.update(accessors)


def _is_embedded() -> bool:
    """True when running inside the API process (accessors injected)."""
    return len(_data_accessors) > 0


# HTTP fallback for standalone CLI mode
_http_client: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(base_url=API_BASE_URL, timeout=15.0)
    return _http_client


def _api_get(path: str, params: dict | None = None) -> dict:
    """HTTP GET fallback used only in standalone CLI mode."""
    client = _get_http_client()
    resp = client.get(path, params={k: v for k, v in (params or {}).items() if v is not None})
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# Tool Definitions
# ─────────────────────────────────────────────

@tool
def get_system_status() -> str:
    """Get a high-level snapshot of the entire system — which agents are running,
    how many nodes are alive/dead, total readings, alert counts, current accuracy,
    and any degraded nodes. Use this FIRST to orient yourself before diving deeper."""
    if _is_embedded():
        data = _data_accessors["system_status"]()
    else:
        data = _api_get("/api/agent/system_status")
    return json.dumps(data, indent=2)


@tool
def get_node_detail(agent_type: str, node_id: str) -> str:
    """Deep-dive into a single node's performance, recent readings, alerts, and
    trend analysis. Use after spotting a degraded or suspicious node.

    Args:
        agent_type: Either 'maintenance' or 'power'
        node_id: The node identifier, e.g. 'node_1'
    """
    if _is_embedded():
        data = _data_accessors["node_detail"](agent_type=agent_type, node_id=node_id)
    else:
        data = _api_get(f"/api/agent/node_detail/{agent_type}/{node_id}")
    return json.dumps(data, indent=2)


@tool
def investigate_failures(
    agent_type: str | None = None,
    node_id: str | None = None,
    kind: str = "all",
    n: int = 20,
) -> str:
    """Retrieve detailed failure/attack events with full sensor context. Use this
    to understand WHAT went wrong — which readings triggered alerts, sensor values
    at that moment, and whether predictions were correct.

    Args:
        agent_type: Filter to 'maintenance' or 'power' (optional, omit for both)
        node_id: Filter to a specific node (optional)
        kind: 'all', 'true_positive', 'false_positive', 'false_negative', or 'missed'
        n: Max number of results (default 20, max 100)
    """
    if _is_embedded():
        data = _data_accessors["failures"](agent_type=agent_type, node_id=node_id, kind=kind, n=n)
    else:
        params = {"kind": kind, "n": str(n)}
        if agent_type:
            params["agent_type"] = agent_type
        if node_id:
            params["node_id"] = node_id
        data = _api_get("/api/agent/failures", params)
    return json.dumps(data, indent=2)


@tool
def get_sensor_summary(agent_type: str, node_id: str | None = None) -> str:
    """Get statistical summary of sensor values (min, max, mean, std) across
    recent readings. Helps understand normal operating ranges and spot outliers.
    Includes separate profiles for normal vs alert readings.

    Args:
        agent_type: Either 'maintenance' or 'power'
        node_id: Filter to a specific node (optional)
    """
    if _is_embedded():
        data = _data_accessors["sensor_summary"](agent_type=agent_type, node_id=node_id)
    else:
        params = {}
        if node_id:
            params["node_id"] = node_id
        data = _api_get(f"/api/agent/sensor_summary/{agent_type}", params)
    return json.dumps(data, indent=2)


@tool
def compare_nodes(agent_type: str) -> str:
    """Side-by-side performance comparison of all nodes of a given type.
    Quickly spot which node is underperforming — shows accuracy, precision,
    recall, F1, alert rate, and identifies best/worst nodes.

    Args:
        agent_type: Either 'maintenance' or 'power'
    """
    if _is_embedded():
        data = _data_accessors["compare_nodes"](agent_type=agent_type)
    else:
        data = _api_get(f"/api/agent/compare_nodes/{agent_type}")
    return json.dumps(data, indent=2)


@tool
def cross_agent_correlation(window_s: int = 60, n: int = 20) -> str:
    """Check whether maintenance failures and power attacks co-occur within a
    time window. Use this to investigate if a power grid attack triggered
    equipment failures or vice versa.

    Args:
        window_s: Time window in seconds to consider 'co-occurring' (default 60)
        n: Max correlated event pairs to return (default 20)
    """
    if _is_embedded():
        data = _data_accessors["correlation"](window_s=window_s, n=n)
    else:
        data = _api_get("/api/agent/correlation", {"window_s": str(window_s), "n": str(n)})
    return json.dumps(data, indent=2)


@tool
def get_recent_activity(
    n: int = 30,
    agent_type: str | None = None,
    alerts_only: bool = False,
) -> str:
    """Get a unified chronological feed of the most recent events across all
    agents and nodes. Use to answer 'what happened recently?' or 'show me
    the latest alerts'.

    Args:
        n: Max events to return (default 30, max 200)
        agent_type: Filter to 'maintenance' or 'power' (optional)
        alerts_only: If True, show only alert events
    """
    if _is_embedded():
        data = _data_accessors["recent_activity"](n=n, agent_type=agent_type, alerts_only=alerts_only)
    else:
        params = {"n": str(n), "alerts_only": str(alerts_only).lower()}
        if agent_type:
            params["agent_type"] = agent_type
        data = _api_get("/api/agent/recent_activity", params)
    return json.dumps(data, indent=2)


@tool
def search_readings(
    agent_type: str,
    field: str,
    min_val: float | None = None,
    max_val: float | None = None,
    node_id: str | None = None,
    n: int = 20,
) -> str:
    """Search and filter readings by sensor value thresholds. Use for questions
    like 'find readings where tool_wear > 200' or 'show high-probability anomalies'.

    Args:
        agent_type: 'maintenance' or 'power'
        field: Sensor field to filter on (e.g. 'tool_wear', 'torque', 'probability', 'r1_vh')
        min_val: Minimum value inclusive (optional)
        max_val: Maximum value inclusive (optional)
        node_id: Filter to a specific node (optional)
        n: Max results (default 20, max 100)
    """
    if _is_embedded():
        data = _data_accessors["search_readings"](
            agent_type=agent_type, field=field,
            min_val=min_val, max_val=max_val,
            node_id=node_id, n=n,
        )
    else:
        params = {"agent_type": agent_type, "field": field, "n": str(n)}
        if min_val is not None:
            params["min_val"] = str(min_val)
        if max_val is not None:
            params["max_val"] = str(max_val)
        if node_id:
            params["node_id"] = node_id
        data = _api_get("/api/agent/search_readings", params)
    return json.dumps(data, indent=2)


# ─────────────────────────────────────────────
# All tools
# ─────────────────────────────────────────────

ALL_TOOLS = [
    get_system_status,
    get_node_detail,
    investigate_failures,
    get_sensor_summary,
    compare_nodes,
    cross_agent_correlation,
    get_recent_activity,
    search_readings,
]


# ─────────────────────────────────────────────
# Models  (lazy-initialised so import doesn't fail without key)
# ─────────────────────────────────────────────

_sonnet: ChatOpenRouter | None = None
_opus: ChatOpenRouter | None = None


def _get_sonnet() -> ChatOpenRouter:
    global _sonnet
    if _sonnet is None:
        _sonnet = ChatOpenRouter(model=FAST_MODEL, temperature=0.1, max_tokens=4096)
    return _sonnet


def _get_opus() -> ChatOpenRouter:
    global _opus
    if _opus is None:
        _opus = ChatOpenRouter(model=POWER_MODEL, temperature=0.2, max_tokens=8192)
    return _opus


# ─────────────────────────────────────────────
# Dynamic Model Selection Middleware
# ─────────────────────────────────────────────

def _needs_complex_reasoning(state: dict) -> bool:
    """Determine if the conversation warrants the more powerful model."""
    messages = state.get("messages", [])

    from langchain_core.messages import ToolMessage as CoreToolMessage
    for msg in messages:
        if isinstance(msg, CoreToolMessage) and msg.name in COMPLEX_TOOLS:
            return True

    if len(messages) > 10:
        return True

    if messages:
        from langchain_core.messages import HumanMessage
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human = msg.content.lower()
                escalation_keywords = [
                    "correlat", "investigat", "root cause", "why",
                    "compare", "false positive", "false negative",
                    "missed", "degraded", "explain", "analyze",
                    "deep dive", "pattern", "anomal",
                ]
                if any(kw in last_human for kw in escalation_keywords):
                    return True
                break

    return False


@wrap_model_call
def dynamic_model_selection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Switch between Sonnet (fast) and Opus (powerful) based on complexity."""
    if _needs_complex_reasoning(request.state):
        selected = _get_opus()
    else:
        selected = _get_sonnet()
    return handler(request.override(model=selected))


# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert anomaly investigation agent for a multi-agent industrial \
monitoring system. The system monitors two domains:

  • Maintenance agents — detect equipment failures using sensor data \
    (tool wear, torque, temperature, rotational speed, etc.)
  • Power agents — detect cyber-attacks on a power grid using relay \
    voltage readings (R1-R4 voltage harmonics, relay status, etc.)

Your primary role is FAILURE INVESTIGATION & CORRELATION. When a user asks \
about the system, follow this investigation protocol:

1. **Orient** — Start with `get_system_status` to understand the big picture.
2. **Triage** — If there are degraded nodes or alerts, use `investigate_failures` \
   to pull the actual failure events with full sensor context.
3. **Deep-dive** — Use `get_node_detail` on specific nodes to check trends, \
   and `get_sensor_summary` to understand normal vs anomalous sensor ranges.
4. **Cross-reference** — Use `cross_agent_correlation` to check if maintenance \
   failures align temporally with power grid attacks.
5. **Pinpoint** — Use `search_readings` to find specific anomalous readings, \
   and `compare_nodes` to identify which nodes diverge from peers.

Always explain your reasoning. When you find correlated events, explain the \
potential causal relationship. Distinguish between true positives, false \
positives, and missed detections. Provide actionable recommendations.

Be concise but thorough. Use data from the tools to back up every claim.\
"""


# ─────────────────────────────────────────────
# Agent Factory + Singleton for embedded mode
# ─────────────────────────────────────────────

_agent_instance = None
_agent_checkpointer = None


def create_investigation_agent(checkpointer=None):
    """Build and return a new investigation agent."""
    return create_agent(
        model=_get_sonnet(),
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[dynamic_model_selection],
        checkpointer=checkpointer,
    )


def get_shared_agent():
    """Return a singleton agent instance for use in the embedded API.
    Uses InMemorySaver so conversations persist across requests."""
    global _agent_instance, _agent_checkpointer
    if _agent_instance is None:
        _agent_checkpointer = InMemorySaver()
        _agent_instance = create_investigation_agent(checkpointer=_agent_checkpointer)
    return _agent_instance


# ─────────────────────────────────────────────
# Streaming SSE generator  (used by API endpoint)
# ─────────────────────────────────────────────

def stream_agent_response(
    message: str,
    thread_id: str = "default",
) -> Generator[str, None, None]:
    """Stream the agent's response as Server-Sent Events.

    Yields SSE-formatted strings:
        event: token     data: {"content": "..."}
        event: tool_call data: {"tool": "...", "args": {...}}
        event: tool_result data: {"tool": "...", "preview": "..."}
        event: model_used data: {"model": "sonnet|opus"}
        event: done      data: {"full_response": "..."}
        event: error     data: {"error": "..."}

    The dashboard frontend connects to POST /api/agent/chat with
    Accept: text/event-stream and reads these events.
    """
    agent = get_shared_agent()
    full_response = []

    try:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["messages", "updates"],
            version="v2",
        ):
            if chunk["type"] == "messages":
                token, metadata = chunk["data"]
                if isinstance(token, AIMessageChunk) and token.text:
                    full_response.append(token.text)
                    yield f"event: token\ndata: {json.dumps({'content': token.text})}\n\n"

            elif chunk["type"] == "updates":
                for step, data in chunk["data"].items():
                    last_msg = data["messages"][-1]

                    if step == "tools":
                        tool_name = getattr(last_msg, "name", "tool")
                        content_str = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                        preview = content_str[:300]
                        yield f"event: tool_result\ndata: {json.dumps({'tool': tool_name, 'preview': preview})}\n\n"

                    elif step == "model":
                        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                yield f"event: tool_call\ndata: {json.dumps({'tool': tc['name'], 'args': tc['args']})}\n\n"

        # Final assembled response
        yield f"event: done\ndata: {json.dumps({'full_response': ''.join(full_response)})}\n\n"

    except Exception as e:
        yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"


# ─────────────────────────────────────────────
# Standalone Interactive CLI  (unchanged)
# ─────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
DIM    = "\033[2m"


def run_interactive(api_url: str):
    """Run the agent in an interactive REPL with token-level streaming."""
    global API_BASE_URL, _http_client
    API_BASE_URL = api_url
    _http_client = httpx.Client(base_url=API_BASE_URL, timeout=15.0)

    checkpointer = InMemorySaver()
    agent = create_investigation_agent(checkpointer=checkpointer)
    thread_id = "investigation-session-1"

    print(f"\n{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  Anomaly Investigation Agent{RESET}")
    print(f"{BOLD}{CYAN}{'─' * 60}{RESET}")
    print(f"{DIM}  API: {api_url}{RESET}")
    print(f"{DIM}  Models: Sonnet (fast) ↔ Opus (complex){RESET}")
    print(f"{DIM}  Type 'quit' to exit, 'reset' to clear memory{RESET}\n")

    while True:
        try:
            user_input = input(f"{GREEN}You:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            thread_id = f"investigation-session-{id(object())}"
            print(f"{DIM}  Memory cleared. New session started.{RESET}\n")
            continue

        print(f"\n{YELLOW}Agent:{RESET} ", end="", flush=True)
        is_streaming_text = False
        try:
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode=["messages", "updates"],
                version="v2",
            ):
                if chunk["type"] == "messages":
                    token, metadata = chunk["data"]
                    if isinstance(token, AIMessageChunk):
                        if token.text:
                            if not is_streaming_text:
                                is_streaming_text = True
                            print(token.text, end="", flush=True)

                elif chunk["type"] == "updates":
                    for step, data in chunk["data"].items():
                        last_msg = data["messages"][-1]

                        if step == "tools":
                            if is_streaming_text:
                                print()
                                is_streaming_text = False
                            tool_name = getattr(last_msg, "name", "tool")
                            content_str = last_msg.content if isinstance(last_msg.content, str) else str(last_msg.content)
                            preview = content_str[:150]
                            print(f"  {DIM}🔧 {tool_name} → {preview}...{RESET}", flush=True)

                        elif step == "model":
                            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                                if is_streaming_text:
                                    print()
                                    is_streaming_text = False
                                for tc in last_msg.tool_calls:
                                    args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                                    print(f"  {DIM}📡 Calling {tc['name']}({args_str}){RESET}", flush=True)

        except httpx.ConnectError:
            print(f"\n  {BOLD}Connection error — is the API running at {api_url}?{RESET}")
        except Exception as e:
            print(f"\n  {BOLD}Error: {e}{RESET}")

        print("\n")


def run_single_query(query: str, api_url: str):
    """Run a single query with token-level streaming."""
    global API_BASE_URL, _http_client
    API_BASE_URL = api_url
    _http_client = httpx.Client(base_url=API_BASE_URL, timeout=15.0)

    agent = create_investigation_agent()

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        if chunk["type"] == "messages":
            token, metadata = chunk["data"]
            if isinstance(token, AIMessageChunk) and token.text:
                print(token.text, end="", flush=True)
        elif chunk["type"] == "updates":
            for step, data in chunk["data"].items():
                last_msg = data["messages"][-1]
                if step == "tools":
                    tool_name = getattr(last_msg, "name", "tool")
                    print(f"\n  🔧 {tool_name} done", flush=True)
                elif step == "model" and isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        args_str = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
                        print(f"\n  📡 Calling {tc['name']}({args_str})", flush=True)
    print()


# ─────────────────────────────────────────────
# Entrypoint  (standalone CLI only)
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model anomaly investigation agent"
    )
    parser.add_argument(
        "--query", "-q",
        type=str, default=None,
        help="Run a single query instead of interactive mode",
    )
    parser.add_argument(
        "--api-url",
        type=str, default="http://127.0.0.1:5000",
        help="Base URL of the dashboard API (default: http://127.0.0.1:5000)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable is not set.")
        print("Get your key at https://openrouter.ai/settings/keys")
        return

    if args.query:
        run_single_query(args.query, args.api_url)
    else:
        run_interactive(args.api_url)


if __name__ == "__main__":
    main()