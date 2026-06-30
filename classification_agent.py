"""
classification_agent.py
-----------------------
Agent (MCP client) that connects to classification_mcp_server.py, discovers its
tools, and uses Gemini *function calling* to ROUTE each input to the correct
specialist classification tool. The chosen tool (itself Gemini-backed) returns
the structured verdict, which the agent returns directly.

Flow per input:
    input -> Gemini picks a tool (routing) -> MCP tool classifies -> structured result

This is "Method 1 (LLM via MCP)" of the multimodal classification project.
"""

import os
import json
import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load GEMINI_API_KEY (and optional GEMINI_MODEL) from a local .env file.
load_dotenv()

# ---------------------------------------------------------------------------
# Gemini client, accessed through its OpenAI-compatible endpoint.
# ---------------------------------------------------------------------------
client = AsyncOpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _kwargs() -> dict:
    return {"model": MODEL, "max_tokens": 1024, "temperature": 0.0}


ROUTER_SYSTEM = (
    "You are a classification router. You receive a single input and must call exactly "
    "one of the available classification tools that matches the input type: document/text, "
    "sensor/telemetry readings, or network traffic/log. Always call a tool; never answer "
    "directly."
)


async def classify(session: ClientSession, user_input: str) -> dict:
    """Route one input to the correct MCP classification tool and return its result."""
    # 1) Discover MCP tools and expose them to the model in OpenAI tool format.
    tools_resp = await session.list_tools()
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema,
            },
        }
        for t in tools_resp.tools
    ]

    # 2) Let Gemini choose the right tool.
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": user_input},
    ]
    resp = await client.chat.completions.create(
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
        **_kwargs(),
    )
    msg = resp.choices[0].message

    # Fallback: model answered without routing (rare).
    if not msg.tool_calls:
        return {"routed_tool": None, "raw_answer": msg.content}

    # 3) Execute the chosen specialist tool over MCP.
    call = msg.tool_calls[0]
    try:
        args = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        args = {}
    tool_result = await session.call_tool(call.function.name, args)

    # 4) Parse the structured result returned by the tool.
    text = tool_result.content[0].text if tool_result.content else "{}"
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = {"raw": text}

    return {"routed_tool": call.function.name, "result": parsed}


async def main():
    server = StdioServerParameters(
        command="python", args=["classification_mcp_server.py"]
    )

    samples = {
        "Document": (
            "INVOICE #4471. Bill to: Acme Corp. Item: Cloud subscription (annual). "
            "Amount due: $1,200. Payment due date: 30 June 2026."
        ),
        "Sensor": (
            "Readings: temperature=92C, vibration=8.4mm/s (baseline 1.2mm/s), "
            "pressure=stable. Bearing temperature rising steadily over the last 10 minutes."
        ),
        "Network": (
            "TCP SYN flood detected from 14 source IPs targeting port 443, "
            "~9000 packets/sec, with no completed handshakes."
        ),
    }

    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # (Optional) read the server status resource.
            status = await session.read_resource("info://server_status")
            print("Server:", status.contents[0].text if status.contents else status)

            for label, text in samples.items():
                print(f"\n=== {label} input ===")
                try:
                    out = await classify(session, text)
                    print(json.dumps(out, indent=2))
                except Exception as e:
                    print(f"Failed to classify {label} input: {e}")


if __name__ == "__main__":
    asyncio.run(main())