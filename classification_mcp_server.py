"""
classification_mcp_server.py
----------------------------
MCP server that exposes per-modality classification *tools*, each backed by
Google Gemini (via its OpenAI-compatible endpoint). This generalizes the
professor's `classify_medical_report` pattern (LLM + strict Pydantic schema)
into three discoverable MCP tools:

    - classify_document        -> Invoice / Report / Contract / Manual
    - classify_sensor_data     -> sensor_type (inferred) + Normal / Fault
    - classify_network_packet  -> Normal / Suspicious / Priority

Run it directly (stdio transport) or let classification_agent.py spawn it.
"""

import os
from typing import List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import OpenAI
from mcp.server.fastmcp import FastMCP

# Load GEMINI_API_KEY (and optional GEMINI_MODEL) from a local .env file.
load_dotenv()

# ---------------------------------------------------------------------------
# Gemini client, accessed through its OpenAI-compatible endpoint.
# Only base_url + api_key + model differ from a normal OpenAI call.
# ---------------------------------------------------------------------------
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# "gemini-2.5-flash" is fast and sits on Gemini's free tier — good default
# for development/demo. Swap to "gemini-2.5-pro" for harder cases if needed.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _completion_kwargs() -> dict:
    """Shared call settings. Gemini supports OpenAI-style JSON mode."""
    return {
        "model": MODEL,
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,  # keep JSON from truncating mid-string
        "temperature": 0.0,  # deterministic classification
    }


mcp = FastMCP("ClassificationServer")


# ---------------------------------------------------------------------------
# Structured output schemas (one per modality).
# ---------------------------------------------------------------------------
class DocumentResult(BaseModel):
    category: Literal["Invoice", "Report", "Contract", "Manual"] = Field(
        description="Primary document category."
    )
    confidence_score: float = Field(ge=0.0, le=1.0)
    key_topics: List[str] = Field(description="Key terms that drove the decision.")
    justification: str = Field(description="One-sentence reasoning for the category.")


class SensorResult(BaseModel):
    sensor_type: str = Field(
        description="Best-guess sensor type inferred from the readings "
        "(e.g. Temperature, Vibration, Pressure, Humidity, RPM/Speed, "
        "Voltage, Current, Flow Rate, or Unknown if it cannot be determined)."
    )
    state: Literal["Normal", "Fault"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    indicators: List[str] = Field(description="Readings that drove the decision.")
    justification: str


class NetworkResult(BaseModel):
    classification: Literal["Normal", "Suspicious", "Priority"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    signals: List[str] = Field(description="Traffic signals that drove the decision.")
    justification: str


# ---------------------------------------------------------------------------
# Generic R1 classify helper.
# ---------------------------------------------------------------------------
def _classify(system_prompt: str, user_text: str, schema: type[BaseModel]) -> dict:
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        **_completion_kwargs(),
    )
    raw = resp.choices[0].message.content
    if not raw or not raw.strip():
        # JSON mode can occasionally return empty content; surface it clearly.
        raise ValueError("Model returned empty content. Retry or adjust the prompt.")
    return schema.model_validate_json(raw).model_dump()


# ---------------------------------------------------------------------------
# Tools (discoverable by the agent over MCP).
# ---------------------------------------------------------------------------
@mcp.tool()
def classify_document(text: str) -> dict:
    """Classify a business document into Invoice, Report, Contract, or Manual.
    Use this for any document / text input. Returns category, confidence_score
    (0-1), key_topics, and justification."""
    system = (
        "You are an expert document analyst. Classify the document into exactly one of: "
        "Invoice, Report, Contract, Manual. Respond with valid json only, matching this schema: "
        '{"category": str, "confidence_score": float, "key_topics": [str], "justification": str}. '
        'Example json: {"category": "Invoice", "confidence_score": 0.94, '
        '"key_topics": ["billing", "amount due"], '
        '"justification": "Contains line items and a total amount due."}'
    )
    return _classify(system, text, DocumentResult)


@mcp.tool()
def classify_sensor_data(readings: str) -> dict:
    """Classify sensor/time-series readings as Normal or Fault, and infer the
    sensor type (e.g. Temperature, Vibration, Pressure). Works even when the
    input is plain numbers with no labels or units. Use this for any sensor
    or machine-telemetry input. Returns sensor_type, state, confidence_score
    (0-1), indicators, and justification."""
    system = (
        "You are an expert industrial monitoring analyst. You may receive labeled readings "
        "(with units) or PLAIN NUMBERS with no labels or units at all — handle both. "
        "\n\n"
        "Step 1 — Infer the sensor type from the numeric pattern, using typical real-world "
        "ranges as a guide:\n"
        "  - Temperature (°C): roughly 0-150, machine operating range ~20-100\n"
        "  - Vibration (mm/s): roughly 0-20, healthy baseline ~0.5-2\n"
        "  - Pressure (bar): roughly 0-10 (or psi: 0-150)\n"
        "  - Humidity (%): 0-100\n"
        "  - RPM / rotational speed: roughly hundreds to tens of thousands\n"
        "  - Voltage (V): roughly 0-500\n"
        "  - Current (A): roughly 0-100\n"
        "  - Flow rate: varies, often tens to hundreds\n"
        "If the input already names the sensor (e.g. 'temperature=92'), trust that label. "
        "If truly ambiguous, make the best inference from magnitude/precision and say so "
        "in the justification; use sensor_type='Unknown' only if no reasonable guess fits.\n\n"
        "Step 2 — Using that inferred type's typical healthy range, classify the reading(s) as "
        "exactly one of: Normal, Fault.\n\n"
        "Respond with valid json only, matching this schema: "
        '{"sensor_type": str, "state": str, "confidence_score": float, '
        '"indicators": [str], "justification": str}. '
        'Example with units: {"sensor_type": "Vibration", "state": "Fault", '
        '"confidence_score": 0.91, "indicators": ["vibration 8.4mm/s vs 1.2 baseline", '
        '"bearing temp rising"], "justification": "Vibration and bearing temperature far '
        'exceed normal range."} '
        'Example with plain numbers (no units), input "92, 8.4, 1.2": '
        '{"sensor_type": "Vibration", "state": "Fault", "confidence_score": 0.74, '
        '"indicators": ["value 92 is far above typical vibration mm/s range", '
        '"large spread between readings suggests an anomaly"], '
        '"justification": "No units were given; based on magnitude and spread, these most '
        'resemble vibration readings, and 92 is far outside a healthy range."}'
    )
    return _classify(system, readings, SensorResult)


@mcp.tool()
def classify_network_packet(packet: str) -> dict:
    """Classify a network packet/log summary as Normal, Suspicious, or Priority.
    Use this for any network traffic, packet, or log input. Returns classification,
    confidence_score (0-1), signals, and justification."""
    system = (
        "You are an expert network security analyst. Classify the traffic as exactly one of: "
        "Normal, Suspicious, Priority. Respond with valid json only, matching this schema: "
        '{"classification": str, "confidence_score": float, "signals": [str], "justification": str}. '
        'Example json: {"classification": "Suspicious", "confidence_score": 0.88, '
        '"signals": ["SYN flood pattern", "no completed handshakes"], '
        '"justification": "High-rate SYN packets from many IPs indicate a possible DoS attempt."}'
    )
    return _classify(system, packet, NetworkResult)


# ---------------------------------------------------------------------------
# Resource (dynamic context the agent can read).
# ---------------------------------------------------------------------------
@mcp.resource("info://server_status")
def get_server_status() -> str:
    return (
        f"ClassificationServer online. Model={MODEL} (via Gemini OpenAI-compatible endpoint). "
        "Tools: classify_document, classify_sensor_data, classify_network_packet."
    )


if __name__ == "__main__":
    # Standard input/output (stdio) transport.
    mcp.run()