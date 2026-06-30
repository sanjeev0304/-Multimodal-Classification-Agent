"""
streamlit_app.py
-----------------
Event-driven Streamlit UI for the Multimodal Classification Agent.

The user SELECTS the input type (Document / Sensor Data / Network Packet),
provides the data, and clicking "Classify" calls the matching MCP tool
(each backed by Gemini) and displays the structured result.

No LLM auto-routing: the user's selection determines which MCP tool runs.

Run:
    streamlit run streamlit_app.py
"""

import asyncio
import json
import os
import sys

import streamlit as st
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# ---------------------------------------------------------------------------
# Modality registry: maps each input type to its MCP tool, the tool's
# argument name, and the field names in that tool's structured JSON result.
# ---------------------------------------------------------------------------
MODALITIES = {
    "Document": {
        "icon": "📄",
        "tool": "classify_document",
        "arg_name": "text",
        "placeholder": "Paste a document's text here (invoice, report, contract, manual)...",
        "example": (
            "INVOICE #4471. Bill to: Acme Corp. Item: Cloud subscription (annual). "
            "Amount due: $1,200. Payment due date: 30 June 2026."
        ),
        "class_field": "category",
        "tags_field": "key_topics",
    },
    "Sensor Data": {
        "icon": "📡",
        "tool": "classify_sensor_data",
        "arg_name": "readings",
        "placeholder": "Paste sensor readings here — labeled (temperature=92C) or plain numbers (92, 8.4, 1.2)...",
        "example": (
            "Readings: temperature=92C, vibration=8.4mm/s (baseline 1.2mm/s), "
            "pressure=stable. Bearing temperature rising steadily over the last 10 minutes."
        ),
        "class_field": "state",
        "tags_field": "indicators",
        "type_field": "sensor_type",
    },
    "Network Packet": {
        "icon": "🌐",
        "tool": "classify_network_packet",
        "arg_name": "packet",
        "placeholder": "Paste a network packet / log summary here...",
        "example": (
            "TCP SYN flood detected from 14 source IPs targeting port 443, "
            "~9000 packets/sec, with no completed handshakes."
        ),
        "class_field": "classification",
        "tags_field": "signals",
    },
}


async def call_classifier(tool_name: str, arg_name: str, value: str) -> dict:
    """Spin up the MCP server, call exactly one tool, tear down.
    A fresh connection per click keeps this simple and reliable for a demo."""
    server = StdioServerParameters(
        command=sys.executable, args=["classification_mcp_server.py"]
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, {arg_name: value})
            text = result.content[0].text if result.content else "{}"
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return {"raw": text}


def run_async(coro):
    """Bridge: run one async MCP call from Streamlit's sync execution model."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multimodal Classification Agent", page_icon="🧠", layout="centered")

st.title("🧠 Multimodal Classification Agent")
st.caption("Select an input type, provide the data, and classify it via Gemini through an MCP tool.")

if not os.environ.get("GEMINI_API_KEY"):
    st.warning("GEMINI_API_KEY is not set. Add it to your .env file before classifying.", icon="⚠️")

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------------------------------------------------------------------
# Event 1: user selects the modality
# ---------------------------------------------------------------------------
modality_name = st.radio(
    "What type of data do you want to classify?",
    options=list(MODALITIES.keys()),
    format_func=lambda m: f"{MODALITIES[m]['icon']}  {m}",
    horizontal=True,
)
modality = MODALITIES[modality_name]
text_key = f"text_{modality_name}"

# ---------------------------------------------------------------------------
# Event 2: user provides input (typed, or pre-filled via "Use example")
# ---------------------------------------------------------------------------
col1, col2 = st.columns([5, 1])
with col2:
    if st.button("Use example", use_container_width=True):
        st.session_state[text_key] = modality["example"]

with col1:
    pass  # spacing only

user_input = st.text_area(
    "Input",
    placeholder=modality["placeholder"],
    height=160,
    key=text_key,
)

# ---------------------------------------------------------------------------
# Event 3: user clicks Classify
# ---------------------------------------------------------------------------
classify_clicked = st.button("Classify", type="primary", use_container_width=True)

if classify_clicked:
    if not user_input.strip():
        st.error("Please enter some input first (or click 'Use example').")
    else:
        with st.spinner(f"Classifying as {modality_name}..."):
            try:
                result = run_async(
                    call_classifier(modality["tool"], modality["arg_name"], user_input)
                )
                st.session_state.history.insert(
                    0, {"modality": modality_name, "input": user_input, "result": result}
                )
            except Exception as e:
                st.error(f"Classification failed: {e}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.subheader("Results")
    with top_r:
        if st.button("Clear", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    for entry in st.session_state.history:
        m = MODALITIES[entry["modality"]]
        result = entry["result"]
        predicted_class = result.get(m["class_field"], "Unknown")
        confidence = float(result.get("confidence_score", 0.0) or 0.0)
        tags = result.get(m["tags_field"], [])
        justification = result.get("justification", "")

        with st.container(border=True):
            top_l, top_r = st.columns([3, 1])
            with top_l:
                type_field = m.get("type_field")
                inferred_type = result.get(type_field) if type_field else None
                if inferred_type:
                    st.markdown(
                        f"**{m['icon']} {entry['modality']} — {inferred_type}** &rarr; **{predicted_class}**"
                    )
                else:
                    st.markdown(f"**{m['icon']} {entry['modality']}** &rarr; **{predicted_class}**")
            with top_r:
                st.metric("Confidence", f"{confidence * 100:.0f}%", label_visibility="collapsed")
            st.progress(min(max(confidence, 0.0), 1.0))
            if tags:
                st.caption(" • ".join(f"`{t}`" for t in tags))
            if justification:
                st.write(justification)
            with st.expander("Raw JSON"):
                st.json(result)