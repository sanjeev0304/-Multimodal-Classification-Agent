"""
streamlit_app.py
-----------------
Unified Multimodal Classification Agent UI.

Two modes, selectable from the sidebar:
  🔬 Classification   — manual input (Document / Sensor Data / Network Packet / Image upload)
  📹 Real-time Scanner — live camera feed with automatic tool scan every 10 seconds

Run:
    streamlit run streamlit_app.py
"""

import asyncio
import base64
import io
import json
import os
import sys
import threading
import time

import av
import cv2
import streamlit as st
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from PIL import Image
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL          = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
SCAN_INTERVAL  = 10  # seconds between auto-scans in real-time mode

# Direct Gemini client — used by the real-time scanner (no MCP subprocess overhead)
gemini_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
TEXT_MODALITIES = {
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
        "placeholder": "Paste sensor readings — labeled (temperature=92C) or plain numbers (92, 8.4, 1.2)...",
        "example": (
            "Readings: temperature=92C, vibration=8.4mm/s (baseline 1.2mm/s), "
            "pressure=stable. Bearing temperature rising over the last 10 minutes."
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

ALL_MODALITIES = list(TEXT_MODALITIES.keys()) + ["🔧 Image (Tools)"]

CATEGORY_ICONS = {
    "Hand Tool":            "🔨",
    "Power Tool":           "⚡",
    "Cutting Tool":         "✂️",
    "Measuring Instrument": "📏",
    "Fastening Tool":       "🔩",
    "Other":                "🔧",
}
CONDITION_COLORS = {
    "Good":    "green",
    "Worn":    "orange",
    "Damaged": "red",
    "Unknown": "gray",
}

VISION_SYSTEM_PROMPT = (
    "You are an expert industrial tool recognition system deployed on a factory floor. "
    "Tools may be scattered, worn, partially obscured, or have faded labels — handle all of these. "
    "Identify every industrial tool visible in the image. For each tool provide:\n"
    "  - tool_name: specific name (e.g. Open-End Wrench, Phillips Screwdriver, Angle Grinder)\n"
    "  - category: Hand Tool / Power Tool / Cutting Tool / Measuring Instrument / Fastening Tool / Other\n"
    "  - condition: Good (clean, intact) / Worn (scratches, minor rust, rounded edges) / "
    "Damaged (cracks, severe rust, broken parts) / Unknown (too obscured to assess)\n"
    "  - confidence_score: 0.0 to 1.0\n\n"
    "Respond with valid JSON only:\n"
    '{"tools_detected":[{"tool_name":str,"category":str,"condition":str,"confidence_score":float}],'
    '"total_count":int,"scene_description":str,"justification":str}\n'
    "If no tools are visible return tools_detected:[] and total_count:0."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def prepare_image(image_bytes: bytes, name: str = "") -> tuple[bytes, str, str]:
    """Resize to max 1024px, detect media type, return (bytes, media_type, b64)."""
    img = Image.open(io.BytesIO(image_bytes))
    fmt = img.format or ("PNG" if name.lower().endswith(".png") else "JPEG")
    media_type = "image/png" if fmt == "PNG" else "image/jpeg"
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        image_bytes = buf.getvalue()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_bytes, media_type, b64


def bgr_frame_to_b64(bgr_frame) -> str:
    """Convert OpenCV BGR frame to JPEG base64 (max 1024px)."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_gemini_vision(b64_image: str, media_type: str = "image/jpeg") -> dict:
    """Direct Gemini vision call — used by both the image upload and real-time scanner."""
    resp = gemini_client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        max_tokens=4096,        # raised from 1024 — prevents truncation on busy scenes
        temperature=0.0,
        messages=[
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{media_type};base64,{b64_image}"}},
                    {"type": "text",
                     "text": "Identify and classify all industrial tools in this image."},
                ],
            },
        ],
    )
    raw = resp.choices[0].message.content
    if not raw or not raw.strip():
        raise ValueError("Gemini returned empty content.")

    # Safety net: if the response was still truncated, recover the largest
    # valid JSON object by trimming after the last complete closing brace.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        last_brace = raw.rfind("}")
        if last_brace != -1:
            try:
                return json.loads(raw[: last_brace + 1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse Gemini response as JSON: {raw[:200]}")


# ── MCP helper (classification mode only) ───────────────────────────────────
async def _call_mcp_tool(tool_name: str, tool_args: dict) -> dict:
    server = StdioServerParameters(
        command=sys.executable, args=["classification_mcp_server.py"]
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)
            text = result.content[0].text if result.content else "{}"
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return {"raw": text}


def run_async(coro):
    return asyncio.run(coro)


# ── Real-time scanner helpers ────────────────────────────────────────────────
class FrameCapture(VideoProcessorBase):
    """Background thread: captures latest video frame."""
    def __init__(self):
        self._frame = None
        self._lock  = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            self._frame = img.copy()
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None


def do_scan(ctx) -> bool:
    """Grab latest frame and call Gemini. Returns True on success."""
    if not ctx.video_processor:
        return False
    frame = ctx.video_processor.get_frame()
    if frame is None:
        return False
    b64    = bgr_frame_to_b64(frame)
    result = call_gemini_vision(b64)   # reuses the same fixed function
    st.session_state.rt_last_result    = result
    st.session_state.rt_last_scan_time = time.time()
    st.session_state.rt_scan_count    += 1
    return True


# ---------------------------------------------------------------------------
# Result card renderers (shared between both modes)
# ---------------------------------------------------------------------------
def render_tool_cards(result: dict):
    """Render per-tool cards from an image classification result."""
    tools = result.get("tools_detected", [])
    total = result.get("total_count", len(tools))
    scene = result.get("scene_description", "")

    if scene:
        st.caption(f"📍 {scene}")

    if not tools:
        st.info("No tools detected.")
        return

    for tool in tools:
        name      = tool.get("tool_name", "Unknown")
        category  = tool.get("category", "Other")
        condition = tool.get("condition", "Unknown")
        conf      = float(tool.get("confidence_score", 0.0))
        icon      = CATEGORY_ICONS.get(category, "🔧")
        color     = CONDITION_COLORS.get(condition, "gray")

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{icon} {name}**")
                st.caption(f"{category}  •  :{color}[**{condition}**]")
            with c2:
                st.metric("", f"{conf * 100:.0f}%", label_visibility="collapsed")
            st.progress(min(max(conf, 0.0), 1.0))


def render_text_result(entry: dict):
    """Render a text modality result card."""
    m          = TEXT_MODALITIES[entry["modality"]]
    result     = entry["result"]
    pred_class = result.get(m["class_field"], "Unknown")
    confidence = float(result.get("confidence_score", 0.0) or 0.0)
    tags       = result.get(m["tags_field"], [])
    just       = result.get("justification", "")

    with st.container(border=True):
        tl, tr = st.columns([3, 1])
        with tl:
            inferred = result.get(m.get("type_field")) if m.get("type_field") else None
            if inferred:
                st.markdown(f"**{m['icon']} {entry['modality']} — {inferred}** → **{pred_class}**")
            else:
                st.markdown(f"**{m['icon']} {entry['modality']}** → **{pred_class}**")
        with tr:
            st.metric("", f"{confidence * 100:.0f}%", label_visibility="collapsed")
        st.progress(min(max(confidence, 0.0), 1.0))
        if tags:
            st.caption(" • ".join(f"`{t}`" for t in tags))
        if just:
            st.write(just)
        with st.expander("Raw JSON"):
            st.json(result)


def render_image_result(entry: dict):
    """Render an image upload result card with thumbnail."""
    result = entry["result"]
    tools  = result.get("tools_detected", [])
    total  = result.get("total_count", len(tools))

    with st.container(border=True):
        hl, hr = st.columns([3, 1])
        with hl:
            st.markdown(f"**🔧 Image (Tools)** — **{total} tool{'s' if total != 1 else ''} detected**")
        with hr:
            st.caption(f"{total} found")

        if entry.get("image_b64"):
            st.image(base64.b64decode(entry["image_b64"]), width=220)

        render_tool_cards(result)

        with st.expander("Raw JSON"):
            st.json(result)


# ===========================================================================
# Page setup
# ===========================================================================
st.set_page_config(
    page_title="Multimodal Classification Agent",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
for key, default in {
    "history":           [],
    "rt_last_scan_time": 0.0,
    "rt_last_result":    None,
    "rt_scan_count":     0,
    "rt_scan_error":     None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar — mode selector
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧠 Classification Agent")
    st.divider()
    mode = st.radio(
        "Mode",
        ["🔬 Classification", "📹 Real-time Scanner"],
        label_visibility="collapsed",
    )
    st.divider()
    if not GEMINI_API_KEY:
        st.warning("GEMINI_API_KEY not set in .env", icon="⚠️")
    else:
        st.success("API key loaded", icon="✅")
    st.caption(f"Model: `{MODEL}`")

# ---------------------------------------------------------------------------
# Auto-refresh ONLY in real-time mode
# (calling it conditionally prevents it from firing in classification mode)
# ---------------------------------------------------------------------------
if mode == "📹 Real-time Scanner":
    st_autorefresh(interval=SCAN_INTERVAL * 1000, key="scanner_refresh")

# ===========================================================================
# MODE 1 — Classification
# ===========================================================================
if mode == "🔬 Classification":
    st.title("🔬 Multimodal Classification Agent")
    st.caption("Select an input type, provide data, and classify it via Gemini through an MCP tool.")

    st.divider()

    # ── Modality selector ────────────────────────────────────────────────────
    modality_name = st.radio(
        "What type of data do you want to classify?",
        options=ALL_MODALITIES,
        horizontal=True,
    )
    st.divider()

    # ── IMAGE modality ───────────────────────────────────────────────────────
    if modality_name == "🔧 Image (Tools)":
        st.subheader("🔧 Industrial Tool Detection")
        st.caption("Upload or capture a photo of industrial tools.")

        cam_tab, upload_tab = st.tabs(["📷 Camera", "📁 Upload"])
        image_file, image_source = None, ""

        with cam_tab:
            cam_img = st.camera_input("Take a photo of the tools")
            if cam_img:
                image_file   = cam_img.getvalue()
                image_source = "camera.jpg"

        with upload_tab:
            up_img = st.file_uploader(
                "Upload an image", type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            if up_img:
                image_file   = up_img.getvalue()
                image_source = up_img.name

        if image_file:
            st.image(image_file, caption="Image to classify", use_column_width=True)

        if st.button("Detect Tools", type="primary", use_container_width=True,
                     disabled=image_file is None):
            with st.spinner("Analysing image for industrial tools..."):
                try:
                    _, media_type, b64 = prepare_image(image_file, image_source)
                    result = call_gemini_vision(b64, media_type)
                    st.session_state.history.insert(0, {
                        "modality":  "🔧 Image (Tools)",
                        "image_b64": base64.b64encode(image_file).decode(),
                        "result":    result,
                    })
                    st.rerun()
                except Exception as e:
                    st.error(f"Detection failed: {e}")

    # ── TEXT modalities ──────────────────────────────────────────────────────
    else:
        modality = TEXT_MODALITIES[modality_name]
        text_key = f"text_{modality_name}"

        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button("Use example", use_container_width=True):
                st.session_state[text_key] = modality["example"]

        user_input = st.text_area(
            "Input", placeholder=modality["placeholder"],
            height=160, key=text_key,
        )

        if st.button("Classify", type="primary", use_container_width=True):
            if not user_input.strip():
                st.error("Please enter some input first (or click 'Use example').")
            else:
                with st.spinner(f"Classifying as {modality_name}..."):
                    try:
                        result = run_async(
                            _call_mcp_tool(modality["tool"],
                                           {modality["arg_name"]: user_input})
                        )
                        st.session_state.history.insert(0, {
                            "modality": modality_name,
                            "input":    user_input,
                            "result":   result,
                        })
                    except Exception as e:
                        st.error(f"Classification failed: {e}")

    # ── History ──────────────────────────────────────────────────────────────
    if st.session_state.history:
        st.divider()
        hl, hr = st.columns([4, 1])
        with hl:
            st.subheader("Results")
        with hr:
            if st.button("Clear", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        for entry in st.session_state.history:
            if entry["modality"] == "🔧 Image (Tools)":
                render_image_result(entry)
            else:
                render_text_result(entry)

# ===========================================================================
# MODE 2 — Real-time Scanner
# ===========================================================================
elif mode == "📹 Real-time Scanner":
    st.title("📹 Real-time Industrial Tool Scanner")
    st.caption(
        f"Live camera feed — automatically scans every **{SCAN_INTERVAL} seconds**. "
        "Click **Scan Now** to trigger an immediate scan."
    )
    st.divider()

    cam_col, result_col = st.columns([3, 2], gap="large")

    # ── Camera feed ──────────────────────────────────────────────────────────
    with cam_col:
        st.subheader("📷 Live Feed")
        ctx = webrtc_streamer(
            key="tool-scanner",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FrameCapture,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        st.write("")
        if st.button("🔍 Scan Now", type="primary", use_container_width=True):
            if not ctx.video_processor:
                st.warning("Allow camera access in the browser first.")
            else:
                with st.spinner("Scanning frame..."):
                    try:
                        do_scan(ctx)
                        st.session_state.rt_scan_error = None
                    except Exception as e:
                        st.session_state.rt_scan_error = str(e)
                st.rerun()

    # ── Results panel ─────────────────────────────────────────────────────────
    with result_col:
        st.subheader("🔍 Scan Results")

        # Auto-scan trigger
        now        = time.time()
        time_since = now - st.session_state.rt_last_scan_time
        time_until = max(0.0, SCAN_INTERVAL - time_since)

        if time_since >= SCAN_INTERVAL and ctx.video_processor:
            with st.spinner("Auto-scanning..."):
                try:
                    do_scan(ctx)
                    st.session_state.rt_scan_error = None
                except Exception as e:
                    st.session_state.rt_scan_error = str(e)
            st.rerun()

        # Status bar
        if st.session_state.rt_scan_error:
            st.error(f"Last scan failed: {st.session_state.rt_scan_error}")
        elif st.session_state.rt_last_scan_time == 0:
            st.info("⏳ Waiting for first scan…")
        else:
            ago = int(time.time() - st.session_state.rt_last_scan_time)
            st.success(
                f"✅ Scan **#{st.session_state.rt_scan_count}** — "
                f"scanned **{ago}s ago** — "
                f"next in **{int(time_until)}s**"
            )

        # Countdown progress bar
        st.progress(min(max(1.0 - (time_until / SCAN_INTERVAL), 0.0), 1.0))
        st.write("")

        # Results
        result = st.session_state.rt_last_result
        if result is None:
            st.info("Point the camera at industrial tools.\nFirst scan runs automatically.")
        else:
            tools   = result.get("tools_detected", [])
            total   = result.get("total_count", len(tools))
            damaged = sum(1 for t in tools if t.get("condition") == "Damaged")

            m1, m2 = st.columns(2)
            m1.metric("Tools in frame", total)
            m2.metric("Damaged", damaged,
                      delta=f"⚠️ {damaged}" if damaged else None)

            st.write("")
            render_tool_cards(result)

            with st.expander("Raw JSON from last scan"):
                st.json(result)
