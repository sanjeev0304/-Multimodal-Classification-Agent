# 🧠 Multimodal Classification Agent

An AI-powered classification system for industrial and business data — documents, sensor telemetry, network traffic, and live camera tool detection — built on **Google Gemini** via **Model Context Protocol (MCP)**, with a unified **Streamlit** web interface.

---

## What It Does

One app, two modes, selectable from the sidebar:

### 🔬 Classification Mode
Manually classify any of four input types. Select the type, provide data, click Classify.

| Input Type | What you provide | What you get back |
|---|---|---|
| **📄 Document** | Paste text (invoice, report, contract, manual) | Category + confidence + key topics + reasoning |
| **📡 Sensor Data** | Readings with or without labels/units | Sensor type (inferred) + Normal/Fault + indicators |
| **🌐 Network Packet** | Traffic summary or log entry | Normal / Suspicious / Priority + signals + reasoning |
| **🔧 Image (Tools)** | Photo from camera or file upload | List of detected tools + category + condition |

### 📹 Real-time Scanner Mode
Live camera feed with automatic tool detection every 10 seconds. Designed for a fixed camera mounted over an industrial workbench.

- Scans every 10 seconds automatically — no user action needed
- **Scan Now** button for immediate on-demand scan
- Countdown progress bar showing time to next scan
- Per-tool result cards updating in real time
- Damaged tool counter (the key industrial safety signal)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI                           │
│                   (streamlit_app.py)                        │
│                                                             │
│  Sidebar ──► 🔬 Classification | 📹 Real-time Scanner       │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴─────────────┐
          │                          │
          ▼                          ▼
  Text / Image upload        Real-time camera
  → MCP Server (stdio)       → Gemini vision (direct)
          │
          ▼
  ┌───────────────────────────┐
  │   MCP Server (FastMCP)    │
  │ classify_document         │
  │ classify_sensor_data      │
  │ classify_network_packet   │
  │ classify_image_tools      │
  └───────────┬───────────────┘
              │
              ▼
       Google Gemini API
  (OpenAI-compatible endpoint)
```

**Key design decisions:**
- **No LLM routing** — the user's selection directly picks which MCP tool runs. No routing failures in live demos.
- **Direct Gemini call for real-time** — the scanner bypasses MCP to avoid spawning a subprocess every 10 seconds.
- **Pydantic schema per modality** — every tool enforces a strict output contract before returning, so the UI always receives well-formed JSON.
- **Image resized to max 1024px** — keeps the base64 payload fast without sacrificing enough detail for tool recognition.
- **autorefresh only in scanner mode** — `st_autorefresh` fires conditionally, so the classification tabs are never disrupted by background reruns.

---

## Project Structure

```
.
├── streamlit_app.py              # Main entry point — both modes in one file
├── classification_mcp_server.py  # MCP server — 4 Gemini-backed classification tools
├── classification_agent.py       # CLI alternative with Gemini auto-routing (optional)
├── requirements.txt              # Python dependencies
├── .env.example                  # API key template
├── .env                          # Your real API key (never commit this)
└── .gitignore                    # Keeps venv/ and .env out of git
```

---

## Setup

### 1. Get a Gemini API key
Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and create an API key.
It must start with `AIzaSy...` — not a `sk-...` token from another tool.

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Set up your `.env` file

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```
GEMINI_API_KEY=AIzaSy...your-real-key...
GEMINI_MODEL=gemini-2.5-flash
```

### 4. Run

```bash
streamlit run streamlit_app.py
```

> ⚠️ Do **not** run `python streamlit_app.py` — that launches it without a browser context and produces only warnings. Always use `streamlit run`.

Opens at `http://localhost:8501`.

---

## Usage Guide

### 🔬 Classification Mode

**Document**
1. Select **📄 Document** in the radio.
2. Paste document text or click **Use example** to load a sample invoice.
3. Click **Classify** → result shows category, confidence bar, key topics, and reasoning.

**Sensor Data**
1. Select **📡 Sensor Data**.
2. Paste labeled readings (`temperature=92C, vibration=8.4mm/s`) or plain numbers (`92, 8.4`).
3. Click **Classify** → result shows inferred sensor type + Normal/Fault state.

> Works with unlabeled numeric input — sensor type is inferred from the magnitude and pattern of the numbers using real-world range heuristics.

**Network Packet**
1. Select **🌐 Network Packet**.
2. Paste a packet summary or log entry.
3. Click **Classify** → result shows Normal / Suspicious / Priority + traffic signals.

**Image (Tools)**
1. Select **🔧 Image (Tools)**.
2. Use the **📷 Camera** tab to capture a photo, or **📁 Upload** to upload a JPEG/PNG.
3. Click **Detect Tools** → result shows a card per detected tool with name, category, and condition.

All results stack in a **Results** section below. Click **Clear** to reset.

---

### 📹 Real-time Scanner Mode

1. Switch to **📹 Real-time Scanner** in the sidebar.
2. Allow camera access when the browser prompts.
3. The live feed appears on the left — point it at your tools.
4. The system automatically scans every **10 seconds** and updates the result panel on the right.
5. Click **🔍 Scan Now** at any time to trigger an immediate scan.

**Result panel shows:**
- ✅ Scan number, time since last scan, time until next scan
- Countdown progress bar
- **Tools in frame** count + **Damaged** count with ⚠️ flag
- A card per tool: name, category icon, condition (color-coded), confidence bar

> The **Damaged** count is the primary industrial safety signal — a non-zero value means a tool needs attention before use.

---

## Sample Inputs

**Sensor — Normal**
```
temperature=45C, vibration=1.1mm/s, pressure=2.3 bar, RPM=1800. All within normal range.
```

**Sensor — Fault**
```
temperature=92C, vibration=8.4mm/s (baseline 1.2mm/s), pressure=stable.
Bearing temperature rising steadily over the last 10 minutes.
```

**Sensor — Plain numbers (no labels)**
```
92, 8.4, 1.2
```

**Network — Suspicious**
```
TCP SYN flood detected from 14 source IPs targeting port 443,
~9000 packets/sec, with no completed handshakes.
```

**Document — Invoice**
```
INVOICE #4471. Bill to: Acme Corp. Item: Cloud subscription (annual).
Amount due: $1,200. Payment due date: 30 June 2026.
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Your Google AI Studio API key. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model to use. Swap to `gemini-2.5-pro` for harder cases. |

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | Gemini API client (OpenAI-compatible endpoint) |
| `mcp` + `FastMCP` | Model Context Protocol server and client |
| `pydantic` | Strict per-modality output schema validation |
| `streamlit` | Web UI framework |
| `streamlit-webrtc` | Live camera feed in the browser (real-time scanner) |
| `streamlit-autorefresh` | Triggers auto-scan every 10 seconds (scanner mode only) |
| `opencv-python-headless` | Frame capture and BGR→RGB conversion |
| `Pillow` | Image resize before encoding |
| `python-dotenv` | Loads API key from `.env` |

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `Please pass a valid API key` | Wrong key format or placeholder still in `.env` | Get a key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — must start with `AIzaSy` |
| `Insufficient Balance` | Pay-as-you-go provider with no credit | Gemini AI Studio keys are free-tier — use those |
| `Unterminated string` / JSON parse error | Gemini response truncated | Already handled — `max_tokens` is set to 4096 with a truncation recovery fallback |
| `missing ScriptRunContext` | Ran with `python` instead of `streamlit run` | Use `streamlit run streamlit_app.py` |
| `429 Too Many Requests` | Free-tier rate limit hit | Wait 60 seconds and retry |
| Camera not working | Browser blocked camera access | Allow camera access in browser settings and refresh |

---

## Notes

- **Condition assessment over size.** For an industrial floor use case, tool size is not reliably readable from images (labels fade, tools are scattered). The system focuses on what is actionable: tool identity, category, and condition (Good / Worn / Damaged).
- **Real-time scan interval.** 10 seconds is the default. Change `SCAN_INTERVAL` at the top of `streamlit_app.py` to adjust.
- **Dual-method design (in progress).** This is Method 1 (LLM via MCP). Method 2 — custom ML models for document, sensor, and network data — is under development. When complete, both methods will classify the same input and their agreement will drive confidence: agreement → auto-act, disagreement → escalate for human review.
- **Production deployment.** `streamlit-webrtc` works on localhost without extra setup. For a cloud-hosted deployment (e.g. a VM on the factory network), a STUN/TURN server is required for WebRTC camera access.
