# Gemini — MCP Classification Agent (Streamlit UI)

Method 1 (LLM via MCP) of the Multimodal Classification Agent project.
An **event-driven Streamlit app**: you select the input type, provide the
data, and it's classified via **Google Gemini** through an **MCP tool**.
Three input types are supported: documents, sensor data, and network
packets. *(Images are out of scope for this phase.)*

## Architecture — user selects, tool classifies

```
                 ┌──────────────────────────┐
   You select  ─►│  Document / Sensor / Net  │   (Streamlit UI)
                 └─────────────┬─────────────┘
                                ▼  calls the matching tool directly
        ┌──────────────── MCP Server (FastMCP) ────────────────┐
        │  classify_document     → Invoice/Report/Contract/Manual │
        │  classify_sensor_data  → Normal/Fault                    │
        │  classify_network_packet → Normal/Suspicious/Priority    │
        │  (each tool = Gemini + strict Pydantic schema)           │
        └──────────────────────────┬──────────────────────────────┘
                                    ▼
                  structured JSON result (class + confidence
                  + key signals + justification) shown in the UI
```

There is **no LLM auto-routing** — your selection in the UI determines which
MCP tool runs. This keeps the demo predictable: no risk of the model picking
the wrong modality on stage.

- **`classification_mcp_server.py`** — the MCP server. Exposes three
  classification *tools*, each backed by Gemini with a strict per-modality
  Pydantic schema (a direct generalization of the professor's
  `classify_medical_report` example).
- **`streamlit_app.py`** — the UI / MCP client. You pick a modality, type or
  paste input, click **Classify**, and it calls that exact MCP tool and
  renders the structured result (class, confidence bar, signals, reasoning).
- **`classification_agent.py`** — a CLI alternative (no UI). Runs three
  built-in sample inputs through Gemini's auto-routing and prints JSON.
  Kept for quick command-line testing; not required for the Streamlit app.

## Setup

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/apikey).
   It must start with `AIzaSy...` — not a `sk-...` key from another tool.

2. Create a virtual environment and install dependencies:

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # macOS/Linux
   venv\Scripts\activate           # Windows

   pip install -r requirements.txt
   ```

3. Create your `.env` file:

   ```bash
   cp .env.example .env
   ```

   Then open `.env` and set your real key:

   ```
   GEMINI_API_KEY=AIzaSy...your-real-key...
   GEMINI_MODEL=gemini-2.5-flash
   ```

## Files

| File | Purpose |
|---|---|
| `classification_mcp_server.py` | MCP server — the 3 classification tools (Gemini-backed) |
| `streamlit_app.py` | **Main entry point.** Streamlit UI — select type, classify, view result |
| `classification_agent.py` | CLI alternative with Gemini auto-routing (optional) |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for your API key — copy to `.env` and fill in |
| `.gitignore` | Keeps `venv/` and your real `.env` out of git |

## Run

```bash
streamlit run streamlit_app.py
```

This opens the app in your browser (usually `http://localhost:8501`). Use it like this:

1. Choose a type with the radio buttons: **Document / Sensor Data / Network Packet**.
2. Either click **"Use example"** to auto-fill a sample, or paste/type your own input.
3. Click **Classify**.
4. The result card shows the predicted class, a confidence bar, the key
   signals the model used, and its reasoning. Expand **"Raw JSON"** to see
   the exact structured output. Past results stack below; **Clear** resets them.

Each click connects to the MCP server fresh, calls the one matching tool, and
disconnects — simple and reliable for a live demo.

## Important notes on Gemini

1. **OpenAI-compatible endpoint.** Gemini is accessed via the standard `openai`
   Python SDK, just pointed at Google's base URL:
   `https://generativelanguage.googleapis.com/v1beta/openai/`. No Gemini-specific
   SDK is required.
2. **Default model.** `gemini-2.5-flash` — fast, sits comfortably on Gemini's
   free tier, and supports both JSON output mode and function calling.
   For harder cases, override with `GEMINI_MODEL=gemini-2.5-pro` in `.env`.
3. **Free tier.** Google AI Studio issues API keys with a free quota (rate-limited
   per minute/day). Good for development and demos; check your current limits on
   the AI Studio dashboard if you start seeing 429 errors.
4. **JSON mode.** Each tool requests `response_format={"type": "json_object"}`,
   matching the professor's reference pattern. `temperature=0.0` keeps results
   deterministic and repeatable.

## How this fits the bigger project

This is **Method 1**. In the full dual-method design, the same input is also
classified by **Method 2 (custom ML models)**, and the two predictions are
compared — agreement → high confidence, disagreement → escalate for review.