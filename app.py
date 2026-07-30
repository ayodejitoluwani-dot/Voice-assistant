# Streamlit webapp for the voice assistant.
#
# Two input modes (tabs): Speak (real ASR via NCAIR models) or Type (skips
# ASR, tests the rest of the pipeline).
#
# Two understanding modes (sidebar):
#  - Agent (default) -- an LLM reasons about the instruction and picks its
#    own sequence of tool calls. Handles compound, multi-step instructions.
#  - Rules -- the original keyword-matching intent classifier + regex
#    entity extraction. Kept for direct comparison against the agent.

import os
import tempfile

import env_config  # loads secrets.env into the environment automatically

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from pipeline import run_pipeline

st.set_page_config(page_title="NCAIR Voice Assistant", page_icon="🎙️", layout="centered")

# ---------------------------------------------------------------------------
# Theme: dark, glowing, voice-first -- see .streamlit/config.toml for the
# base widget palette. This block handles everything config.toml can't:
# fonts, cards, the signature glowing orb, and gradient buttons.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --va-bg: #0A0C16;
    --va-card: #151829;
    --va-card-border: #232744;
    --va-blue: #4FA8FF;
    --va-violet: #8B6FF0;
    --va-text: #F4F5F9;
    --va-muted: #8890A8;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: radial-gradient(circle at 50% -10%, #1B2040 0%, var(--va-bg) 55%);
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

code, pre, .stCode, [data-testid="stJson"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ---- Hero header ---- */
.va-hero { text-align: center; padding: 1.6rem 0 0.4rem 0; }
.va-hero h1 {
    font-size: 2rem; margin-bottom: 0.2rem; color: var(--va-text);
    background: linear-gradient(90deg, #F4F5F9 30%, var(--va-blue) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.va-hero p { color: var(--va-muted); font-size: 0.95rem; margin-top: 0; }

/* ---- Signature glowing orb ---- */
.va-orb-wrap { display: flex; justify-content: center; margin: 0.6rem 0 1.4rem 0; }
.va-orb {
    width: 92px; height: 92px; border-radius: 50%;
    border: 2px solid rgba(79, 168, 255, 0.55);
    box-shadow: 0 0 24px 4px rgba(79, 168, 255, 0.25), inset 0 0 20px rgba(139, 111, 240, 0.25);
    animation: va-pulse 2.6s ease-in-out infinite;
}
@keyframes va-pulse {
    0%, 100% { box-shadow: 0 0 20px 2px rgba(79, 168, 255, 0.20), inset 0 0 16px rgba(139, 111, 240, 0.20); transform: scale(1); }
    50% { box-shadow: 0 0 34px 8px rgba(79, 168, 255, 0.40), inset 0 0 26px rgba(139, 111, 240, 0.35); transform: scale(1.035); }
}

/* ---- Result cards ---- */
.va-card {
    background: var(--va-card); border: 1px solid var(--va-card-border);
    border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 0.85rem;
    border-left: 3px solid var(--va-blue);
}
.va-card.va-violet { border-left-color: var(--va-violet); }
.va-card.va-response { border-left-color: #4FDBA0; }
.va-card-label {
    font-family: 'Space Grotesk', sans-serif; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--va-muted);
    margin-bottom: 0.35rem;
}
.va-card-body { color: var(--va-text); font-size: 0.98rem; line-height: 1.5; }
.va-card-caption { color: var(--va-muted); font-size: 0.82rem; margin-top: 0.4rem; }

/* ---- Buttons ---- */
[data-testid="stButton"] button, [data-testid="baseButton-secondary"] {
    background: linear-gradient(90deg, var(--va-blue), var(--va-violet)) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.55rem 1.2rem !important;
    box-shadow: 0 4px 14px rgba(79, 168, 255, 0.25);
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] { background: #0D0F1C; border-right: 1px solid var(--va-card-border); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 0.95rem !important; color: var(--va-muted); }

/* ---- Tabs ---- */
[data-testid="stTabs"] button { font-family: 'Space Grotesk', sans-serif; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def card(label, body_text, variant="", caption=None):
    classes = f"va-card {variant}".strip()
    caption_html = f'<div class="va-card-caption">{caption}</div>' if caption else ""
    st.markdown(
        f'<div class="{classes}"><div class="va-card-label">{label}</div>'
        f'<div class="va-card-body">{body_text}</div>{caption_html}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="va-hero"><h1>NCAIR Voice Assistant</h1>'
    '<p>Speak in Hausa, Yoruba, or Igbo — get it understood, acted on, and answered back.</p></div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="va-orb-wrap"><div class="va-orb"></div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar: language + understanding mode + agent backend
# ---------------------------------------------------------------------------
st.sidebar.markdown("### Language")
lang = st.sidebar.selectbox("Language", ["hausa", "yoruba", "igbo"], label_visibility="collapsed")

st.sidebar.markdown("### Understanding mode")
mode = st.sidebar.radio(
    "Understanding mode",
    ["agent", "rules"],
    label_visibility="collapsed",
    help=(
        "agent: an AI model reasons about the instruction and can chain multiple "
        "tool calls for compound requests. rules: the original keyword-"
        "matching classifier, one action per instruction."
    ),
)

backend = "gemini"
if mode == "agent":
    st.sidebar.markdown("### Agent backend")
    backend = st.sidebar.radio(
        "Agent backend",
        ["gemini", "ncair", "claude"],
        label_visibility="collapsed",
        help=(
            "gemini: Google's Gemini via API (fast, free tier, needs GEMINI_API_KEY). "
            "ncair: NCAIR's own N-ATLaS model running locally -- no API key, but "
            "slower on CPU-only machines (see README). claude: Anthropic's Claude "
            "via API (fast, needs ANTHROPIC_API_KEY, paid)."
        ),
    )

_backend_notes = {
    "ncair": "Needs the N-ATLaS GGUF model downloaded locally (see README).",
    "claude": "Needs ANTHROPIC_API_KEY set in your environment (paid).",
    "gemini": "Needs GEMINI_API_KEY set in your environment (free tier available).",
}
st.sidebar.caption(
    _backend_notes[backend] if mode == "agent"
    else "Rules mode only needs what's already set up (Google credentials)."
)

# ---------------------------------------------------------------------------
# Input: Speak or Type
# ---------------------------------------------------------------------------
result = None
tab_speak, tab_type = st.tabs(["🎙️  Speak", "⌨️  Type"])

with tab_speak:
    audio = mic_recorder(
        start_prompt="🎙️ Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
        use_container_width=True,
        key="recorder",
    )

    if audio and audio.get("bytes"):
        st.audio(audio["bytes"])

        # asr.transcribe() takes a file path, so save the recorded bytes to
        # a temp file first.
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio["bytes"])
            tmp_path = tmp.name

        try:
            with st.spinner("Transcribing and running the pipeline..."):
                result = run_pipeline(lang=lang, audio_path=tmp_path, return_details=True, mode=mode, backend=backend)
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
        finally:
            os.unlink(tmp_path)

with tab_type:
    typed_text = st.text_input(
        "Type an instruction in the selected language",
        placeholder="e.g. schedule a meeting with John tomorrow at 3pm, remind me beforehand, and email him about it",
        label_visibility="collapsed",
    )
    if st.button("Run", key="run_typed") and typed_text.strip():
        with st.spinner("Thinking..." if mode == "agent" else "Processing..."):
            result = run_pipeline(lang=lang, text=typed_text, return_details=True, mode=mode, backend=backend)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if result:
    st.markdown("&nbsp;", unsafe_allow_html=True)

    card("Heard", result["original_text"], caption=f"Confidence {result['confidence']:.2f}")
    card("Translated (English)", result["translated_text"], variant="va-violet")

    if result["mode"] == "agent":
        steps = result.get("steps")
        st.caption(f"Backend: **{result.get('backend')}**")
        if steps:
            st.markdown(f"**Agent's plan — {len(steps)} step(s) executed:**")
            for i, step in enumerate(steps, 1):
                with st.expander(f"Step {i}: {step['tool']}", expanded=True):
                    st.write("**Input:**")
                    st.json(step["input"])
                    st.write("**Result:**")
                    st.write(step["result"])
        else:
            st.caption("No tools were called for this instruction.")
    else:
        st.caption(f"Intent: **{result.get('intent') or 'not recognized'}**")
        st.json(result.get("entities", {}))

    response_caption = None
    if result.get("response_en") and result["response_en"] != result["response"]:
        response_caption = f"English: {result['response_en']}"
    card("Response", result["response"], variant="va-response", caption=response_caption)
