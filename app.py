# Streamlit webapp for the voice assistant.
#
# Two input modes:
#  - Record audio (real ASR via NCAIR models -- needs Hugging Face access
#    approved for the gated model, and internet access to download it)
#  - Type text directly (skips ASR entirely -- useful for testing the rest
#    of the pipeline: translation, understanding, actions)
#
# Two understanding modes (pick in the sidebar):
#  - Agent (default) -- an LLM reasons about the instruction and picks its
#    own sequence of tool calls. Handles compound, multi-step instructions.
#    Default backend is NCAIR's own N-ATLaS model, running locally.
#  - Rules -- the original keyword-matching intent classifier + regex
#    entity extraction. Kept for direct comparison against the agent.

import os
import tempfile

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from pipeline import run_pipeline

st.set_page_config(page_title="Voice Assistant", layout="centered")
st.title("🎙️ Multilingual Voice Assistant")

lang = st.selectbox("Language", ["hausa", "yoruba", "igbo"])
mode = st.sidebar.radio(
    "Understanding mode",
    ["agent", "rules"],
    help=(
        "agent: an AI model reasons about the instruction and can chain multiple "
        "tool calls for compound requests. rules: the original keyword-"
        "matching classifier, one action per instruction."
    ),
)

backend = "ncair"
if mode == "agent":
    backend = st.sidebar.radio(
        "Agent backend",
        ["ncair", "claude", "gemini"],
        help=(
            "ncair: NCAIR's own N-ATLaS model running locally -- no API key, but "
            "slower on CPU-only machines (see README). claude: Anthropic's Claude "
            "via API (fast, needs ANTHROPIC_API_KEY, paid). gemini: Google's "
            "Gemini via API (fast, needs GEMINI_API_KEY, free tier available)."
        ),
    )

_backend_notes = {
    "ncair": "agent+ncair needs the N-ATLaS GGUF model downloaded locally (see README).",
    "claude": "agent+claude needs ANTHROPIC_API_KEY set in your environment (paid).",
    "gemini": "agent+gemini needs GEMINI_API_KEY set in your environment (free tier available).",
}
st.sidebar.caption(
    _backend_notes[backend] if mode == "agent"
    else "rules mode only needs what's already set up (Google credentials)."
)

st.write("---")
st.subheader("Option A: Record audio")

audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=False,
    use_container_width=True,
    key="recorder",
)

result = None

if audio and audio.get("bytes"):
    st.audio(audio["bytes"])

    # asr.transcribe() takes a file path, so save the recorded bytes to a
    # temp file first.
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

st.write("---")
st.subheader("Option B: Type text (skips ASR, tests the rest of the pipeline)")

typed_text = st.text_input(
    "Type an instruction in the selected language",
    placeholder="e.g. schedule a meeting with John tomorrow at 3pm, remind me beforehand, and email him about it",
)
if st.button("Run on typed text") and typed_text.strip():
    with st.spinner("Thinking..." if mode == "agent" else "Processing..."):
        result = run_pipeline(lang=lang, text=typed_text, return_details=True, mode=mode, backend=backend)

# --- Display results -------------------------------------------------------
if result:
    st.write("---")
    st.subheader("Result")

    st.write("**Heard:**")
    st.code(result["original_text"])
    st.write(f"**Confidence:** {result['confidence']:.2f}")

    st.write("**Translated (English):**")
    st.code(result["translated_text"])

    if result["mode"] == "agent":
        steps = result.get("steps")
        st.write(f"**Backend:** {result.get('backend')}")
        if steps:
            st.write(f"**Agent's plan — {len(steps)} step(s) executed:**")
            for i, step in enumerate(steps, 1):
                with st.expander(f"Step {i}: {step['tool']}", expanded=True):
                    st.write("**Input:**")
                    st.json(step["input"])
                    st.write("**Result:**")
                    st.write(step["result"])
        else:
            st.write("**Agent's plan:** no tools were called for this instruction.")
    else:
        st.write(f"**Intent:** {result.get('intent') or 'not recognized'}")
        st.write("**Entities:**")
        st.json(result.get("entities", {}))

    st.write("**Response:**")
    st.success(result["response"])
    if result.get("response_en") and result["response_en"] != result["response"]:
        st.caption(f"English: {result['response_en']}")
