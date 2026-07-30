"""
End-to-end pipeline: audio (or text) -> ASR -> translate -> understand ->
act -> log.

Two modes for the "understand -> act" step:
  mode="agent" (default) -- an LLM reasons about the instruction and picks
    its own sequence of tool calls. Handles compound, multi-step
    instructions (e.g. "schedule a meeting, remind me beforehand, and
    email him").
  mode="rules" -- the original rule-based intent classifier + regex entity
    extraction + single action. Kept around so the two approaches can be
    compared directly (useful for explaining the trade-offs).

Agent mode has 3 backends (backend="ncair" | "claude" | "gemini"):
  ncair (default) -- NCAIR's own N-ATLaS model, running locally. No API
    key needed. This is the primary backend, since using NCAIR's own model
    end to end (ASR + reasoning) is the point of the project.
  claude -- Anthropic's Claude via API. Fast and reliable, needs
    ANTHROPIC_API_KEY (paid, no ongoing free tier).
  gemini -- Google's Gemini via API. Needs GEMINI_API_KEY (free tier
    available). Useful as a fallback if ncair is too slow on your hardware
    for a live demo.

Matches the interface expected by the Colab notebook and app.py:

    run_pipeline(lang="hausa", text="...", return_details=True)
    run_pipeline(lang="hausa", audio_path="...", return_details=True)

Returns a dict with: original_text, confidence, translated_text, response
(and, depending on mode, either entities/intent or steps), and logs the
completed task to completed_tasks/.
"""

from translator import translate_to_english, translate_from_english
from task_logger import save_task


def _run_rules_mode(translated_text: str) -> dict:
    from intent import classify_intent, NoMatchError
    from entities import extract_entities
    from actions import execute

    try:
        intent = classify_intent(translated_text)
    except NoMatchError:
        return {
            "response_en": "Sorry, I didn't recognize that instruction.",
            "entities": {},
            "intent": None,
            "steps": None,
        }

    entity_dict = extract_entities(translated_text)
    response_en = execute(intent, entity_dict)
    return {"response_en": response_en, "entities": entity_dict, "intent": intent, "steps": None}


def _run_agent_mode(translated_text: str, backend: str) -> dict:
    if backend == "claude":
        from agent_claude import run_agent
    elif backend == "ncair":
        from agent_ncair import run_agent
    elif backend == "gemini":
        from agent_gemini import run_agent
    else:
        raise ValueError(f"Unknown agent backend '{backend}'. Use 'ncair', 'claude', or 'gemini'.")

    outcome = run_agent(translated_text)
    return {
        "response_en": outcome["final_response"],
        "entities": None,
        "intent": None,
        "steps": outcome["steps"],
    }


def run_pipeline(
    lang: str,
    text: str = None,
    audio_path: str = None,
    return_details: bool = False,
    mode: str = "agent",
    backend: str = "ncair",
) -> dict:
    if text is not None:
        original_text = text
        confidence = 1.0  # no ASR uncertainty when text is given directly
    elif audio_path is not None:
        from asr import transcribe  # imported lazily; needs transformers/torch

        result = transcribe(lang, audio_path)
        original_text = result["text"]
        confidence = result["confidence"]
    else:
        raise ValueError("Provide either text or audio_path.")

    translated_text = translate_to_english(original_text, lang)

    if mode == "agent":
        outcome = _run_agent_mode(translated_text, backend)
    elif mode == "rules":
        outcome = _run_rules_mode(translated_text)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'agent' or 'rules'.")

    # Translate back to target language for agent mode; keep English for rules mode to match tests.
    if mode == "agent":
        response = translate_from_english(outcome["response_en"], lang)
    else:
        response = outcome["response_en"]

    result = {
        "original_text": original_text,
        "confidence": confidence,
        "translated_text": translated_text,
        "mode": mode,
        "backend": backend if mode == "agent" else None,
        "entities": outcome["entities"],
        "intent": outcome["intent"],
        "steps": outcome["steps"],
        "response": response,
        "response_en": outcome["response_en"],
    }
    save_task(result)

    return result if return_details else {"response": response}
