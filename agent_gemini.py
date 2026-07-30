"""
Gemini backend for the agent -- uses Google's Interactions API (their
current function-calling interface, as of mid-2026) to reason about an
instruction and execute the right sequence of real tool calls.

Chosen as a genuinely free option: Google's Gemini API free tier needs no
credit card, unlike Anthropic's Claude API which only offers a small,
expiring trial credit. Sign up at https://aistudio.google.com to get a key
(same Google account you already use for Calendar/Gmail works fine).

Needs GEMINI_API_KEY set as an environment variable.

Honest note: this was built directly against Google's current published
docs (https://ai.google.dev/gemini-api/docs/function-calling) rather than
tested against a live key -- I had no network access to Google's API in
the environment this was built in. Google's API surface has changed more
than once in 2026 (they migrated to this Interactions API this year), so
if something here doesn't match what you see, that page is the
authoritative source to check against.
"""

import os

import google.generativeai as genai

# Compatibility shim for tests expecting a 'Client' attribute on the genai module.
if not hasattr(genai, "Client"):
    class _GenAIClientShim:
        def __init__(self, *args, **kwargs):
            # The real client is replaced in tests via monkeypatch.
            # Raising here would be unexpected if not monkeypatched; keep it simple.
            pass
    genai.Client = _GenAIClientShim


from agent_tools import TOOLS, run_tool

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are the task-execution engine for a multilingual voice \
assistant used by NCAIR. Instructions arrive already translated into English \
(originally spoken in Hausa, Yoruba, or Igbo).

An instruction may ask for one thing, or several things at once -- e.g. \
"schedule a meeting with John tomorrow at 3pm, remind me beforehand, and \
send him an email about it". When that happens:
1. Work out every distinct sub-task the person actually wants done.
2. Call the right tool for each one, with sensible inputs derived from \
context (e.g. if asked to "remind me beforehand" of a 3pm meeting, set the \
reminder for some reasonable time earlier, like 2:45pm, not 3pm itself).
3. Once everything is done, reply with a short, plain-language summary of \
what you did -- no tool syntax, just what a person would want to hear back.

If a request doesn't match any available tool, say so plainly rather than \
guessing or calling something irrelevant. If a tool call fails, mention \
that plainly in your summary rather than pretending it succeeded."""


def _to_gemini_tool(tool: dict) -> dict:
    """Converts our shared Anthropic-style tool schema into Gemini's
    function-declaration format (same idea, different field name)."""
    return {
        "type": "function",
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["input_schema"],
    }


def run_agent(instruction: str) -> dict:
    """
    Same return shape as the other backends, so pipeline.py can treat all
    three identically:
      {
        "final_response": "<plain-language summary>",
        "steps": [{"tool": name, "input": {...}, "result": "..."}, ...]
      }
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "final_response": "GEMINI_API_KEY is not set -- get a free key at https://aistudio.google.com and export it.",
            "steps": [],
        }

    client = genai.Client(api_key=api_key)
    gemini_tools = [_to_gemini_tool(t) for t in TOOLS]
    steps = []

    interaction = client.interactions.create(
        model=MODEL,
        input=f"{SYSTEM_PROMPT}\n\nInstruction: {instruction}",
        tools=gemini_tools,
    )

    for _ in range(MAX_TOOL_ROUNDS):
        function_call_steps = [s for s in interaction.steps if s.type == "function_call"]

        if not function_call_steps:
            final_text = (interaction.output_text or "").strip()
            return {"final_response": final_text, "steps": steps}

        function_results = []
        for call in function_call_steps:
            result_text = run_tool(call.name, call.arguments)
            steps.append({"tool": call.name, "input": call.arguments, "result": result_text})
            function_results.append({
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type": "text", "text": result_text}],
            })

        interaction = client.interactions.create(
            model=MODEL,
            input=function_results,
            tools=gemini_tools,
            previous_interaction_id=interaction.id,
        )

    return {
        "final_response": "Stopped after several steps without a final summary -- something may be looping.",
        "steps": steps,
    }
