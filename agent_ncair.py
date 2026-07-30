"""
NCAIR N-ATLaS backend for the agent -- runs entirely on your machine, no
API key needed, using NCAIR's own model instead of Claude.

Unlike Claude, N-ATLaS has no built-in structured tool-calling API, so
this builds a manual protocol: the model is told about the available
tools as plain text and instructed to always respond with a JSON object
shaped like either

    {"action": "tool_call", "tool": "<name>", "input": {...}}
    {"action": "final", "response": "<plain language summary>"}

One tool call per model turn (simpler and more reliable for this model
than asking it to plan several calls at once, which is more of a
strength Claude has from tool-use-specific training). The loop keeps
feeding tool results back until the model says "final" or a safety cap
is hit.

Setup (one-time):
  1. Request access to NCAIR1/N-ATLaS on Hugging Face (same gated-access
     flow as the ASR models) -- https://huggingface.co/NCAIR1/N-ATLaS
  2. Download the quantized GGUF version (much more practical for CPU-only
     laptops than the ~20GB full-precision weights):
       python download_models.py ncair-llm
  3. pip install llama-cpp-python (this compiles from source -- needs
     Xcode Command Line Tools on Mac: xcode-select --install)

Honest performance note: an 8B model on CPU, even quantized, is slow --
expect anywhere from 10 seconds to a couple of minutes per response
depending on your machine, versus Claude's roughly 1-3 seconds. Test this
well before relying on it for a live demo.
"""

import json
import os
import re

from agent_tools import TOOLS, run_tool, tools_as_text

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "ncair-llm", "n-atlas-q4_k_m.gguf"
)
MAX_TOOL_ROUNDS = 6
MAX_NEW_TOKENS = 400

SYSTEM_PROMPT = f"""You are the task-execution engine for a multilingual voice \
assistant built by NCAIR. Instructions arrive already translated into English.

You have access to these tools:
{tools_as_text()}

An instruction may need one tool call, several in sequence, or none at all. \
On every turn, respond with ONLY a single JSON object, nothing else -- no \
explanation, no markdown formatting, just the JSON object on its own.

To call a tool:
{{"action": "tool_call", "tool": "<tool name>", "input": {{...matching that tool's arguments...}}}}

Once everything needed has been done (or if no tool applies), give your \
final answer instead:
{{"action": "final", "response": "<a short, plain-language summary of what was done, for a person to read>"}}

Call only one tool per turn. If a multi-step instruction needs several \
tools, call the first one now -- you'll be given its result and can call \
the next one on your next turn."""

_model = None


def _get_model():
    global _model
    if _model is None:
        from llama_cpp import Llama

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"{MODEL_PATH} not found. Run 'python download_models.py ncair-llm' "
                "first (after requesting Hugging Face access to NCAIR1/N-ATLaS)."
            )
        _model = Llama(model_path=MODEL_PATH, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
    return _model


def _build_prompt(turns: list) -> str:
    """Builds a Llama-3 chat-template-formatted prompt from a list of
    (role, content) tuples, matching N-ATLaS's expected format."""
    parts = ["<|begin_of_text|>"]
    for role, content in turns:
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _extract_json(text: str) -> dict:
    """Pulls the first balanced {...} block out of the model's raw output
    and parses it. Raises ValueError if nothing parseable is found --
    models occasionally wrap JSON in commentary or markdown despite
    instructions not to."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(match.group(0))


def run_agent(instruction: str) -> dict:
    """
    Same return shape as agent_claude.run_agent, so pipeline.py can treat
    both backends identically:
      {
        "final_response": "<plain-language summary>",
        "steps": [{"tool": name, "input": {...}, "result": "..."}, ...]
      }
    """
    model = _get_model()
    turns = [("system", SYSTEM_PROMPT), ("user", instruction)]
    steps = []

    for _ in range(MAX_TOOL_ROUNDS):
        prompt = _build_prompt(turns)
        output = model(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.1,
            repeat_penalty=1.12,
            stop=["<|eot_id|>"],
        )
        raw_text = output["choices"][0]["text"].strip()

        try:
            parsed = _extract_json(raw_text)
        except (ValueError, json.JSONDecodeError) as e:
            # Model didn't follow the protocol -- surface its raw text
            # rather than crashing, and stop (retrying blindly risks a
            # confusing loop with a smaller model like this).
            return {
                "final_response": f"[agent didn't return valid JSON, showing raw output] {raw_text}",
                "steps": steps,
            }

        if parsed.get("action") == "final":
            return {"final_response": parsed.get("response", "").strip(), "steps": steps}

        if parsed.get("action") == "tool_call":
            tool_name = parsed.get("tool")
            tool_input = parsed.get("input", {})
            result_text = run_tool(tool_name, tool_input)
            steps.append({"tool": tool_name, "input": tool_input, "result": result_text})

            turns.append(("assistant", raw_text))
            turns.append(("user", f"Tool result: {result_text}"))
            continue

        return {
            "final_response": f"[agent returned an unrecognized action] {raw_text}",
            "steps": steps,
        }

    return {
        "final_response": "Stopped after several steps without a final summary -- something may be looping.",
        "steps": steps,
    }
