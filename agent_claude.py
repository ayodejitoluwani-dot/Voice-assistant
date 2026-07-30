"""
Claude backend for the agent -- uses Anthropic's tool-use (function
calling) API to reason about an instruction and execute the right
sequence of real tool calls.

Needs an Anthropic API key (from console.anthropic.com -- separate from
whatever chat interface you're using elsewhere) set as ANTHROPIC_API_KEY.

This is the default, tested backend. See agent_ncair.py for the
alternative that uses NCAIR's own N-ATLaS model running locally instead.
"""

import os

import anthropic

from agent_tools import TOOLS, run_tool

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
MAX_TOOL_ROUNDS = 6  # safety cap so a confused agent can't loop forever

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
3. Call tools one at a time, using each tool's result to inform the next \
call if needed.
4. Once everything is done, reply with a short, plain-language summary of \
what you did -- no tool syntax, just what a person would want to hear back.

If a request doesn't match any available tool, say so plainly rather than \
guessing or calling something irrelevant. If a tool call fails, mention \
that plainly in your summary rather than pretending it succeeded."""


def run_agent(instruction: str) -> dict:
    """
    Give Claude the instruction and let it decide what tool(s) to call, in
    what order. Returns:
      {
        "final_response": "<plain-language summary>",
        "steps": [{"tool": name, "input": {...}, "result": "..."}, ...]
      }
    """
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

    messages = [{"role": "user", "content": instruction}]
    steps = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        tool_uses = [block for block in response.content if block.type == "tool_use"]

        if not tool_uses:
            final_text = "".join(block.text for block in response.content if block.type == "text")
            return {"final_response": final_text.strip(), "steps": steps}

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for call in tool_uses:
            result_text = run_tool(call.name, call.input)
            steps.append({"tool": call.name, "input": call.input, "result": result_text})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": result_text,
            })

        messages.append({"role": "user", "content": tool_results})

    return {
        "final_response": "Stopped after several steps without a final summary -- something may be looping.",
        "steps": steps,
    }
