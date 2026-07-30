"""
Tests for agent_ncair.py -- the local NCAIR N-ATLaS backend's manual
JSON tool-call protocol.

These mock _get_model() entirely, so no real GGUF file or llama-cpp-python
installation is needed to verify the loop logic: parsing the model's JSON
output, executing tools, feeding results back, and stopping on a "final"
action. Also covers the two realistic failure modes for a smaller,
non-tool-trained model: invalid JSON and an unrecognized action key.
"""

import json

import agent_ncair


class FakeModel:
    """Stands in for llama_cpp.Llama -- returns one canned response per call,
    in the same {"choices": [{"text": ...}]} shape the real thing returns."""

    def __init__(self, texts):
        self._texts = iter(texts)

    def __call__(self, prompt, **kwargs):
        return {"choices": [{"text": next(self._texts)}]}


def test_agent_ncair_executes_tool_then_finalizes(monkeypatch):
    texts = [
        json.dumps({"action": "tool_call", "tool": "schedule_event",
                    "input": {"summary": "Meeting with John", "time": "3pm tomorrow"}}),
        json.dumps({"action": "final", "response": "Scheduled the meeting."}),
    ]
    monkeypatch.setattr(agent_ncair, "_get_model", lambda: FakeModel(texts))

    outcome = agent_ncair.run_agent("schedule a meeting with John tomorrow at 3pm")

    assert len(outcome["steps"]) == 1
    assert outcome["steps"][0]["tool"] == "schedule_event"
    assert outcome["final_response"] == "Scheduled the meeting."
    # No credentials.json in this test environment -- confirm graceful failure.
    assert "Google isn't set up yet" in outcome["steps"][0]["result"]


def test_agent_ncair_handles_multi_step_sequentially(monkeypatch):
    texts = [
        json.dumps({"action": "tool_call", "tool": "schedule_event",
                    "input": {"summary": "Meeting", "time": "3pm"}}),
        json.dumps({"action": "tool_call", "tool": "set_reminder",
                    "input": {"text": "Meeting", "time": "2:45pm"}}),
        json.dumps({"action": "final", "response": "Scheduled and reminded."}),
    ]
    monkeypatch.setattr(agent_ncair, "_get_model", lambda: FakeModel(texts))

    outcome = agent_ncair.run_agent("schedule a meeting at 3pm and remind me beforehand")

    assert [s["tool"] for s in outcome["steps"]] == ["schedule_event", "set_reminder"]
    assert outcome["final_response"] == "Scheduled and reminded."


def test_agent_ncair_handles_invalid_json_gracefully(monkeypatch):
    texts = ["I think you should schedule a meeting, let me do that for you now."]
    monkeypatch.setattr(agent_ncair, "_get_model", lambda: FakeModel(texts))

    outcome = agent_ncair.run_agent("schedule a meeting")

    assert outcome["steps"] == []
    assert "didn't return valid JSON" in outcome["final_response"]


def test_agent_ncair_handles_unrecognized_action(monkeypatch):
    texts = [json.dumps({"action": "do_something_weird"})]
    monkeypatch.setattr(agent_ncair, "_get_model", lambda: FakeModel(texts))

    outcome = agent_ncair.run_agent("schedule a meeting")

    assert "unrecognized action" in outcome["final_response"]


def test_extract_json_pulls_first_balanced_object():
    text = 'Sure, here is my answer: {"action": "final", "response": "done"} thanks!'
    parsed = agent_ncair._extract_json(text)
    assert parsed == {"action": "final", "response": "done"}
