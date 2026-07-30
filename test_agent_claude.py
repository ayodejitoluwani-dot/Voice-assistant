"""
Tests for agent_claude.py -- the Claude-driven tool-calling loop.

These mock the Anthropic client entirely (no real API key or network call
needed) so we can verify the loop's logic: it keeps calling tools until
Claude stops requesting them, executes multiple tool calls per round
correctly, and produces a final summary. The underlying tool execution
(e.g. actually hitting Google Calendar) is exercised for real, which in
this test environment has no credentials.json -- so we're also confirming
it fails gracefully rather than crashing the agent loop.
"""

import agent_claude as agent


class FakeBlock:
    def __init__(self, type_, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeMessages:
    def __init__(self, responses):
        self._responses = iter(responses)

    def create(self, **kwargs):
        return next(self._responses)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def test_agent_executes_multiple_tool_calls_then_summarizes(monkeypatch):
    # Round 1: Claude asks for two tools in one turn (a compound instruction).
    round_1 = FakeResponse([
        FakeBlock("tool_use", id="call_1", name="schedule_event",
                  input={"summary": "Meeting with John", "time": "3pm tomorrow"}),
        FakeBlock("tool_use", id="call_2", name="set_reminder",
                  input={"text": "Meeting with John", "time": "2:45pm tomorrow"}),
    ])
    # Round 2: Claude has the tool results and gives a final plain-language summary.
    round_2 = FakeResponse([
        FakeBlock("text", text="Scheduled the meeting and set a reminder beforehand."),
    ])

    fake_client = FakeClient([round_1, round_2])
    monkeypatch.setattr(agent.anthropic, "Anthropic", lambda: fake_client)

    outcome = agent.run_agent("schedule a meeting with John tomorrow at 3pm and remind me beforehand")

    assert len(outcome["steps"]) == 2
    assert outcome["steps"][0]["tool"] == "schedule_event"
    assert outcome["steps"][1]["tool"] == "set_reminder"
    assert outcome["final_response"] == "Scheduled the meeting and set a reminder beforehand."
    # No credentials.json in this test environment -- confirm graceful
    # failure rather than a crash propagating out of the loop.
    assert "Google isn't set up yet" in outcome["steps"][0]["result"]


def test_agent_stops_immediately_when_no_tools_needed(monkeypatch):
    # Claude decides nothing needs to be done -- just replies with text.
    round_1 = FakeResponse([
        FakeBlock("text", text="I couldn't find an action that matches that request."),
    ])
    fake_client = FakeClient([round_1])
    monkeypatch.setattr(agent.anthropic, "Anthropic", lambda: fake_client)

    outcome = agent.run_agent("what's the weather like")

    assert outcome["steps"] == []
    assert "couldn't find" in outcome["final_response"]


def test_run_tool_phone_call_is_always_logged_only():
    result = agent.run_tool("phone_call", {"contact": "Musa"})
    assert "not actually dialled" in result


def test_run_tool_unknown_tool_name():
    result = agent.run_tool("does_not_exist", {})
    assert "Unknown tool" in result
