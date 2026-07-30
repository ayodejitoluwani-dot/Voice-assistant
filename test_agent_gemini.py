"""
Tests for agent_gemini.py -- mocks google.genai.Client entirely (no real
API key or network call needed) to verify the Interactions API loop logic:
executing function_call steps, feeding function_result back with
previous_interaction_id, and stopping once no more function_call steps
are returned.
"""

import agent_gemini


class FakeStep:
    def __init__(self, type_, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeInteraction:
    def __init__(self, id_, steps, output_text=""):
        self.id = id_
        self.steps = steps
        self.output_text = output_text


class FakeInteractionsAPI:
    def __init__(self, interactions):
        self._interactions = iter(interactions)

    def create(self, **kwargs):
        return next(self._interactions)


class FakeClient:
    def __init__(self, interactions):
        self.interactions = FakeInteractionsAPI(interactions)


def test_agent_gemini_executes_tool_then_finalizes(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    interaction_1 = FakeInteraction(
        id_="int_1",
        steps=[FakeStep("function_call", id="call_1", name="schedule_event",
                         arguments={"summary": "Meeting with John", "time": "3pm tomorrow"})],
    )
    interaction_2 = FakeInteraction(
        id_="int_2",
        steps=[],
        output_text="Scheduled the meeting.",
    )

    fake_client = FakeClient([interaction_1, interaction_2])
    monkeypatch.setattr(agent_gemini.genai, "Client", lambda api_key=None: fake_client)

    outcome = agent_gemini.run_agent("schedule a meeting with John tomorrow at 3pm")

    assert len(outcome["steps"]) == 1
    assert outcome["steps"][0]["tool"] == "schedule_event"
    assert outcome["final_response"] == "Scheduled the meeting."
    # No credentials.json in this test environment -- confirm graceful failure.
    assert "Google isn't set up yet" in outcome["steps"][0]["result"]


def test_agent_gemini_handles_parallel_function_calls(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")

    interaction_1 = FakeInteraction(
        id_="int_1",
        steps=[
            FakeStep("function_call", id="call_1", name="schedule_event",
                     input={"summary": "Meeting", "time": "3pm"}, arguments={"summary": "Meeting", "time": "3pm"}),
            FakeStep("function_call", id="call_2", name="set_reminder",
                     arguments={"text": "Meeting", "time": "2:45pm"}),
        ],
    )
    interaction_2 = FakeInteraction(id_="int_2", steps=[], output_text="Done with both.")

    fake_client = FakeClient([interaction_1, interaction_2])
    monkeypatch.setattr(agent_gemini.genai, "Client", lambda api_key=None: fake_client)

    outcome = agent_gemini.run_agent("schedule a meeting at 3pm and remind me beforehand")

    assert [s["tool"] for s in outcome["steps"]] == ["schedule_event", "set_reminder"]
    assert outcome["final_response"] == "Done with both."


def test_agent_gemini_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    outcome = agent_gemini.run_agent("schedule a meeting")

    assert outcome["steps"] == []
    assert "GEMINI_API_KEY is not set" in outcome["final_response"]


def test_to_gemini_tool_maps_schema_correctly():
    from agent_tools import TOOLS

    gemini_tool = agent_gemini._to_gemini_tool(TOOLS[0])
    assert gemini_tool["type"] == "function"
    assert gemini_tool["name"] == TOOLS[0]["name"]
    assert gemini_tool["parameters"] == TOOLS[0]["input_schema"]
