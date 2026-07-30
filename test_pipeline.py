"""
Tests for everything that doesn't require the ASR model or network access:
intent classification, entity extraction, action handlers, task logging,
and the full pipeline using text input (which skips ASR).

Run with: python -m pytest test_pipeline.py -v
"""

import os
import shutil

import pytest

from intent import classify_intent, NoMatchError
from entities import extract_entities
from actions import execute
from task_logger import save_task, load_all_tasks


# --- Intent classification -------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("schedule a meeting with Musa tomorrow", "schedule_event"),
    ("send a message to Musa that I will be late", "send_message"),
    ("call Musa now", "phone_call"),
    ("remind me to submit the report at 5pm", "set_reminder"),
])
def test_classify_intent(text, expected):
    assert classify_intent(text) == expected


def test_classify_intent_raises_on_no_match():
    with pytest.raises(NoMatchError):
        classify_intent("this is unrelated gibberish xyz123")


# --- Entity extraction ------------------------------------------------------

def test_extract_entities_finds_contact_and_message():
    entities = extract_entities("send a message to Musa that I will be late")
    assert entities.get("contact") == "Musa"
    assert "late" in entities.get("message", "")


def test_extract_entities_finds_phone_number():
    entities = extract_entities("call 080-1234-5678 now")
    assert "080" in entities.get("phone_number", "")


def test_extract_entities_finds_time():
    entities = extract_entities("remind me tomorrow to call the vendor")
    assert entities.get("time", "").lower() == "tomorrow"


# --- Action handlers ---------------------------------------------------
# schedule_event, send_message, and set_reminder now attempt real Google
# Calendar/Gmail calls. In this test environment there's no credentials.json,
# so these confirm the graceful fallback path (no crash, honest message,
# original details preserved) rather than a live send. phone_call has no
# real integration at all (needs Twilio), so it stays logged-only always.

def test_execute_send_message():
    response = execute("send_message", {"contact": "Musa", "message": "I will be late"})
    assert "Musa" in response
    assert "late" in response


def test_execute_phone_call():
    response = execute("phone_call", {"contact": "Musa"})
    assert "logged, not actually dialled" in response


def test_execute_schedule_event():
    response = execute("schedule_event", {"time": "5pm", "contact": "Musa"})
    assert "5pm" in response


def test_execute_set_reminder():
    response = execute("set_reminder", {"time": "5pm", "message": "submit the report"})
    assert "5pm" in response


# --- Task logging --------------------------------------------------------

def test_save_and_load_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    task_id = save_task({"intent": "send_message", "response": "test"})
    assert task_id

    tasks = load_all_tasks()
    assert len(tasks) == 1
    assert tasks[0]["intent"] == "send_message"


# --- Full pipeline with text input (no ASR needed) -----------------------

def test_full_pipeline_with_text_input_rules_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from pipeline import run_pipeline

    result = run_pipeline(lang="hausa", text="call Musa now", return_details=True, mode="rules")
    assert result["intent"] == "phone_call"
    assert "logged, not actually dialled" in result["response"]
    assert result["confidence"] == 1.0
