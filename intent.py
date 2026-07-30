"""Rule-based intent classification on the English-translated text.

Matches the four supported actions: schedule_event, send_message,
phone_call, set_reminder. Deliberately simple (keyword matching) so it's
easy to test and debug, same reasoning as the main project report.
"""

INTENTS = ["schedule_event", "send_message", "phone_call", "set_reminder"]

PATTERNS = {
    "schedule_event": ["schedule", "calendar", "meeting", "appointment", "book a"],
    "send_message": ["send a message", "send message", "tell", "message", "text", "mail", "email"],
    "phone_call": ["call", "phone", "dial", "ring"],
    "set_reminder": ["remind", "reminder", "don't forget", "remember to"],
}


class NoMatchError(ValueError):
    pass


def classify_intent(english_text: str) -> str:
    lowered = english_text.lower().strip()

    for intent in INTENTS:
        for phrase in PATTERNS[intent]:
            if phrase in lowered:
                return intent

    raise NoMatchError(f"No intent pattern matched: {english_text!r}")
