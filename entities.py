"""
Extract simple entities (contact, phone number, message, time) from the
English-translated text.

This is regex/heuristic-based, not a real NER model -- good enough to prove
the pipeline concept, but expect it to miss or misparse anything that
doesn't roughly match these patterns. Treat it the same way as the
intent_parser in the main project: a fixed, testable starting point to
replace with something smarter later.
"""

import re

PHONE_PATTERN = re.compile(r"(\+?\d[\d\-\s]{6,}\d)")
TIME_PATTERN = re.compile(
    r"\b(\d{1,2}(:\d{2})?\s?(am|pm)|tomorrow|today|tonight|"
    r"next\s+\w+|in\s+\d+\s+(minutes?|hours?|days?))\b",
    re.IGNORECASE,
)
# Naive contact extraction: word following "to", "with", or "tell".
# Case-insensitive because translated text (from Hausa/Yoruba/Igbo) usually
# comes back lowercase, unlike typed English which tends to capitalize names.
CONTACT_PATTERN = re.compile(r"\b(?:to|with|tell)\s+([a-zA-Z]+)\b", re.IGNORECASE)


def extract_entities(english_text: str) -> dict:
    """
    Returns a dict with any of these keys that were found:
    contact, phone_number, message, time. Missing keys mean nothing matched.
    """
    entities = {}

    phone_match = PHONE_PATTERN.search(english_text)
    if phone_match:
        entities["phone_number"] = phone_match.group(1).strip()

    time_match = TIME_PATTERN.search(english_text)
    if time_match:
        entities["time"] = time_match.group(0).strip()

    contact_match = CONTACT_PATTERN.search(english_text)
    if contact_match:
        entities["contact"] = contact_match.group(1).capitalize()

    # Message content: text after "that" or "saying", a common pattern for
    # "tell X that Y" / "send a message saying Y" style instructions.
    message_match = re.search(r"\b(?:that|saying)\s+(.+)$", english_text, re.IGNORECASE)
    if message_match:
        entities["message"] = message_match.group(1).strip().rstrip(".")
    else:
        # Fall back to the whole text minus anything we already extracted --
        # better than nothing for downstream logging.
        entities["message"] = english_text.strip()

    return entities
