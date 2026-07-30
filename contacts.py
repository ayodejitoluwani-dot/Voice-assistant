"""
Simple local contact book: maps a spoken name (e.g. "Musa") to a real
email address, since entities.py only extracts a name, not an email.

Edit contacts.json to add real contacts. Names are matched
case-insensitively.
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTACTS_PATH = os.path.join(BASE_DIR, "contacts.json")


def load_contacts() -> dict:
    if not os.path.exists(CONTACTS_PATH):
        return {}
    with open(CONTACTS_PATH) as f:
        return json.load(f)


def lookup_email(name: str):
    if not name:
        return None
    return load_contacts().get(name.strip().lower())
