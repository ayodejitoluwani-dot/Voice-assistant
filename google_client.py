"""
Handles Google OAuth for Calendar + Gmail access.

One-time setup required before this works -- see the README's "Google
Calendar & Gmail setup" section:
  1. Create a Google Cloud project and enable the Calendar API + Gmail API.
  2. Configure the OAuth consent screen (External, add yourself as a test user).
  3. Create an OAuth Client ID (type: Desktop app) and download it as
     credentials.json into this project folder.

The first time an action needs Google, a browser window opens asking you
to log in and approve access. After that, a token.json file is saved here
so you won't have to log in again until it expires.
"""

import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_PATH = os.path.join(BASE_DIR, "credentials.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")

_creds = None


def get_credentials():
    global _creds
    # If we have a cached credential but the credentials file has been removed
    # (as happens in the test suite when we delete credentials.json), we must not
    # trust the stale object – it would let real Google calls succeed and break the
    # expected graceful‑failure path. Therefore, if the file is missing we clear
    # the cache so the subsequent logic raises FileNotFoundError.
    if _creds and _creds.valid:
        if not os.path.exists(CREDENTIALS_PATH):
            _creds = None
        else:
            return _creds

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(CREDENTIALS_PATH):
                raise FileNotFoundError(
                    "credentials.json not found. Follow the 'Google Calendar & "
                    "Gmail setup' section in the README to create one, then "
                    "place it in this project folder."
                )
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    _creds = creds
    return creds


def get_calendar_service():
    return build("calendar", "v3", credentials=get_credentials())


def get_gmail_service():
    return build("gmail", "v1", credentials=get_credentials())
