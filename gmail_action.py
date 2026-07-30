"""
Real Gmail send integration.

Used as the closest available real channel for "send_message" since real
SMS needs a telephony provider like Twilio, which isn't set up. This sends
an actual email through your Gmail account instead.
"""

import base64
from email.mime.text import MIMEText

from google_client import get_gmail_service


def send_email(to_address: str, subject: str, body: str) -> dict:
    service = get_gmail_service()

    message = MIMEText(body)
    message["to"] = to_address
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

    sent = service.users().messages().send(userId="me", body={"raw": raw}).execute()
    return {"id": sent.get("id")}
