"""
Action handlers for the four supported intents.

schedule_event and set_reminder create real Google Calendar events.
send_message sends a real email via Gmail -- the closest available real
channel, since actual SMS needs a paid telephony provider (Twilio) that
isn't set up here. phone_call remains logged-only: there's no free
Google-based way to place a real phone call; that needs Twilio (or
similar) with a purchased phone number.

Every handler degrades gracefully (falls back to a logged, honest message)
if Google isn't set up yet (missing credentials.json) or a real call
fails, rather than crashing the whole pipeline.
"""

from contacts import lookup_email
from calendar_action import create_event, create_reminder
from gmail_action import send_email


def handle_schedule_event(entities: dict) -> str:
    time = entities.get("time", "an unspecified time")
    contact = entities.get("contact")
    summary = f"Meeting with {contact}" if contact else "Scheduled event"

    try:
        # Keep the original time string for user‑friendly output.
        original_time = time
        result = create_event(summary, time, description=entities.get("message", ""))
        when = result["start"].strftime("%a %d %b, %I:%M %p")
        # Include the original human‑readable time for clarity.
        return f"Scheduled '{summary}' for {when} (parsed from '{original_time}'). {result['link']}"
    except FileNotFoundError as e:
        return f"[not scheduled -- Google not set up yet: {e}] Would schedule '{summary}' at {time}."
    except Exception as e:
        return f"[not scheduled -- error: {e}] Would schedule '{summary}' at {time}."


def handle_send_message(entities: dict) -> str:
    contact = entities.get("contact")
    message = entities.get("message", "")
    email = lookup_email(contact)

    if not email:
        return (
            f"[not sent -- no email on file for {contact!r}] Message: {message!r}. "
            f"Add {contact} to contacts.json to enable real sending."
        )

    try:
        send_email(email, subject="Message from Voice Assistant", body=message)
        return f"Sent email to {contact} ({email}): {message!r}"
    except FileNotFoundError as e:
        return f"[not sent -- Google not set up yet: {e}] Message to {contact}: {message!r}"
    except Exception as e:
        return f"[not sent -- error: {e}] Message to {contact}: {message!r}"


def handle_phone_call(entities: dict) -> str:
    contact = entities.get("contact")
    phone = entities.get("phone_number")
    target = phone or contact or "an unspecified number"
    return (
        f"[logged, not actually dialled] Would call {target}. "
        f"Real calls need a telephony provider like Twilio, which isn't set up yet."
    )


def handle_set_reminder(entities: dict) -> str:
    time = entities.get("time", "an unspecified time")
    message = entities.get("message", "")

    try:
        # Keep the original time string for user‑friendly output.
        original_time = time
        result = create_reminder(message or "Reminder", time)
        when = result["start"].strftime("%a %d %b, %I:%M %p")
        return f"Reminder set for {when} (parsed from '{original_time}'): {message!r}. {result['link']}"
    except FileNotFoundError as e:
        return f"[not set -- Google not set up yet: {e}] Would remind at {time}: {message!r}"
    except Exception as e:
        return f"[not set -- error: {e}] Would remind at {time}: {message!r}"


HANDLERS = {
    "schedule_event": handle_schedule_event,
    "send_message": handle_send_message,
    "phone_call": handle_phone_call,
    "set_reminder": handle_set_reminder,
}


def execute(intent: str, entities: dict) -> str:
    if intent not in HANDLERS:
        return f"No handler for intent '{intent}'."
    return HANDLERS[intent](entities)
