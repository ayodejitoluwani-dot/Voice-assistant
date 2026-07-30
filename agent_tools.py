"""
Shared tool catalog and execution logic, used by BOTH agent backends
(agent_claude.py and agent_ncair.py) so there's exactly one real
implementation of each action regardless of which model is doing the
reasoning.

TOOLS is written in Anthropic's tool-schema format since that's the
richer of the two formats (has typed input_schema) -- agent_ncair.py
renders this same list into a plain-text description for its prompt,
rather than keeping a second, possibly-drifting copy.
"""

from calendar_action import create_event, create_reminder, list_events, cancel_event
from contacts import lookup_email
from gmail_action import send_email as _send_email

TOOLS = [
    {
        "name": "schedule_event",
        "description": (
            "Create a real event on the user's Google Calendar. Use for meetings, "
            "appointments, or any 'schedule X' request."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Short event title, e.g. 'Meeting with John'"},
                "time": {"type": "string", "description": "Natural language time, e.g. '3pm tomorrow', 'next monday'"},
                "description": {"type": "string", "description": "Optional longer description/notes for the event"},
            },
            "required": ["summary", "time"],
        },
    },
    {
        "name": "set_reminder",
        "description": (
            "Create a real Google Calendar reminder (a calendar event with a popup "
            "notification) at a specific time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What to be reminded of"},
                "time": {"type": "string", "description": "Natural language time for the reminder to fire"},
            },
            "required": ["text", "time"],
        },
    },
    {
        "name": "send_email",
        "description": (
            "Send a real email via Gmail to a known contact. The contact name must "
            "already exist in contacts.json -- if it's not found, this will report "
            "that rather than sending anything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "contact": {"type": "string", "description": "Contact's name, e.g. 'Musa'"},
                "message": {"type": "string", "description": "The message body to send"},
            },
            "required": ["contact", "message"],
        },
    },
    {
        "name": "phone_call",
        "description": (
            "Log a phone call request. NOTE: this does not actually dial anyone -- "
            "there is no telephony provider (e.g. Twilio) configured. It only "
            "records that a call was requested."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "contact": {"type": "string", "description": "Who to call, by name"},
                "phone_number": {"type": "string", "description": "Phone number, if one was mentioned"},
            },
            "required": [],
        },
    },
    {
        "name": "list_calendar_events",
        "description": "Look up the user's upcoming Google Calendar events in a given time range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "time_min": {"type": "string", "description": "Start of range, natural language, e.g. 'now', 'tomorrow'"},
                "time_max": {"type": "string", "description": "End of range, natural language, e.g. 'next friday'"},
            },
            "required": [],
        },
    },
    {
        "name": "cancel_calendar_event",
        "description": "Find and cancel (delete) an upcoming calendar event whose title matches the given text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "search_text": {"type": "string", "description": "Text to match against event titles, e.g. 'meeting with John'"},
            },
            "required": ["search_text"],
        },
    },
]


def run_tool(name: str, tool_input: dict) -> str:
    """Executes one tool call against the real integrations and returns a
    plain-text result the calling model can read back."""
    try:
        if name == "schedule_event":
            result = create_event(tool_input["summary"], tool_input["time"], tool_input.get("description", ""))
            when = result["start"].strftime("%a %d %b, %I:%M %p")
            return f"Scheduled '{tool_input['summary']}' for {when}. Link: {result['link']}"

        if name == "set_reminder":
            result = create_reminder(tool_input["text"], tool_input["time"])
            when = result["start"].strftime("%a %d %b, %I:%M %p")
            return f"Reminder set for {when}: '{tool_input['text']}'. Link: {result['link']}"

        if name == "send_email":
            contact = tool_input["contact"]
            email = lookup_email(contact)
            if not email:
                return f"Could not send -- no email on file for '{contact}'. Add them to contacts.json first."
            _send_email(email, subject="Message from Voice Assistant", body=tool_input["message"])
            return f"Email sent to {contact} ({email}): '{tool_input['message']}'"

        if name == "phone_call":
            target = tool_input.get("phone_number") or tool_input.get("contact") or "an unspecified number"
            return f"Logged only, not actually dialled -- would call {target}. Real calls need a telephony provider like Twilio."

        if name == "list_calendar_events":
            events = list_events(tool_input.get("time_min", "now"), tool_input.get("time_max"))
            if not events:
                return "No upcoming events found in that range."
            lines = [f"- {e['summary']} at {e['start']}" for e in events]
            return "Upcoming events:\n" + "\n".join(lines)

        if name == "cancel_calendar_event":
            result = cancel_event(tool_input["search_text"])
            if result["found"]:
                return f"Cancelled '{result['summary']}' (was at {result['start']})."
            return f"No upcoming event matching '{tool_input['search_text']}' was found."

        return f"Unknown tool '{name}'."

    except FileNotFoundError as e:
        return f"Could not complete '{name}' -- Google isn't set up yet ({e})."
    except Exception as e:
        return f"Could not complete '{name}' -- error: {e}"


def tools_as_text() -> str:
    """Renders TOOLS as a plain-text catalog for prompting a model (like
    NCAIR's N-ATLaS) that doesn't have structured tool-calling support."""
    lines = []
    for tool in TOOLS:
        props = tool["input_schema"]["properties"]
        required = set(tool["input_schema"].get("required", []))
        arg_descriptions = ", ".join(
            f"{name}{'*' if name in required else ''}: {info['description']}"
            for name, info in props.items()
        )
        lines.append(f"- {tool['name']}({arg_descriptions})\n  {tool['description']}")
    return "\n".join(lines)
