"""
Real Google Calendar integration for schedule_event and set_reminder.

Times come from entities.py as loose natural-language strings (e.g. "5pm",
"tomorrow", "next monday") -- dateparser turns those into an actual
datetime. If it can't parse the string at all, we fall back to one hour
from now rather than failing the whole action.
"""

import datetime as dt

import dateparser

from google_client import get_calendar_service

TIMEZONE = "Africa/Lagos"


def _parse_time(time_str: str) -> dt.datetime:
    parsed = dateparser.parse(
        time_str,
        settings={"PREFER_DATES_FROM": "future", "RETURN_AS_TIMEZONE_AWARE": False},
    )
    if parsed is None:
        parsed = dt.datetime.now() + dt.timedelta(hours=1)
    return parsed


def create_event(summary: str, time_str: str, description: str = "") -> dict:
    start = _parse_time(time_str)
    end = start + dt.timedelta(minutes=30)

    service = get_calendar_service()
    event = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start.isoformat(), "timeZone": TIMEZONE},
        "end": {"dateTime": end.isoformat(), "timeZone": TIMEZONE},
    }
    created = service.events().insert(calendarId="primary", body=event).execute()
    return {"start": start, "link": created.get("htmlLink")}


def create_reminder(summary: str, time_str: str) -> dict:
    start = _parse_time(time_str)
    end = start + dt.timedelta(minutes=15)

    service = get_calendar_service()
    event = {
        "summary": f"Reminder: {summary}",
        "start": {"dateTime": start.isoformat(), "timeZone": TIMEZONE},
        "end": {"dateTime": end.isoformat(), "timeZone": TIMEZONE},
        "reminders": {
            "useDefault": False,
            "overrides": [{"method": "popup", "minutes": 0}],
        },
    }
    created = service.events().insert(calendarId="primary", body=event).execute()
    return {"start": start, "link": created.get("htmlLink")}


def list_events(time_min_str: str = "now", time_max_str: str = None, max_results: int = 10) -> list:
    """
    Query upcoming events between time_min_str and time_max_str (both loose
    natural-language strings, same as create_event/create_reminder). Defaults
    to everything from now through the next 7 days.
    """
    time_min = _parse_time(time_min_str) if time_min_str != "now" else dt.datetime.now()
    time_max = _parse_time(time_max_str) if time_max_str else time_min + dt.timedelta(days=7)

    service = get_calendar_service()
    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min.isoformat() + "Z",
            timeMax=time_max.isoformat() + "Z",
            singleEvents=True,
            orderBy="startTime",
            maxResults=max_results,
        )
        .execute()
    )
    events = []
    for e in result.get("items", []):
        start = e["start"].get("dateTime", e["start"].get("date"))
        events.append({"summary": e.get("summary", "(no title)"), "start": start, "link": e.get("htmlLink")})
    return events


def cancel_event(search_text: str, search_window_days: int = 30) -> dict:
    """
    Find the first upcoming event whose title contains search_text
    (case-insensitive) and delete it. Returns the cancelled event's details,
    or {"found": False} if nothing matched.
    """
    service = get_calendar_service()
    time_min = dt.datetime.now()
    time_max = time_min + dt.timedelta(days=search_window_days)

    result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=time_min.isoformat() + "Z",
            timeMax=time_max.isoformat() + "Z",
            singleEvents=True,
            orderBy="startTime",
            maxResults=50,
        )
        .execute()
    )

    for e in result.get("items", []):
        if search_text.lower() in e.get("summary", "").lower():
            service.events().delete(calendarId="primary", eventId=e["id"]).execute()
            return {"found": True, "summary": e.get("summary"), "start": e["start"].get("dateTime", e["start"].get("date"))}

    return {"found": False}
