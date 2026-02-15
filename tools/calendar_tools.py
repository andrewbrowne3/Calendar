"""
Calendar API tool functions.
Each function corresponds to a tool the agent can call via THINK/ACT/OBSERVE.
"""

import json
from datetime import datetime

import requests

CALENDAR_API_BASE = "http://localhost:2011"

DEFAULT_EMAIL = "andrewbrowne161@gmail.com"
DEFAULT_PASSWORD = "Sierra-Ciara$"


def get_current_datetime() -> str:
    now = datetime.now()
    iso_format = now.strftime("%Y-%m-%dT%H:%M:%S%z")
    readable = now.strftime("%A, %B %d, %Y at %I:%M %p")
    return f"Current date and time: {iso_format}\n({readable})"


def login(email: str = None, password: str = None) -> str:
    email = email or DEFAULT_EMAIL
    password = password or DEFAULT_PASSWORD
    try:
        result = requests.post(
            f"{CALENDAR_API_BASE}/api/auth/login/",
            json={"email": email, "password": password},
            headers={"Content-Type": "application/json"},
        )
        result.raise_for_status()
        data = result.json()
        if "access" in data:
            return json.dumps({"token": data["access"], "message": "Login successful"})
        return json.dumps(data)
    except requests.exceptions.RequestException as e:
        return f"ERROR: Login failed - {e}"


def get_calendars(token: str) -> str:
    try:
        result = requests.get(
            f"{CALENDAR_API_BASE}/api/calendars/",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        result.raise_for_status()
        return json.dumps(result.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"ERROR: Get calendars failed - {e}"


def get_events(token: str) -> str:
    try:
        result = requests.get(
            f"{CALENDAR_API_BASE}/api/events/",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        result.raise_for_status()
        return json.dumps(result.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"ERROR: Get events failed - {e}"


def post_event(
    token: str,
    calendar_id: str,
    title: str,
    start_time: str,
    end_time: str,
    description: str = "",
    reminder_minutes: list = None,
) -> str:
    try:
        event_data = {
            "calendar": calendar_id,
            "title": title,
            "start_time": start_time,
            "end_time": end_time,
            "description": description,
        }
        if reminder_minutes:
            event_data["reminder_minutes"] = reminder_minutes
        result = requests.post(
            f"{CALENDAR_API_BASE}/api/events/",
            json=event_data,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        result.raise_for_status()
        return json.dumps(result.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"ERROR: Create event failed - {e}"


def patch_event(
    token: str,
    event_id: str,
    title: str = None,
    start_time: str = None,
    end_time: str = None,
    description: str = None,
    reminder_minutes: list = None,
) -> str:
    try:
        updates = {}
        if title is not None:
            updates["title"] = title
        if start_time is not None:
            updates["start_time"] = start_time
        if end_time is not None:
            updates["end_time"] = end_time
        if description is not None:
            updates["description"] = description
        if reminder_minutes is not None:
            updates["reminder_minutes"] = reminder_minutes

        result = requests.patch(
            f"{CALENDAR_API_BASE}/api/events/{event_id}/",
            json=updates,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        result.raise_for_status()
        return json.dumps(result.json(), indent=2)
    except requests.exceptions.RequestException as e:
        return f"ERROR: Update event failed - {e}"


def delete_event(token: str, event_id: str) -> str:
    try:
        result = requests.delete(
            f"{CALENDAR_API_BASE}/api/events/{event_id}/",
            headers={"Authorization": f"Bearer {token}"},
        )
        result.raise_for_status()
        return f"Event {event_id} deleted successfully"
    except requests.exceptions.RequestException as e:
        return f"ERROR: Delete event failed - {e}"


def final_answer(answer: str) -> str:
    return answer


TOOL_REGISTRY = {
    "get_current_datetime": get_current_datetime,
    "login": login,
    "get_calendars": get_calendars,
    "get_events": get_events,
    "post_event": post_event,
    "patch_event": patch_event,
    "delete_event": delete_event,
    "final_answer": final_answer,
}
