"""Lab 06 Part D - agent with Google Calendar tools."""
import os
from datetime import datetime, timedelta, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

from agent_memory import build_agent, chat_loop, init_db, search_memory
from agent_tools import calculator

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
LK = timezone(timedelta(hours=5, minutes=30))   # Sri Lanka has no daylight saving


def get_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)    # opens the browser to log in once
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


@tool
def get_todays_schedule() -> str:
    """List today's events from the user's Google Calendar."""
    start = datetime.now(LK).replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    events = get_service().events().list(
        calendarId="primary", timeMin=start.isoformat(), timeMax=end.isoformat(),
        singleEvents=True, orderBy="startTime",
    ).execute().get("items", [])
    if not events:
        return "No events today."
    lines = []
    for e in events:
        when = e["start"].get("dateTime", e["start"].get("date"))
        time_text = when[11:16] if "T" in when else "all day"
        lines.append(f"{time_text}  {e.get('summary', '(no title)')}")
    return "\n".join(lines)


@tool
def create_event(title: str, start: str, duration_minutes: int = 60) -> str:
    """Create a calendar event. start must look like 2026-11-26 15:00 (24-hour, Sri Lanka time)."""
    try:
        begin = datetime.strptime(start, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Error: start must look like 2026-11-26 15:00"
    if input(f"\n  Create '{title}' on {begin:%d %b %Y at %H:%M}? (y/n) ").strip().lower() != "y":
        return "The user cancelled, so no event was created."
    finish = begin + timedelta(minutes=duration_minutes)
    body = {
        "summary": title,
        "start": {"dateTime": begin.isoformat(), "timeZone": "Asia/Colombo"},
        "end": {"dateTime": finish.isoformat(), "timeZone": "Asia/Colombo"},
    }
    get_service().events().insert(calendarId="primary", body=body).execute()
    return f"Created '{title}' on {begin:%A %d %B at %H:%M}."


def system_prompt():
    now = datetime.now(LK)
    return (
        f"You are a helpful assistant. Today is {now:%A %d %B %Y} and the time is {now:%H:%M} in Sri Lanka. "
        "Use get_todays_schedule to check today's calendar. Use create_event to add events, converting "
        "phrases like 'tomorrow at 3pm' into YYYY-MM-DD HH:MM. Use calculator for maths and "
        "search_memory for things the user told you earlier. Keep answers short."
    )


if __name__ == "__main__":
    init_db()
    tools = [calculator, search_memory, get_todays_schedule, create_event]
    chat_loop(build_agent(tools, system_prompt()))
