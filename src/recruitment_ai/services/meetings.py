"""Interview meetings (diagram part 7): the video meeting of an interview a recruiter scheduled.

Providers (MEETING_PROVIDER):
- google: a Google Calendar event with a Google Meet link; Google e-mails the invitation to the
  candidate (scope calendar.events, authorised once with ``python scripts/google_calendar_setup.py``);
- jitsi: a Jitsi Meet room (https://meet.jit.si/<random name>), no account or setup;
- auto (default): google when it is set up (its token file exists), else jitsi. A Google error then
  falls back to jitsi, with a note, instead of stopping the workflow;
- none: no link (the e-mail gives the date and time only).
Every meeting also gets an iCalendar invite (``invite.ics``) attached to our e-mail, except when
Google already sent its own invitation.
"""

import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from recruitment_ai.utils.logging import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[3]
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
JITSI_URL = "https://meet.jit.si"
TIME_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")


def zone(name: str) -> tuple[Any, str | None]:
    """The time zone, or UTC with a note (Windows needs the ``tzdata`` package for IANA names)."""
    try:
        return ZoneInfo(name), None
    except (ZoneInfoNotFoundError, ValueError):
        return UTC, f"time zone {name!r} unknown on this machine (pip install tzdata): UTC used"


def parse_interview_time(text: str, tz: Any) -> datetime:
    """'2026-10-25 10:00' (what the Human validation page sends) as an aware datetime."""
    value = (text or "").strip()
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=tz)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Interview date {text!r} not understood: use YYYY-MM-DD HH:MM") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=tz)


def project_path(value: str) -> Path:
    """A path of the settings: relative ones are in the project folder."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def google_ready(settings: Any) -> bool:
    """Google Calendar can be used: the authorisation token exists and the Google libraries are installed."""
    if not project_path(settings.google_token_file).is_file():
        return False
    try:
        import google.oauth2.credentials  # noqa: F401
        import googleapiclient.discovery  # noqa: F401
    except ImportError:
        return False
    return True


def resolved_provider(settings: Any) -> str:
    """The provider used for the next meeting."""
    provider = settings.meeting_provider
    if provider == "auto":
        return "google" if google_ready(settings) else "jitsi"
    return provider


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-")[:40] or "Interview"


def jitsi_link(job_title: str | None) -> str:
    """A hard-to-guess room name: anyone with the link can join, like a Meet link."""
    return f"{JITSI_URL}/Interview-{_slug(job_title or '')}-{secrets.token_hex(4)}"


def _google_credentials(settings: Any) -> Any:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    path = project_path(settings.google_token_file)
    credentials = Credentials.from_authorized_user_file(str(path), GOOGLE_SCOPES)
    if not credentials.valid:
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
            path.write_text(credentials.to_json(), encoding="utf-8")
        else:
            raise RuntimeError("The Google authorisation is no longer valid: run "
                               "python scripts/google_calendar_setup.py again")
    return credentials


def google_event(settings: Any, title: str, description: str, start: datetime, end: datetime,
                 attendee: str | None) -> dict[str, Any]:
    """A Google Calendar event with a Meet link; Google e-mails the invitation to the attendee."""
    from googleapiclient.discovery import build

    service = build("calendar", "v3", credentials=_google_credentials(settings), cache_discovery=False)
    body: dict[str, Any] = {
        "summary": title, "description": description,
        "start": {"dateTime": start.isoformat(), "timeZone": settings.interview_timezone},
        "end": {"dateTime": end.isoformat(), "timeZone": settings.interview_timezone},
        "conferenceData": {"createRequest": {"requestId": uuid.uuid4().hex,
                                             "conferenceSolutionKey": {"type": "hangoutsMeet"}}},
        "reminders": {"useDefault": True},
    }
    if attendee:
        body["attendees"] = [{"email": attendee}]
    created = service.events().insert(calendarId=settings.google_calendar_id, body=body, conferenceDataVersion=1,
                                      sendUpdates="all" if attendee else "none").execute()
    link = created.get("hangoutLink") or next(
        (p.get("uri") for p in (created.get("conferenceData") or {}).get("entryPoints") or []
         if p.get("entryPointType") == "video"), None)
    return {"url": link, "event_id": created.get("id"), "calendar_link": created.get("htmlLink")}


def create_meeting(settings: Any, title: str, description: str, start_text: str, attendee: str | None,
                   job_title: str | None = None) -> dict[str, Any]:
    """The interview's meeting: provider, link, start and end (ISO, with the time zone) and notes."""
    tz, tz_note = zone(settings.interview_timezone)
    start = parse_interview_time(start_text, tz)
    end = start + timedelta(minutes=settings.interview_minutes)
    notes = [tz_note] if tz_note else []
    provider = resolved_provider(settings)
    meeting: dict[str, Any] = {"provider": provider, "url": None, "title": title, "start": start.isoformat(),
                               "end": end.isoformat(), "timezone": settings.interview_timezone,
                               "minutes": settings.interview_minutes, "event_id": None, "calendar_link": None,
                               "invitation_by": None}
    if provider == "google":
        try:
            meeting.update(google_event(settings, title, description, start, end, attendee))
            meeting["invitation_by"] = "google" if attendee else None
        except Exception as exc:
            if settings.meeting_provider != "auto":
                raise RuntimeError(f"Google Calendar could not create the meeting ({type(exc).__name__}: "
                                   f"{str(exc)[:200]}). Check the Google setup, then continue the workflow") from exc
            logger.warning(f"Google Calendar failed, Jitsi used instead: {exc}")
            notes.append(f"Google Calendar failed ({type(exc).__name__}: {str(exc)[:150]}): a Jitsi link was used")
            provider = meeting["provider"] = "jitsi"
    if provider == "jitsi":
        meeting["url"] = jitsi_link(job_title)
    if notes:
        meeting["notes"] = notes
    return meeting


# --------------------------------------------------------------------------- iCalendar invite


def _escape(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> str:
    """RFC 5545: lines of at most 75 octets, continued with a space."""
    out, current = [], ""
    for ch in line:
        if len((current + ch).encode("utf-8")) > (75 if not out else 74):
            out.append(current)
            current = ch
        else:
            current += ch
    out.append(current)
    return "\r\n ".join(out)


def _utc(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def ics_invite(meeting: dict[str, Any], description: str, organizer: str, organizer_name: str,
               attendee: str | None, attendee_name: str | None = None) -> str:
    """The calendar invitation attached to the e-mail (opens in Outlook, Google Calendar, Apple Calendar)."""
    lines = ["BEGIN:VCALENDAR", "PRODID:-//AI Recruitment Automation//Interview//EN", "VERSION:2.0",
             "CALSCALE:GREGORIAN", "METHOD:REQUEST", "BEGIN:VEVENT",
             f"UID:{uuid.uuid4()}@recruitment-ai", f"DTSTAMP:{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
             f"DTSTART:{_utc(meeting['start'])}", f"DTEND:{_utc(meeting['end'])}",
             f"SUMMARY:{_escape(meeting.get('title') or 'Interview')}", f"DESCRIPTION:{_escape(description)}"]
    if meeting.get("url"):
        lines += [f"LOCATION:{_escape(meeting['url'])}", f"URL:{meeting['url']}"]
    lines.append(f"ORGANIZER;CN={_escape(organizer_name)}:mailto:{organizer}")
    if attendee:
        lines.append(f"ATTENDEE;CN={_escape(attendee_name or attendee)};ROLE=REQ-PARTICIPANT;"
                     f"PARTSTAT=NEEDS-ACTION;RSVP=TRUE:mailto:{attendee}")
    lines += ["STATUS:CONFIRMED", "SEQUENCE:0", "BEGIN:VALARM", "TRIGGER:-PT15M", "ACTION:DISPLAY",
              "DESCRIPTION:Interview in 15 minutes", "END:VALARM", "END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"
