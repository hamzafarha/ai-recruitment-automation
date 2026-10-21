"""E-mails to the candidates and interview meetings (services/mailer.py, services/meetings.py), written by
the application pipeline's "acknowledge" and "notify_candidate" steps.

Uses the ``env`` fixture of conftest.py (SQLite + fake models). SMTP is replaced by a fake server: no
e-mail leaves the machine, Google is never called (conftest sets EMAIL_MODE=outbox, MEETING_PROVIDER=jitsi).
"""

import base64
import smtplib
from datetime import UTC, datetime

import pytest

from recruitment_ai.services import mailer, meetings
from tests.fakes import ANN_CV, JOB

V1 = "/api/v1"
JUDGEMENT = {"requirements": [
    {"id": "M1", "level": "demonstrated", "evidence": "Built REST APIs in Python", "years": 5},
    {"id": "M2", "level": "demonstrated", "evidence": "REST APIs in Python with FastAPI"},
    {"id": "M3", "level": "demonstrated", "evidence": "FastAPI and PostgreSQL"}], "summary": "Strong backend profile."}


class FakeSMTP:
    """smtplib.SMTP replaced: keeps the messages; ``fail`` makes the login fail like a wrong password."""

    sent: list = []
    fail = False

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        self.host, self.port = host, port

    def __enter__(self) -> "FakeSMTP":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def ehlo(self) -> None:
        pass

    def starttls(self, context: object = None) -> None:
        pass

    def login(self, user: str, password: str) -> None:
        if FakeSMTP.fail:
            raise smtplib.SMTPAuthenticationError(535, b"Username and Password not accepted")

    def send_message(self, message: object) -> None:
        FakeSMTP.sent.append(message)


@pytest.fixture
def smtp(monkeypatch):
    FakeSMTP.sent, FakeSMTP.fail = [], False
    monkeypatch.setattr(mailer.smtplib, "SMTP", FakeSMTP)
    return FakeSMTP


def submit(client, container, job_id: str) -> str:
    container.llm.structured = {"MatchJudgement": [JUDGEMENT]}
    body = {"filename": "cv.pdf", "content_base64": base64.b64encode(ANN_CV.encode()).decode(), "job_id": job_id}
    response = client.post(f"{V1}/applications/submit", json=body)
    assert response.status_code == 202, response.text
    return response.json()["id"]


def analysis_of(client, app_id: str) -> str:
    return client.get(f"{V1}/applications/{app_id}").json()["analyses"][0]["id"]


def test_an_interview_is_emailed_with_a_video_meeting_and_a_calendar_invite(env):
    client, container, _ = env
    job = client.post(f"{V1}/jobs", json=JOB).json()
    app_id = submit(client, container, job["id"])
    ack = next(s for s in client.get(f"{V1}/applications/{app_id}").json()["steps"] if s["step"] == "acknowledge")
    assert ack["result"]["email"]["status"] == "outbox" and ack["result"]["email"]["to"] == "ann.lee@example.com"

    client.post(f"{V1}/analyses/{analysis_of(client, app_id)}/review",
                json={"decision": "interview", "reviewer": "Sam", "interview_at": "2026-10-20 10:00"})
    app = client.get(f"{V1}/applications/{app_id}").json()
    assert app["status"] == "closed"
    result = next(s for s in app["steps"] if s["step"] == "notify_candidate")["result"]
    meeting = result["meeting"]
    assert meeting["provider"] == "jitsi" and meeting["url"].startswith("https://meet.jit.si/Interview-Python-Backend")
    assert result["email"]["status"] == "outbox" and result["email"]["attachments"] == ["invite.ics"]
    assert "ics" not in meeting  # the invite is attached to the e-mail, not repeated in the step log

    listed = client.get(f"{V1}/emails").json()
    assert listed["count"] == 2 and listed["by_status"] == {"outbox": 2}
    decision = next(e for e in listed["data"] if e["kind"] == "decision")
    full = client.get(f"{V1}/emails/{decision['id']}").json()
    assert meeting["url"] in full["text"] and "When: Tuesday 20 October 2026 at 10:00" in full["text"]
    assert meeting["url"] in full["html"]
    tz, _ = meetings.zone("Africa/Tunis")
    start = datetime(2026, 10, 20, 10, 0, tzinfo=tz).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    assert f"DTSTART:{start}" in full["ics"] and "ATTENDEE;CN=Ann Lee" in full["ics"]
    assert "METHOD:REQUEST" in full["ics"]

    # every step's full output
    outputs = client.get(f"{V1}/applications/{app_id}/outputs").json()
    assert outputs["extract"]["profile"]["full_name"] == "Ann Lee"
    assert {"experience", "education", "skills", "languages"} <= set(outputs["extract"]["profile"])
    assert "FastAPI" in outputs["document"]["cv_text"]
    assert outputs["store"]["qdrant"]["count"] > 0 and outputs["store"]["candidate"]["full_name"] == "Ann Lee"
    job_trace = outputs["screen"]["jobs"][0]
    cv_call = next(c for c in job_trace["agent"]["tool_calls"] if c["tool"] == "search_cv")
    assert cv_call["results"] and {"label", "section", "score", "text"} <= set(cv_call["results"][0])
    assert job_trace["raw_answer"] and "Rules:" in outputs["screen"]["system_prompts"]["analysis"]
    assert outputs["validation"]["reviews"][0]["decision"] == "interview"
    assert outputs["notify_candidate"]["emails"][0]["meeting"]["url"] == meeting["url"]
    assert outputs["acknowledge"]["emails"][0]["kind"] == "acknowledgement"

    # right to erasure: the e-mails go with the candidate
    client.delete(f"{V1}/candidates/{app['candidate_id']}")
    assert client.get(f"{V1}/emails").json()["count"] == 0


def test_smtp_mode_sends_to_the_redirect_address_and_a_failure_stops_the_answer(env, smtp):
    client, container, _ = env
    container.settings = container.settings.model_copy(update={
        "email_mode": "smtp", "smtp_host": "smtp.test", "smtp_user": "hr@company.tn",
        "email_redirect_to": "demo@company.tn"})
    config = client.get(f"{V1}/emails/settings").json()
    assert config["mode"] == "smtp" and config["redirect_to"] == "demo@company.tn"
    assert not config["smtp_password_set"] and config["meeting_provider_used"] == "jitsi"
    job = client.post(f"{V1}/jobs", json=JOB).json()

    app_id = submit(client, container, job["id"])
    client.post(f"{V1}/analyses/{analysis_of(client, app_id)}/review", json={"decision": "shortlist"})
    assert client.get(f"{V1}/applications/{app_id}").json()["status"] == "closed"
    assert [m["To"] for m in smtp.sent] == ["demo@company.tn", "demo@company.tn"]  # never the candidate's
    assert smtp.sent[1]["Subject"].startswith("Your application")
    emails = client.get(f"{V1}/emails").json()["data"]
    assert {e["status"] for e in emails} == {"sent"} and all(e["redirected"] for e in emails)

    # a wrong SMTP password: the acknowledgement fails without stopping the workflow, the answer stops it
    smtp.fail = True
    app_id = submit(client, container, job["id"])
    app = client.get(f"{V1}/applications/{app_id}").json()
    assert app["status"] == "waiting_review"
    ack = next(s for s in app["steps"] if s["step"] == "acknowledge")["result"]["email"]
    assert ack["status"] == "failed" and "SMTPAuthenticationError" in ack["error"]
    client.post(f"{V1}/analyses/{analysis_of(client, app_id)}/review", json={"decision": "reject", "notes": "No"})
    app = client.get(f"{V1}/applications/{app_id}").json()
    assert app["status"] == "failed" and app["stage"] == "notify_candidate" and "not e-mailed" in app["error"]

    smtp.fail = False  # the password is fixed: the failed e-mail is sent again, the workflow continues
    sent = client.post(f"{V1}/emails/{ack['id']}/send").json()
    assert sent["status"] == "sent" and client.post(f"{V1}/emails/{ack['id']}/send").status_code == 409
    assert client.post(f"{V1}/applications/{app_id}/continue").json()["status"] == "closed"


def test_reserved_example_addresses_are_never_emailed(env, smtp):
    client, container, _ = env
    container.settings = container.settings.model_copy(update={"email_mode": "smtp", "smtp_host": "smtp.test"})
    job = client.post(f"{V1}/jobs", json=JOB).json()
    app_id = submit(client, container, job["id"])
    client.post(f"{V1}/analyses/{analysis_of(client, app_id)}/review", json={"decision": "shortlist"})
    assert client.get(f"{V1}/applications/{app_id}").json()["status"] == "closed"  # skipped is not a failure
    assert smtp.sent == []
    emails = client.get(f"{V1}/emails").json()["data"]
    assert {e["status"] for e in emails} == {"skipped"} and "reserved" in emails[0]["error"]


def test_the_meeting_provider_and_its_fallback(monkeypatch, test_settings):
    settings = test_settings.model_copy(update={"meeting_provider": "auto"})
    monkeypatch.setattr(meetings, "google_ready", lambda s: True)
    monkeypatch.setattr(meetings, "google_event", lambda *a, **k: {"url": "https://meet.google.com/abc-defg-hij",
                                                                  "event_id": "e1", "calendar_link": "https://cal/e1"})
    made = meetings.create_meeting(settings, "Interview", "Structured interview", "2026-11-02 14:30", "a@b.tn")
    assert made["provider"] == "google" and made["url"].startswith("https://meet.google.com/")
    assert made["invitation_by"] == "google" and made["end"].startswith("2026-11-02T15:15")

    def broken(*args: object, **kwargs: object) -> dict:
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(meetings, "google_event", broken)
    fallback = meetings.create_meeting(settings, "Interview", "", "2026-11-02T14:30", "a@b.tn", "Data Analyst")
    assert fallback["provider"] == "jitsi" and "quota exceeded" in fallback["notes"][-1]
    assert "Interview-Data-Analyst-" in fallback["url"]
    with pytest.raises(RuntimeError, match="Google Calendar could not create"):
        meetings.create_meeting(settings.model_copy(update={"meeting_provider": "google"}), "Interview", "",
                                "2026-11-02 14:30", "a@b.tn")
    with pytest.raises(ValueError, match="not understood"):
        meetings.parse_interview_time("next Monday", UTC)


def test_the_calendar_invite_follows_rfc_5545():
    meeting = {"title": "Interview — Data, AI; and more", "start": "2026-11-02T14:30:00+01:00",
               "end": "2026-11-02T15:15:00+01:00", "url": "https://meet.jit.si/Interview-x"}
    ics = meetings.ics_invite(meeting, "A long description " * 12, "hr@company.tn", "Recruiting team",
                              "ann@lee.tn", "Ann Lee")
    lines = ics.split("\r\n")
    assert lines[0] == "BEGIN:VCALENDAR" and ics.endswith("END:VCALENDAR\r\n")
    assert all(len(line.encode()) <= 75 for line in lines)
    assert "DTSTART:20261102T133000Z" in lines and "SUMMARY:Interview — Data\\, AI\\; and more" in lines
    assert any(line.startswith("DESCRIPTION:") for line in lines) and any(line.startswith(" ") for line in lines)
