"""PostgreSQL tables of the application (diagram part 8, Data Storage).

Profiles, job requirements and reports are stored whole as JSON (they are produced and read as
documents by the pipeline); the columns next to them are the fields the app filters and sorts on.
Works on PostgreSQL and on SQLite (unit tests).

- candidates: the structured CV (CandidateProfile) + the CV text used as evidence;
- jobs: the job and its requirements (M1.. must-haves, N1.. nice-to-haves);
- applications: one application through the recruitment pipeline, from the CV received to the
  final decision, with the log of every step (services/pipeline.py);
- analyses: one candidate-job report (MatchReport), with its review status;
- reviews: the recruiter's decisions on an analysis (human validation, part 9);
- notifications: events shown in the app and sent to n8n;
- emails: the e-mails to the candidates (acknowledgement, answer after the decision, interview invitation
  with its meeting link), sent by SMTP or kept in the outbox;
- agent_runs: questions asked to the AI assistant, with the tools it used (traceability).
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from recruitment_ai.database.connection import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    external_id = Column(String(100), index=True, nullable=True)  # e.g. "syn01" for demo data (no duplicates)
    full_name = Column(String(255), nullable=True)
    email = Column(String(255), index=True, nullable=True)
    phone = Column(String(50), nullable=True)
    location = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    profile = Column(JSON, nullable=False, default=dict)
    cv_text = Column(Text, nullable=True)
    cv_filename = Column(String(255), nullable=True)
    source = Column(String(30), nullable=False, default="upload")  # upload | demo | n8n | portal
    extraction = Column(JSON, nullable=True)  # model, timings, input type...
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    analyses = relationship("Analysis", back_populates="candidate", cascade="all, delete-orphan")
    applications = relationship("Application", back_populates="candidate", cascade="all, delete-orphan")
    emails = relationship("Email", cascade="all, delete-orphan")  # right to erasure: deleted with the candidate


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    external_id = Column(String(100), index=True, nullable=True)
    title = Column(String(255), nullable=False)
    company = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    employment_type = Column(String(100), nullable=True)
    role = Column(String(80), nullable=True)  # knowledge-base role key, e.g. "ml_engineer"
    description = Column(Text, nullable=False, default="")
    requirements = Column(JSON, nullable=False, default=list)
    status = Column(String(20), nullable=False, default="open")
    source = Column(String(30), nullable=False, default="manual")  # manual | demo | n8n
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    analyses = relationship("Analysis", back_populates="job", cascade="all, delete-orphan")
    applications = relationship("Application", back_populates="job")


class Application(Base):
    """One application through the recruitment pipeline (CV received -> final decision)."""

    __tablename__ = "applications"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=True, index=True)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True)  # None = spontaneous
    source = Column(String(30), nullable=False, default="upload")  # upload | portal | email | n8n
    filename = Column(String(255), nullable=True)
    orchestrator = Column(String(20), nullable=False, default="api")  # n8n | api: who runs the next steps
    status = Column(String(30), nullable=False, default="running")  # running | waiting_review | closed | failed
    stage = Column(String(30), nullable=False, default="intake")  # the step running, waiting or failed
    steps = Column(JSON, nullable=False, default=list)  # one entry per step: status, timings, result, error
    work = Column(JSON, nullable=True)  # what passes between steps (CV text, profile, reports), removed once stored
    decision = Column(String(30), nullable=True)  # final decision: interview | shortlist | reject
    resume_url = Column(String(1000), nullable=True)  # n8n Wait node: the recruiter's decision resumes the workflow
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="applications")
    job = relationship("Job", back_populates="applications")
    analyses = relationship("Analysis", back_populates="application", order_by="Analysis.created_at")
    emails = relationship("Email", back_populates="application", cascade="all, delete-orphan",
                          order_by="Email.created_at")


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=False, index=True)
    job_id = Column(String(36), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    # the application whose screening produced it (None: analysis asked from the Matching page)
    application_id = Column(String(36), ForeignKey("applications.id", ondelete="SET NULL"), nullable=True, index=True)
    method = Column(String(20), nullable=False, default="llm")  # llm | agent | semantic
    grade = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    must_coverage = Column(Float, nullable=True)
    nice_coverage = Column(Float, nullable=True)
    action = Column(String(30), nullable=True)
    report = Column(JSON, nullable=False, default=dict)
    # pending_review | shortlisted | interview | on_hold | rejected
    status = Column(String(30), nullable=False, default="pending_review")
    final_grade = Column(Integer, nullable=True)  # grade after the recruiter's adjustment
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="analyses")
    job = relationship("Job", back_populates="analyses")
    application = relationship("Application", back_populates="analyses")
    reviews = relationship("Review", back_populates="analysis", cascade="all, delete-orphan",
                           order_by="Review.created_at")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False, index=True)
    reviewer = Column(String(100), nullable=False, default="recruiter")
    decision = Column(String(30), nullable=False)  # shortlist | interview | hold | reject
    adjusted_grade = Column(Integer, nullable=True)
    interview_at = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis = relationship("Analysis", back_populates="reviews")


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    event = Column(String(60), nullable=False)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=True)
    payload = Column(JSON, nullable=True)
    source = Column(String(30), nullable=False, default="app")  # app | n8n
    read = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class Email(Base):
    """An e-mail to a candidate (acknowledgement, answer after the decision), sent or kept in the outbox."""

    __tablename__ = "emails"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    application_id = Column(String(36), ForeignKey("applications.id", ondelete="CASCADE"), nullable=True, index=True)
    candidate_id = Column(String(36), ForeignKey("candidates.id", ondelete="CASCADE"), nullable=True, index=True)
    kind = Column(String(30), nullable=False)  # acknowledgement | decision
    decision = Column(String(30), nullable=True)  # interview | shortlist | reject (kind "decision")
    to_address = Column(String(255), nullable=True)  # where it was sent (EMAIL_REDIRECT_TO in demos)
    candidate_address = Column(String(255), nullable=True)  # the candidate's address from the CV
    subject = Column(String(255), nullable=False)
    text = Column(Text, nullable=False)
    html = Column(Text, nullable=True)
    status = Column(String(20), nullable=False)  # sent | outbox | failed | skipped
    mode = Column(String(20), nullable=False)  # smtp | outbox
    error = Column(Text, nullable=True)
    meeting = Column(JSON, nullable=True)  # interview: provider, link, start, end, calendar event
    attachments = Column(JSON, nullable=True)  # names of the attached files (invite.ics)
    sent_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    application = relationship("Application", back_populates="emails")


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    steps = Column(JSON, nullable=True)
    sources = Column(JSON, nullable=True)
    seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
