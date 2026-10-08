"""SQLAlchemy models for recruitment automation."""

import uuid
from datetime import datetime

from sqlalchemy import (
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
    """Generate string UUID."""
    return str(uuid.uuid4())


class Candidate(Base):
    """Candidate entity."""

    __tablename__ = "candidates"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    phone = Column(String(50), nullable=True)
    location = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    raw_cv_path = Column(String(512), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    documents = relationship(
        "CandidateDocument", back_populates="candidate", cascade="all, delete-orphan"
    )
    educations = relationship(
        "Education", back_populates="candidate", cascade="all, delete-orphan"
    )
    experiences = relationship(
        "Experience", back_populates="candidate", cascade="all, delete-orphan"
    )
    skills = relationship(
        "Skill", back_populates="candidate", cascade="all, delete-orphan"
    )
    certifications = relationship(
        "Certification", back_populates="candidate", cascade="all, delete-orphan"
    )
    projects = relationship(
        "Project", back_populates="candidate", cascade="all, delete-orphan"
    )
    languages = relationship(
        "Language", back_populates="candidate", cascade="all, delete-orphan"
    )
    matches = relationship(
        "CandidateMatch", back_populates="candidate", cascade="all, delete-orphan"
    )


class CandidateDocument(Base):
    """Uploaded and parsed candidate CV document."""

    __tablename__ = "candidate_documents"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size_bytes = Column(Integer, nullable=False)
    page_count = Column(Integer, default=1)
    extracted_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="documents")


class Education(Base):
    """Candidate education history."""

    __tablename__ = "educations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False)
    institution = Column(String(255), nullable=False)
    degree = Column(String(255), nullable=False)
    field_of_study = Column(String(255), nullable=True)
    start_date = Column(String(50), nullable=True)
    end_date = Column(String(50), nullable=True)
    grade = Column(String(50), nullable=True)

    candidate = relationship("Candidate", back_populates="educations")


class Experience(Base):
    """Candidate work experience."""

    __tablename__ = "experiences"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False)
    company = Column(String(255), nullable=False)
    title = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    start_date = Column(String(50), nullable=True)
    end_date = Column(String(50), nullable=True)
    is_current = Column(Boolean, default=False)
    description = Column(Text, nullable=True)

    candidate = relationship("Candidate", back_populates="experiences")


class Skill(Base):
    """Candidate skill."""

    __tablename__ = "skills"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False)
    name = Column(String(100), nullable=False, index=True)
    category = Column(String(100), nullable=True)
    proficiency_level = Column(String(50), nullable=True)
    years_experience = Column(Float, nullable=True)

    candidate = relationship("Candidate", back_populates="skills")


class Certification(Base):
    """Candidate certification."""

    __tablename__ = "certifications"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False)
    name = Column(String(255), nullable=False)
    issuer = Column(String(255), nullable=True)
    issue_date = Column(String(50), nullable=True)
    expiration_date = Column(String(50), nullable=True)
    credential_id = Column(String(255), nullable=True)

    candidate = relationship("Candidate", back_populates="certifications")


class Project(Base):
    """Candidate projects."""

    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    technologies = Column(String(512), nullable=True)
    url = Column(String(512), nullable=True)

    candidate = relationship("Candidate", back_populates="projects")


class Language(Base):
    """Candidate languages."""

    __tablename__ = "languages"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False)
    language = Column(String(100), nullable=False)
    proficiency = Column(String(50), nullable=True)

    candidate = relationship("Candidate", back_populates="languages")


class Job(Base):
    """Job vacancy description."""

    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    title = Column(String(255), nullable=False)
    department = Column(String(255), nullable=True)
    location = Column(String(255), nullable=True)
    employment_type = Column(String(100), nullable=True)
    description = Column(Text, nullable=False)
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    requirements = relationship(
        "JobRequirement", back_populates="job", cascade="all, delete-orphan"
    )
    matches = relationship(
        "CandidateMatch", back_populates="job", cascade="all, delete-orphan"
    )


class JobRequirement(Base):
    """Specific job requirement item."""

    __tablename__ = "job_requirements"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    requirement_text = Column(Text, nullable=False)
    requirement_type = Column(String(50), default="required")
    importance = Column(String(50), default="high")
    min_years_experience = Column(Float, nullable=True)

    job = relationship("Job", back_populates="requirements")


class CandidateMatch(Base):
    """Match calculation between Candidate and Job."""

    __tablename__ = "candidate_matches"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    overall_score = Column(Float, nullable=False)
    semantic_score = Column(Float, nullable=True)
    skill_score = Column(Float, nullable=True)
    experience_score = Column(Float, nullable=True)
    recommendation = Column(String(100), nullable=True)
    match_status = Column(String(50), default="pending_review")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    candidate = relationship("Candidate", back_populates="matches")
    job = relationship("Job", back_populates="matches")
    feedback = relationship(
        "RecruiterFeedback", back_populates="match", cascade="all, delete-orphan"
    )


class Analysis(Base):
    """Deep candidate-job qualitative analysis."""

    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    candidate_id = Column(String(36), ForeignKey("candidates.id"), nullable=False)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=False)
    strengths = Column(Text, nullable=True)
    weaknesses = Column(Text, nullable=True)
    missing_skills = Column(Text, nullable=True)
    summary_report = Column(Text, nullable=True)
    agent_trace_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class RecruiterFeedback(Base):
    """Human-in-the-loop recruiter validation and feedback."""

    __tablename__ = "recruiter_feedbacks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    match_id = Column(String(36), ForeignKey("candidate_matches.id"), nullable=False)
    recruiter_id = Column(String(100), default="default_recruiter")
    decision = Column(String(50), nullable=False)  # Approved, Rejected, Modified
    rating = Column(Integer, nullable=True)
    comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    match = relationship("CandidateMatch", back_populates="feedback")
