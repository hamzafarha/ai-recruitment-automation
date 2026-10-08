"""Structured Pydantic schemas for recruitment automation data models."""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class SchemaBase(BaseModel):
    """Base schema with standard serialization configurations."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        extra="ignore",
    )


class Education(SchemaBase):
    """Candidate education entry."""

    institution: str = Field(..., description="University or school name")
    degree: str = Field(..., description="Degree or diploma earned")
    field_of_study: str | None = Field(
        default=None, description="Major or area of study"
    )
    start_date: str | None = Field(
        default=None, description="Start date (e.g. YYYY-MM or YYYY)"
    )
    end_date: str | None = Field(default=None, description="End or graduation date")
    grade: str | None = Field(default=None, description="GPA or honors distinction")


class Experience(SchemaBase):
    """Candidate professional work experience."""

    company: str = Field(..., description="Organization or company name")
    title: str = Field(..., description="Job role / position title")
    location: str | None = Field(default=None, description="Job location or Remote")
    start_date: str | None = Field(default=None, description="Start date")
    end_date: str | None = Field(default=None, description="End date or 'Present'")
    is_current: bool = Field(
        default=False, description="Whether this is the current job"
    )
    description: str | None = Field(
        default=None, description="Summary of responsibilities and achievements"
    )


class Skill(SchemaBase):
    """Candidate skill details."""

    name: str = Field(..., description="Skill or technology name")
    category: str | None = Field(
        default=None, description="e.g. Technical, Soft, Language, Domain"
    )
    proficiency_level: (
        Literal["beginner", "intermediate", "advanced", "expert"] | None
    ) = Field(default=None, description="Self-assessed or inferred proficiency")
    years_experience: float | None = Field(
        default=None, ge=0, description="Estimated years of practical experience"
    )


class Certification(SchemaBase):
    """Candidate license or certification."""

    name: str = Field(..., description="Certificate name")
    issuer: str | None = Field(default=None, description="Issuing organization")
    issue_date: str | None = Field(default=None, description="Issue date")
    expiration_date: str | None = Field(default=None, description="Expiration date")
    credential_id: str | None = Field(
        default=None, description="Credential ID or verification link"
    )


class Project(SchemaBase):
    """Candidate project highlight."""

    title: str = Field(..., description="Project name")
    description: str | None = Field(
        default=None, description="Overview and key deliverables"
    )
    technologies: list[str] = Field(
        default_factory=list, description="Stack or tools used"
    )
    url: str | None = Field(
        default=None, description="Public code repository or demo link"
    )


class Language(SchemaBase):
    """Language spoken/written."""

    language: str = Field(..., description="Language name")
    proficiency: str | None = Field(
        default=None, description="Proficiency level (e.g., Native, Fluent, B2, C1)"
    )


class CandidateProfile(SchemaBase):
    """Complete structured candidate profile extracted from CV."""

    schema_version: str = Field(
        default="1.0.0", description="Schema version identifier"
    )
    id: str | None = Field(default=None, description="Unique candidate ID")
    full_name: str = Field(..., description="Candidate full name")
    email: EmailStr | str | None = Field(
        default=None, description="Primary email address"
    )
    phone: str | None = Field(default=None, description="Primary contact phone number")
    location: str | None = Field(
        default=None, description="Current location or residence"
    )
    summary: str | None = Field(default=None, description="Professional summary or bio")
    education: list[Education] = Field(
        default_factory=list, description="Education history"
    )
    experience: list[Experience] = Field(
        default_factory=list, description="Work experience"
    )
    skills: list[Skill] = Field(default_factory=list, description="Skills catalog")
    certifications: list[Certification] = Field(
        default_factory=list, description="Certifications"
    )
    projects: list[Project] = Field(default_factory=list, description="Projects")
    languages: list[Language] = Field(default_factory=list, description="Languages")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class JobRequirement(SchemaBase):
    """Specific job requirement."""

    requirement_text: str = Field(..., description="Text of the requirement")
    requirement_type: Literal["required", "preferred", "optional"] = Field(
        default="required", description="Requirement level"
    )
    importance: Literal["high", "medium", "low"] = Field(
        default="high", description="Weight/importance level"
    )
    min_years_experience: float | None = Field(default=None, ge=0)


class JobDescription(SchemaBase):
    """Structured job vacancy description."""

    schema_version: str = Field(default="1.0.0")
    id: str | None = Field(default=None)
    title: str = Field(..., description="Job role title")
    department: str | None = Field(default=None)
    location: str | None = Field(default=None)
    employment_type: str | None = Field(default=None)
    description: str = Field(..., description="Full description text")
    requirements: list[JobRequirement] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MatchEvidence(SchemaBase):
    """Evidence snippet retrieved to justify candidate match score."""

    criterion: str = Field(..., description="Requirement or skill evaluated")
    source_chunk: str = Field(
        ..., description="Document text excerpt supporting the match"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    page_or_section: str | None = Field(default=None)


class CandidateRecommendation(SchemaBase):
    """Actionable recommendation for the recruiter."""

    decision: Literal["strong_fit", "fit", "potential_fit", "no_fit"] = Field(...)
    recommendation_summary: str = Field(...)
    suggested_interview_topics: list[str] = Field(default_factory=list)


class CandidateMatch(SchemaBase):
    """Quantitative candidate-to-job match assessment."""

    candidate_id: str
    job_id: str
    overall_score: Annotated[float, Field(ge=0.0, le=100.0)]
    semantic_score: float | None = Field(default=None, ge=0.0, le=100.0)
    skill_score: float | None = Field(default=None, ge=0.0, le=100.0)
    experience_score: float | None = Field(default=None, ge=0.0, le=100.0)
    evidence: list[MatchEvidence] = Field(default_factory=list)
    recommendation: CandidateRecommendation | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RecruiterFeedback(SchemaBase):
    """Human-in-the-loop recruiter decision and feedback."""

    match_id: str
    recruiter_id: str = "default_recruiter"
    decision: Literal["Approved", "Rejected", "Modified"] = Field(
        ..., description="Recruiter review decision"
    )
    rating: int | None = Field(default=None, ge=1, le=5)
    comments: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AnalysisReport(SchemaBase):
    """Deep candidate qualitative analysis report."""

    candidate_id: str
    job_id: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    summary_report: str
    match_score: float = Field(ge=0.0, le=100.0)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
