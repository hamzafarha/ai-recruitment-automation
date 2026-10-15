"""Unit tests for Pydantic data schemas."""

import pytest
from pydantic import ValidationError

from recruitment_ai.extraction.schemas import (
    CandidateMatch,
    CandidateProfile,
    CandidateRecommendation,
    Education,
    Experience,
    JobDescription,
    JobRequirement,
    Language,
    MatchEvidence,
    RecruiterFeedback,
    Skill,
)


def test_candidate_profile_schema_valid():
    """Verify CandidateProfile schema instantiation and defaults."""
    profile = CandidateProfile(
        full_name="Jane Doe",
        email="jane@example.com",
        skills=[Skill(name="Python", proficiency_level="expert", years_experience=5.0)],
        education=[Education(institution="MIT", degree="B.S. CS")],
        experience=[Experience(company="Acme Corp", title="Software Engineer")],
        languages=[Language(language="English", proficiency="Native")],
    )
    assert profile.full_name == "Jane Doe"
    assert len(profile.skills) == 1
    assert profile.skills[0].name == "Python"
    assert profile.schema_version == "1.0.0"


def test_candidate_profile_allows_missing_name():
    """Anonymised CVs have no name: the profile must accept None, not force a value."""
    profile = CandidateProfile(experience=[Experience(title="Developer")])
    assert profile.full_name is None
    assert profile.experience[0].company is None


def test_candidate_profile_schema_validation_error():
    """Verify validation still rejects invalid values."""
    with pytest.raises(ValidationError):
        CandidateProfile(skills=[Skill(name="Python", proficiency_level="guru")])


def test_job_description_schema():
    """Verify JobDescription and JobRequirement schemas."""
    job = JobDescription(
        title="ML Engineer",
        description="Build state-of-the-art NLP models",
        requirements=[
            JobRequirement(
                requirement_text="5+ years Python and PyTorch",
                requirement_type="required",
                importance="high",
            )
        ],
    )
    assert job.title == "ML Engineer"
    assert len(job.requirements) == 1


def test_candidate_match_schema():
    """Verify CandidateMatch schema and nested evidence."""
    match = CandidateMatch(
        candidate_id="cand-001",
        job_id="job-001",
        overall_score=88.5,
        evidence=[
            MatchEvidence(
                criterion="Python",
                source_chunk="5 years Python experience",
                confidence=0.95,
            )
        ],
        recommendation=CandidateRecommendation(
            decision="strong_fit",
            recommendation_summary="High technical match",
        ),
    )
    assert match.overall_score == 88.5
    assert match.recommendation.decision == "strong_fit"


def test_recruiter_feedback_schema():
    """Verify RecruiterFeedback validation."""
    fb = RecruiterFeedback(
        match_id="match-123",
        decision="Approved",
        rating=5,
        comments="Excellent candidate fit.",
    )
    assert fb.decision == "Approved"
    assert fb.rating == 5
