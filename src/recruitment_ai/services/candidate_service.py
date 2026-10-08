"""Service coordinating candidate processing, parsing, matching, and repository updates."""

from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.database.models import (
    RecruiterFeedback,
)
from recruitment_ai.database.repositories import (
    CandidateRepository,
    FeedbackRepository,
    JobRepository,
    MatchRepository,
)
from recruitment_ai.document_processing.cleaner import DocumentCleaner
from recruitment_ai.extraction.schemas import CandidateMatch as MatchSchema
from recruitment_ai.extraction.schemas import CandidateProfile, JobDescription
from recruitment_ai.extraction.schemas import RecruiterFeedback as FeedbackSchema
from recruitment_ai.ingestion.parser import UnifiedDocumentParser
from recruitment_ai.matching.matcher import MatchingEngine


class CandidateService:
    """Core domain business service."""

    def __init__(self, db: Session | None = None) -> None:
        self.db = db
        self.parser = UnifiedDocumentParser()
        self.cleaner = DocumentCleaner()
        self.matcher = MatchingEngine()
        if db:
            self.candidate_repo = CandidateRepository(db)
            self.job_repo = JobRepository(db)
            self.match_repo = MatchRepository(db)
            self.feedback_repo = FeedbackRepository(db)

    def process_and_clean_document(self, file_path: Path) -> dict[str, Any]:
        """Parse raw document and extract cleaned text and metadata."""
        parsed = self.parser.parse(file_path)
        cleaned_text = self.cleaner.clean_text(parsed.raw_text)
        meta = self.cleaner.extract_metadata(cleaned_text)

        return {
            "filename": parsed.filename,
            "file_type": parsed.file_type,
            "total_pages": parsed.total_pages,
            "cleaned_text": cleaned_text,
            "metadata": meta.model_dump(),
        }

    def list_candidates(self) -> list[dict[str, Any]]:
        """List candidates from database or return placeholder when unattached."""
        if not self.db:
            return []
        candidates = self.candidate_repo.list()
        return [
            {
                "id": c.id,
                "full_name": c.full_name,
                "email": c.email,
                "created_at": c.created_at.isoformat(),
            }
            for c in candidates
        ]

    def list_jobs(self) -> list[dict[str, Any]]:
        """List jobs from database."""
        if not self.db:
            return []
        jobs = self.job_repo.list()
        return [
            {
                "id": j.id,
                "title": j.title,
                "department": j.department,
                "status": j.status,
            }
            for j in jobs
        ]

    def calculate_match(
        self, candidate: CandidateProfile, job: JobDescription
    ) -> MatchSchema:
        """Run candidate-job matching algorithm."""
        return self.matcher.match(candidate, job)

    def save_feedback(self, feedback: FeedbackSchema) -> dict[str, Any]:
        """Save human-in-the-loop recruiter feedback."""
        if self.db:
            db_feedback = RecruiterFeedback(
                match_id=feedback.match_id,
                recruiter_id=feedback.recruiter_id,
                decision=feedback.decision,
                rating=feedback.rating,
                comments=feedback.comments,
            )
            created = self.feedback_repo.create(db_feedback)
            return {"id": created.id, "status": "persisted"}
        return {"status": "saved_offline", "feedback": feedback.model_dump()}
