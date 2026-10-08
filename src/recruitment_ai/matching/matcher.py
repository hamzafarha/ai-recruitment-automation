"""Deterministic and semantic multi-factor matching engine."""

from recruitment_ai.extraction.schemas import (
    CandidateMatch,
    CandidateProfile,
    CandidateRecommendation,
    JobDescription,
    MatchEvidence,
)


class MatchingEngine:
    """Computes multidimensional candidate-to-job match."""

    def compute_skill_overlap(
        self, candidate_skills: list[str], required_skills: list[str]
    ) -> float:
        """Calculate Jaccard-like normalized overlap between skill sets."""
        if not required_skills:
            return 100.0
        c_set = {s.lower().strip() for s in candidate_skills}
        r_set = {s.lower().strip() for s in required_skills}
        intersection = c_set.intersection(r_set)
        return (len(intersection) / len(r_set)) * 100.0

    def match(
        self,
        candidate: CandidateProfile,
        job: JobDescription,
        semantic_score: float = 80.0,
    ) -> CandidateMatch:
        """Compute combined score from skill matching, semantic similarity, and experience."""
        c_skills = [s.name for s in candidate.skills]
        j_skills = [
            req.requirement_text
            for req in job.requirements
            if req.requirement_type == "required"
        ]

        skill_score = self.compute_skill_overlap(c_skills, j_skills)
        exp_score = min(100.0, len(candidate.experience) * 25.0)

        # Weighted score: 40% semantic, 40% skills, 20% experience
        overall_score = round(
            (semantic_score * 0.4) + (skill_score * 0.4) + (exp_score * 0.2), 2
        )

        decision = "no_fit"
        if overall_score >= 80:
            decision = "strong_fit"
        elif overall_score >= 65:
            decision = "fit"
        elif overall_score >= 50:
            decision = "potential_fit"

        return CandidateMatch(
            candidate_id=candidate.id or "temp-candidate",
            job_id=job.id or "temp-job",
            overall_score=overall_score,
            semantic_score=semantic_score,
            skill_score=round(skill_score, 2),
            experience_score=round(exp_score, 2),
            evidence=[
                MatchEvidence(
                    criterion="Skill Overlap",
                    source_chunk=f"Candidate has {len(c_skills)} skills matched against {len(j_skills)} requirements.",
                    confidence=0.85,
                )
            ],
            recommendation=CandidateRecommendation(
                decision=decision,  # type: ignore[arg-type]
                recommendation_summary=f"Automated evaluation score: {overall_score}%",
                suggested_interview_topics=["Technical depth", "Domain projects"],
            ),
        )
