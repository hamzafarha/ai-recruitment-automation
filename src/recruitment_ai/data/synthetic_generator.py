"""Synthetic data generation engine producing controlled CVs, JDs, and hard-negative matching pairs."""

from pathlib import Path

from pydantic import BaseModel, Field

from recruitment_ai.extraction.schemas import (
    CandidateProfile,
    Education,
    Experience,
    JobDescription,
    JobRequirement,
    Skill,
)


class SyntheticMatchingPair(BaseModel):
    """Ground truth matching pair for ranking evaluation and hard negative training."""

    pair_id: str
    candidate_id: str
    job_id: str
    ground_truth_score: float = Field(..., ge=0.0, le=100.0)
    match_category: str  # positive, partial_match, hard_negative
    rationale: str
    candidate: CandidateProfile
    job: JobDescription


class SyntheticDataGenerator:
    """Generates synthetic CVs, job descriptions, and hard-negative evaluation pairs."""

    def __init__(self, output_root: Path | None = None) -> None:
        self.output_root = output_root or Path("./data/synthetic")
        self.cvs_dir = self.output_root / "cvs"
        self.jobs_dir = self.output_root / "job_descriptions"
        self.pairs_dir = self.output_root / "matching_pairs"

    def generate_all(self) -> dict[str, int]:
        """Generate complete synthetic dataset suite and persist to disk."""
        self.cvs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.pairs_dir.mkdir(parents=True, exist_ok=True)

        cvs = self._generate_synthetic_cvs()
        jobs = self._generate_synthetic_jobs()
        pairs = self._generate_synthetic_pairs(cvs, jobs)

        for cv in cvs:
            file_path = self.cvs_dir / f"{cv.id}.json"
            file_path.write_text(cv.model_dump_json(indent=2), encoding="utf-8")

        for job in jobs:
            file_path = self.jobs_dir / f"{job.id}.json"
            file_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")

        for pair in pairs:
            file_path = self.pairs_dir / f"{pair.pair_id}.json"
            file_path.write_text(pair.model_dump_json(indent=2), encoding="utf-8")

        return {
            "cvs_generated": len(cvs),
            "jobs_generated": len(jobs),
            "matching_pairs_generated": len(pairs),
        }

    def _generate_synthetic_cvs(self) -> list[CandidateProfile]:
        """Generate CV variations including standard, junior, career-pivot, and edge cases."""
        return [
            # Standard Strong Candidate
            CandidateProfile(
                id="syn_cv_01_strong_ml",
                full_name="Elena Rostova (Synthetic)",
                email="elena.synthetic@example.com",
                summary="Staff ML Engineer with 7+ years building enterprise RAG, Qdrant search, and LLM fine-tuning pipelines.",
                skills=[
                    Skill(
                        name="Python", proficiency_level="expert", years_experience=7.0
                    ),
                    Skill(
                        name="PyTorch", proficiency_level="expert", years_experience=6.0
                    ),
                    Skill(
                        name="Qdrant",
                        proficiency_level="advanced",
                        years_experience=3.0,
                    ),
                    Skill(
                        name="Transformers",
                        proficiency_level="expert",
                        years_experience=5.0,
                    ),
                    Skill(
                        name="FastAPI",
                        proficiency_level="advanced",
                        years_experience=4.0,
                    ),
                    Skill(
                        name="Docker",
                        proficiency_level="advanced",
                        years_experience=5.0,
                    ),
                ],
                experience=[
                    Experience(
                        company="NeuralScale AI",
                        title="Senior AI Architect",
                        start_date="2021",
                        end_date="Present",
                        description="Lead LLM infrastructure and Qdrant semantic search serving 5M daily queries.",
                    ),
                    Experience(
                        company="Cognitive Corp",
                        title="Machine Learning Engineer",
                        start_date="2018",
                        end_date="2021",
                        description="Implemented PyTorch models and computer vision pipelines.",
                    ),
                ],
                education=[
                    Education(
                        institution="Tech Institute",
                        degree="M.S. Artificial Intelligence",
                        grade="3.9",
                    )
                ],
            ),
            # Hard Negative Candidate: Right keywords (Python, AI) but junior / wrong seniority
            CandidateProfile(
                id="syn_cv_02_junior_mismatch",
                full_name="Lucas Martin (Synthetic)",
                email="lucas.synthetic@example.com",
                summary="Recent computer science graduate passionate about basic AI scripting.",
                skills=[
                    Skill(
                        name="Python",
                        proficiency_level="beginner",
                        years_experience=1.0,
                    ),
                    Skill(
                        name="PyTorch",
                        proficiency_level="beginner",
                        years_experience=0.5,
                    ),
                    Skill(
                        name="HTML",
                        proficiency_level="intermediate",
                        years_experience=2.0,
                    ),
                ],
                experience=[
                    Experience(
                        company="University Lab",
                        title="Research Assistant",
                        start_date="2023",
                        end_date="2024",
                        description="Assisted in running basic benchmark evaluations.",
                    )
                ],
                education=[
                    Education(
                        institution="City College", degree="B.S. Computer Science"
                    )
                ],
            ),
            # Edge Case: Career Pivot with strong engineering but self-taught AI
            CandidateProfile(
                id="syn_cv_03_career_pivot",
                full_name="Amara Okafor (Synthetic)",
                email="amara.synthetic@example.com",
                summary="Principal Backend Engineer (10+ years distributed systems) transitioning into Generative AI engineering.",
                skills=[
                    Skill(
                        name="Python", proficiency_level="expert", years_experience=10.0
                    ),
                    Skill(
                        name="PostgreSQL",
                        proficiency_level="expert",
                        years_experience=9.0,
                    ),
                    Skill(
                        name="Docker", proficiency_level="expert", years_experience=8.0
                    ),
                    Skill(
                        name="LangGraph",
                        proficiency_level="intermediate",
                        years_experience=1.0,
                    ),
                    Skill(
                        name="Qdrant",
                        proficiency_level="intermediate",
                        years_experience=1.0,
                    ),
                ],
                experience=[
                    Experience(
                        company="CloudData Inc",
                        title="Principal Backend Engineer",
                        start_date="2016",
                        end_date="Present",
                        description="Architected distributed database clusters handling 100k writes/sec.",
                    )
                ],
                education=[
                    Education(
                        institution="State University",
                        degree="B.S. Electrical Engineering",
                    )
                ],
            ),
            # Edge Case: Missing formal education, deep practical open-source experience
            CandidateProfile(
                id="syn_cv_04_no_formal_degree",
                full_name="Kai Chen (Synthetic)",
                email="kai.synthetic@example.com",
                summary="Core open-source contributor to deep learning and embedding frameworks with 5 years self-directed engineering.",
                skills=[
                    Skill(
                        name="PyTorch", proficiency_level="expert", years_experience=5.0
                    ),
                    Skill(
                        name="Transformers",
                        proficiency_level="expert",
                        years_experience=4.0,
                    ),
                    Skill(
                        name="CUDA", proficiency_level="advanced", years_experience=3.0
                    ),
                ],
                experience=[
                    Experience(
                        company="OpenSource Community",
                        title="Core Maintainer",
                        start_date="2019",
                        end_date="Present",
                        description="Maintained model optimization repositories with 10k GitHub stars.",
                    )
                ],
                education=[],  # Edge case: No formal degree
            ),
        ]

    def _generate_synthetic_jobs(self) -> list[JobDescription]:
        """Generate job vacancies with strict requirements and edge case constraints."""
        return [
            # Job 1: Senior AI / RAG Engineer
            JobDescription(
                id="syn_job_01_lead_rag",
                title="Lead AI / RAG Systems Engineer",
                department="AI Research & Engineering",
                description="We are seeking an experienced ML engineer to architect enterprise RAG pipelines, vector search with Qdrant, and multi-agent workflows.",
                requirements=[
                    JobRequirement(
                        requirement_text="5+ years experience in Python and PyTorch",
                        requirement_type="required",
                        importance="high",
                        min_years_experience=5.0,
                    ),
                    JobRequirement(
                        requirement_text="Hands-on experience deploying Qdrant or vector databases",
                        requirement_type="required",
                        importance="high",
                    ),
                    JobRequirement(
                        requirement_text="Demonstrated production experience with LLMs and Transformers",
                        requirement_type="required",
                        importance="high",
                    ),
                ],
            ),
            # Job 2: Senior Backend Distributed Systems
            JobDescription(
                id="syn_job_02_backend_architect",
                title="Senior Distributed Systems Architect",
                department="Infrastructure",
                description="Lead our high-throughput transactional database infrastructure and microservices.",
                requirements=[
                    JobRequirement(
                        requirement_text="7+ years distributed backend development in Python or Go",
                        requirement_type="required",
                        importance="high",
                        min_years_experience=7.0,
                    ),
                    JobRequirement(
                        requirement_text="Expertise in PostgreSQL performance tuning and Docker",
                        requirement_type="required",
                        importance="high",
                    ),
                ],
            ),
        ]

    def _generate_synthetic_pairs(
        self, cvs: list[CandidateProfile], jobs: list[JobDescription]
    ) -> list[SyntheticMatchingPair]:
        """Formulate positive, partial, and hard-negative matching pairs."""
        cv_map = {c.id: c for c in cvs}
        job_map = {j.id: j for j in jobs}

        return [
            # Strong Positive Match
            SyntheticMatchingPair(
                pair_id="pair_01_strong_match",
                candidate_id="syn_cv_01_strong_ml",
                job_id="syn_job_01_lead_rag",
                ground_truth_score=94.0,
                match_category="positive",
                rationale="Candidate meets and exceeds all 5+ years PyTorch, Qdrant, and Transformers requirements.",
                candidate=cv_map["syn_cv_01_strong_ml"],
                job=job_map["syn_job_01_lead_rag"],
            ),
            # Hard Negative Match: Junior candidate against Senior Lead role (keyword overlap but seniority mismatch)
            SyntheticMatchingPair(
                pair_id="pair_02_hard_negative",
                candidate_id="syn_cv_02_junior_mismatch",
                job_id="syn_job_01_lead_rag",
                ground_truth_score=32.0,
                match_category="hard_negative",
                rationale="Hard Negative: Candidate lists Python and PyTorch keywords but lacks the mandatory 5+ years experience and production architecture skills.",
                candidate=cv_map["syn_cv_02_junior_mismatch"],
                job=job_map["syn_job_01_lead_rag"],
            ),
            # Partial Match: Strong distributed backend engineer transitioning into AI
            SyntheticMatchingPair(
                pair_id="pair_03_partial_match",
                candidate_id="syn_cv_03_career_pivot",
                job_id="syn_job_01_lead_rag",
                ground_truth_score=68.0,
                match_category="partial_match",
                rationale="Partial Match: Exceptional backend distributed systems background with emerging AI skills, but less than 5 years dedicated PyTorch/RAG depth.",
                candidate=cv_map["syn_cv_03_career_pivot"],
                job=job_map["syn_job_01_lead_rag"],
            ),
        ]
