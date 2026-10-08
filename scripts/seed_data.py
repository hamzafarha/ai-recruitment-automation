"""Seed synthetic test data (Strictly anonymous / synthetic data only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def seed_synthetic_fixtures() -> None:
    """Generate synthetic candidate and job fixtures for testing."""
    fixtures_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    synthetic_cv_text = """
JOHN DOE (SYNTHETIC CANDIDATE)
Email: john.doe.synthetic@example.com
Phone: +1 555-019-2834
Location: Remote / New York, NY

SUMMARY
Results-oriented Senior Machine Learning Engineer with 6+ years of experience designing and deploying
scalable ML models, RAG architectures, and NLP pipelines in production.

EXPERIENCE
Senior AI Engineer | DataCore Systems | 2021 - Present
- Architected vector search and RAG retrieval systems using Qdrant and Sentence Transformers.
- Built FastAPI microservices processing 100k+ candidate documents daily.
- Optimized fine-tuned transformer inference latency by 45%.

Machine Learning Engineer | Apex AI | 2018 - 2021
- Developed document extraction models with PyTorch and Hugging Face Transformers.
- Implemented OCR pipeline for scanned financial reports.

EDUCATION
Master of Science in Computer Science | Columbia University | 2016 - 2018
Bachelor of Science in Software Engineering | State University | 2012 - 2016

SKILLS
Python, PyTorch, Hugging Face, Sentence Transformers, Qdrant, Docker, FastAPI, PostgreSQL, LangChain
    """

    synthetic_job_text = """
JOB DESCRIPTION: Senior AI / Machine Learning Engineer
Department: AI Research & Engineering
Location: Remote

Overview:
We are seeking an experienced ML Engineer to lead development of our agentic AI recruitment system.

Key Responsibilities:
- Build high-performance document parsing and OCR pipelines.
- Implement vector search with Qdrant and modern embedding models.
- Deploy scalable FastAPI endpoints with LangGraph orchestration.

Requirements:
- Strong proficiency in Python, PyTorch, and Transformers.
- Experience with RAG pipelines, vector databases (Qdrant), and LangChain/LangGraph.
- Proven track record deploying microservices via Docker.
    """

    (fixtures_dir / "sample_cv.txt").write_text(
        synthetic_cv_text.strip(), encoding="utf-8"
    )
    (fixtures_dir / "sample_job.txt").write_text(
        synthetic_job_text.strip(), encoding="utf-8"
    )
    print(f"Synthetic test fixtures successfully generated in {fixtures_dir}")


if __name__ == "__main__":
    seed_synthetic_fixtures()
