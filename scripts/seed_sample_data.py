"""Seed initial sample data across all 14 datasets, ontologies, synthetic, and knowledge bases."""

import json
from pathlib import Path

from recruitment_ai.data.synthetic_generator import SyntheticDataGenerator


def seed_knowledge_base(data_dir: Path) -> None:
    kb_dir = data_dir / "knowledge_base"

    # HR Policies
    hr_policies = kb_dir / "hr_policies"
    hr_policies.mkdir(parents=True, exist_ok=True)
    (hr_policies / "remote_work_policy.md").write_text(
        """# Remote Work and Flexible Hours Policy

## 1. Eligibility
All technical roles (Software Engineering, ML/AI, Data Science, DevOps) are eligible for remote or hybrid work arrangements.

## 2. Core Working Hours
Team members must be available for synchronous communication between 10:00 AM and 3:00 PM CET.

## 3. Equipment & Stipend
A home office setup stipend of €1,500 is provided alongside high-performance workstation hardware (Apple M3 Max or Dell Precision workstation).
""",
        encoding="utf-8",
    )

    (hr_policies / "compensation_and_levels.md").write_text(
        """# Engineering Career Ladder and Compensation Bands

| Level | Title | Min Experience | Target Base (€) |
|---|---|---|---|
| IC3 | Junior Engineer | 0–2 years | 45,000 – 60,000 |
| IC4 | Mid-Level Engineer | 2–5 years | 65,000 – 85,000 |
| IC5 | Senior Engineer | 5–8 years | 90,000 – 125,000 |
| IC6 | Staff Engineer / Tech Lead | 8+ years | 130,000 – 175,000 |

Stock options and performance bonuses (10-25%) apply to levels IC4 and above.
""",
        encoding="utf-8",
    )

    (hr_policies / "diversity_and_inclusion.md").write_text(
        """# Equal Opportunity & Fair Evaluation Standards

1. Blind initial resume screening where PII (name, gender, age, nationality) is redacted.
2. Skill-first qualification mapping evaluated using standardized ontologies (ESCO / O*NET).
3. Structured scoring rubrics applied consistently across all candidates.
""",
        encoding="utf-8",
    )

    # Job Profiles
    job_profiles = kb_dir / "job_profiles"
    job_profiles.mkdir(parents=True, exist_ok=True)
    (job_profiles / "senior_ml_engineer.md").write_text(
        """# Job Profile: Senior Machine Learning Engineer

## Overview
Designs, deploys, and maintains production-grade machine learning pipelines, LLM fine-tuning loops, and vector search systems.

## Key Requirements
- 5+ years experience in Python, PyTorch, and ML inference optimization.
- Proven experience fine-tuning LLMs (LoRA, QLoRA) and building RAG pipelines with vector databases.
- Familiarity with MLOps tooling (Docker, Kubernetes, MLflow, Prometheus).
""",
        encoding="utf-8",
    )

    (job_profiles / "nlp_engineer.md").write_text(
        """# Job Profile: NLP / Generative AI Engineer

## Overview
Develops specialized natural language processing systems, information extraction architectures, and automated agent workflows.

## Key Requirements
- Experience with Transformers, Hugging Face ecosystem, and LangChain/LangGraph.
- Deep expertise in named entity recognition (NER), text classification, and embedding representations (BGE-M3, E5).
""",
        encoding="utf-8",
    )

    # Skill Definitions
    skill_defs = kb_dir / "skill_definitions"
    skill_defs.mkdir(parents=True, exist_ok=True)
    (skill_defs / "core_ai_skills.md").write_text(
        """# AI / Machine Learning Competency Taxonomy

- **PyTorch**: Deep learning framework, autograd mechanics, distributed data parallel (DDP).
- **Retrieval-Augmented Generation (RAG)**: Dense retrieval, sparse BM25, cross-encoder reranking, chunking strategies.
- **LLM Fine-Tuning**: Supervised Fine-Tuning (SFT), LoRA parameter-efficient tuning, quantization (bitsandbytes).
- **Vector Databases**: Qdrant, Milvus, Chroma vector indexing (HNSW, Cosine similarity).
""",
        encoding="utf-8",
    )

    # Recruitment Rules
    rec_rules = kb_dir / "recruitment_rules"
    rec_rules.mkdir(parents=True, exist_ok=True)
    (rec_rules / "screening_rules.md").write_text(
        """# Screening and Fit Scoring Guidelines

1. **Must-Have Skill Coverage**: Must score at least 70% match on required skills to proceed past initial screening.
2. **Experience Tolerance**: A candidate within -1 year of requested experience with strong open-source or project demonstrations qualifies for interview.
3. **Hard Disqualifiers**: Missing core non-negotiable requirements explicitly marked as mandatory in the job description.
4. **Evaluation Threshold**: A match score >= 75 is categorized as a Recommended Shortlist. Scores 50-74 require human recruiter review.
""",
        encoding="utf-8",
    )


def seed_ontologies(data_dir: Path) -> None:
    esco_dir = data_dir / "raw" / "ontologies" / "esco"
    esco_dir.mkdir(parents=True, exist_ok=True)
    esco_samples = [
        {
            "concept_uri": "http://data.europa.eu/esco/skill/s1",
            "preferred_label": "Machine Learning",
            "alt_labels": ["ML", "Statistical Learning", "Deep Learning"],
            "skill_type": "knowledge",
            "source": "ESCO",
        },
        {
            "concept_uri": "http://data.europa.eu/esco/skill/s2",
            "preferred_label": "Python",
            "alt_labels": ["Python Programming", "Python 3"],
            "skill_type": "knowledge",
            "source": "ESCO",
        },
        {
            "concept_uri": "http://data.europa.eu/esco/skill/s3",
            "preferred_label": "Natural Language Processing",
            "alt_labels": ["NLP", "Text Processing", "Computational Linguistics"],
            "skill_type": "knowledge",
            "source": "ESCO",
        },
        {
            "concept_uri": "http://data.europa.eu/esco/skill/s4",
            "preferred_label": "PyTorch",
            "alt_labels": ["Torch", "PyTorch Framework"],
            "skill_type": "knowledge",
            "source": "ESCO",
        },
        {
            "concept_uri": "http://data.europa.eu/esco/skill/s5",
            "preferred_label": "Docker",
            "alt_labels": ["Containerization", "Docker Containers"],
            "skill_type": "skill/competence",
            "source": "ESCO",
        },
    ]
    (esco_dir / "skills_sample.json").write_text(
        json.dumps(esco_samples, indent=2), encoding="utf-8"
    )

    onet_dir = data_dir / "raw" / "ontologies" / "onet"
    onet_dir.mkdir(parents=True, exist_ok=True)
    onet_samples = [
        {
            "concept_uri": "15-1252.00",
            "preferred_label": "Software Developers",
            "alt_labels": [
                "Full Stack Developer",
                "Software Engineer",
                "Backend Developer",
            ],
            "skill_type": "occupation",
            "source": "ONET",
        },
        {
            "concept_uri": "15-2051.00",
            "preferred_label": "Data Scientists",
            "alt_labels": ["Machine Learning Engineer", "AI Scientist", "Data Modeler"],
            "skill_type": "occupation",
            "source": "ONET",
        },
    ]
    (onet_dir / "skills_sample.json").write_text(
        json.dumps(onet_samples, indent=2), encoding="utf-8"
    )


def seed_raw_samples(data_dir: Path) -> None:
    # 1. Resume parser sample
    rp_dir = data_dir / "raw" / "resumes" / "resume_parser"
    rp_dir.mkdir(parents=True, exist_ok=True)
    (rp_dir / "sample_parsed_cv.json").write_text(
        json.dumps(
            {
                "id": "raw-cv-001",
                "name": "Jane Doe",
                "email": "jane.doe@example.com",
                "title": "Senior AI Engineer",
                "summary": "AI Engineer with 6 years experience deploying production LLM systems.",
                "skills": ["Python", "PyTorch", "LangChain", "Qdrant", "FastAPI"],
                "experience": [
                    {
                        "company": "TechNova AI",
                        "title": "Lead ML Engineer",
                        "start_date": "2021",
                        "end_date": "Present",
                        "description": "Led team of 4 building retrieval-augmented generation systems.",
                    }
                ],
                "education": [
                    {
                        "institution": "Tech University",
                        "degree": "M.Sc. Computer Science",
                        "graduation_year": 2018,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 2. Resume NER sample
    ner_dir = data_dir / "raw" / "resumes" / "resume_ner"
    ner_dir.mkdir(parents=True, exist_ok=True)
    (ner_dir / "sample_ner_cv.json").write_text(
        json.dumps(
            {
                "text": "Jane Doe is a Senior AI Engineer at TechNova AI with expertise in PyTorch and FastAPI.",
                "entities": [
                    {"label": "PERSON", "text": "Jane Doe", "start": 0, "end": 8},
                    {
                        "label": "TITLE",
                        "text": "Senior AI Engineer",
                        "start": 14,
                        "end": 32,
                    },
                    {"label": "COMPANY", "text": "TechNova AI", "start": 36, "end": 47},
                    {"label": "SKILL", "text": "PyTorch", "start": 66, "end": 73},
                    {"label": "SKILL", "text": "FastAPI", "start": 78, "end": 85},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 3. Resume skill extractor sample
    rse_dir = data_dir / "raw" / "job_descriptions" / "resume_skill_extractor"
    rse_dir.mkdir(parents=True, exist_ok=True)
    (rse_dir / "sample_skill_extractor.json").write_text(
        json.dumps(
            {
                "job_title": "Machine Learning Engineer",
                "skills_extracted": [
                    "Python",
                    "PyTorch",
                    "Docker",
                    "SQL",
                    "Transformers",
                ],
                "confidence_scores": {"Python": 0.99, "PyTorch": 0.95, "Docker": 0.88},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 4. Tech jobs 2026 sample
    tj_dir = data_dir / "raw" / "job_descriptions" / "tech_jobs_2026"
    tj_dir.mkdir(parents=True, exist_ok=True)
    (tj_dir / "sample_tech_job_2026.json").write_text(
        json.dumps(
            {
                "job_id": "tech-job-2026-001",
                "title": "Agentic AI Systems Engineer",
                "year": 2026,
                "remote": True,
                "tech_stack": ["LangGraph", "Qdrant", "Python", "Qwen2.5", "FastAPI"],
                "responsibilities": "Develop autonomous multi-agent systems and enterprise RAG solutions.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 7. Job matching sample
    jm_dir = data_dir / "raw" / "matching" / "job_matching"
    jm_dir.mkdir(parents=True, exist_ok=True)
    (jm_dir / "sample_matching_pair.json").write_text(
        json.dumps(
            {
                "pair_id": "jm-sample-001",
                "resume_id": "raw-cv-001",
                "job_id": "tech-job-2026-001",
                "label": 1,
                "score": 0.89,
                "overlap_skills": ["Python", "FastAPI", "Qdrant"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 8. DocLayNet sample
    dln_dir = data_dir / "raw" / "document_layout" / "doclaynet"
    dln_dir.mkdir(parents=True, exist_ok=True)
    (dln_dir / "sample_doclaynet_layout.json").write_text(
        json.dumps(
            {
                "page_hash": "doclaynet-sample-p1",
                "layout_elements": [
                    {
                        "category": "header",
                        "box": [50, 50, 500, 100],
                        "confidence": 0.98,
                    },
                    {
                        "category": "section_heading",
                        "box": [50, 110, 300, 140],
                        "confidence": 0.95,
                    },
                    {
                        "category": "text_block",
                        "box": [50, 150, 550, 300],
                        "confidence": 0.92,
                    },
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def seed_evaluation_data(data_dir: Path) -> None:
    eval_dir = data_dir / "evaluation"

    # Benchmark
    bench_dir = eval_dir / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "matching_benchmark_v1.json").write_text(
        json.dumps(
            {
                "benchmark_name": "ai_recruitment_v1_benchmark",
                "version": "1.0",
                "total_pairs": 50,
                "metric_targets": {"ndcg@5": 0.85, "precision@3": 0.80, "map": 0.78},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Human labels
    human_dir = eval_dir / "human_labels"
    human_dir.mkdir(parents=True, exist_ok=True)
    (human_dir / "annotator_evaluations.json").write_text(
        json.dumps(
            [
                {
                    "evaluation_id": "eval-001",
                    "candidate_id": "syn-cv-001",
                    "job_id": "syn-job-001",
                    "annotator_id": "expert_recruiter_1",
                    "human_score": 92.0,
                    "model_score": 90.0,
                    "agreed": True,
                    "notes": "Excellent alignment on NLP and Python core skills.",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    # Agent traces
    trace_dir = eval_dir / "agent_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "sample_trace.json").write_text(
        json.dumps(
            {
                "trace_id": "trace-test-001",
                "workflow": "recruitment_analysis_graph",
                "agent_invocations": [
                    {
                        "agent": "parser_agent",
                        "status": "completed",
                        "duration_ms": 320,
                    },
                    {"agent": "rag_agent", "status": "completed", "duration_ms": 140},
                    {
                        "agent": "matching_agent",
                        "status": "completed",
                        "duration_ms": 210,
                    },
                    {
                        "agent": "evaluator_agent",
                        "status": "completed",
                        "duration_ms": 180,
                    },
                ],
                "total_duration_ms": 850,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    data_dir = Path("./data")
    print("1. Seeding Knowledge Base...")
    seed_knowledge_base(data_dir)

    print("2. Seeding Ontologies (ESCO & O*NET)...")
    seed_ontologies(data_dir)

    print("3. Seeding Raw Dataset Samples...")
    seed_raw_samples(data_dir)

    print("4. Generating Synthetic CVs, JDs, and Matching Pairs...")
    generator = SyntheticDataGenerator(output_root=data_dir / "synthetic")
    counts = generator.generate_all()
    print(f"   Generated: {counts}")

    print("5. Seeding Evaluation and Benchmark Data...")
    seed_evaluation_data(data_dir)

    print("Sample data seeding successfully completed across all 14 datasets!")


if __name__ == "__main__":
    main()
