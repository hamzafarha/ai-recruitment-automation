"""Unit tests for evaluation-set construction (Step 1c), on small hand-made files."""

import csv
import json
import re
from pathlib import Path

import pytest

from recruitment_ai.data.tech_jobs import ROLE_PATTERNS
from recruitment_ai.evaluation.eval_set import build_cv_pool, shortlist_jobs
from recruitment_ai.extraction.schemas import JobDescription

EVAL_JOBS = Path(__file__).resolve().parents[2] / "data" / "processed" / "eval_v1" / "jobs"

INDEX_COLUMNS = [
    "split", "resume_id", "category", "is_tech_category", "tech_title_match", "leak_train",
    "corrupted", "template_identity", "grounded", "text_len", "job_titles",
]


def _index_row(split, rid, *, tech=True, title=True, leak=False, corrupted=False):
    return {
        "split": split, "resume_id": rid, "category": "Data Science" if tech else "Accountant",
        "is_tech_category": str(tech), "tech_title_match": str(title), "leak_train": str(leak),
        "corrupted": str(corrupted), "template_identity": "False", "grounded": "0.95",
        "text_len": "100", "job_titles": "Data Scientist",
    }


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        _index_row("validation", "v1"),
        _index_row("validation", "v2", title=False),       # excluded: not a tech job title
        _index_row("validation", "v3", leak=True),         # excluded: in train
        _index_row("test", "t1"),
        _index_row("test", "t2", corrupted=True),          # excluded: corrupted
        _index_row("test", "t3"),                           # excluded: same text as v1
        _index_row("test", "t4", tech=False),               # not a tech category
        _index_row("train", "x1"),                          # train is never in the pool
    ]
    index = tmp_path / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    texts = {"v1": "Same CV text", "v2": "Other", "v3": "Leaked", "t1": "Unique", "t2": "Bad",
             "t3": "same cv TEXT!", "t4": "Accounting"}
    curated = tmp_path / "curated"
    curated.mkdir()
    for split, ids in (("validation", ["v1", "v2", "v3"]), ("test", ["t1", "t2", "t3", "t4"])):
        with (curated / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for rid in ids:
                f.write(json.dumps({"resume_id": rid, "resume_text": texts[rid], "target_json": {}}) + "\n")
    return index, curated


def test_build_cv_pool_filters_and_splits(files):
    pool, stats = build_cv_pool(*files)
    assert [(r["resume_id"], r["eval_split"]) for r in pool] == [("v1", "dev"), ("t1", "test")]
    assert stats == {
        "tech_category": 6, "leak_train": 1, "corrupted": 1,
        "not_tech_title": 1, "duplicate_text": 1, "pool": 2,
    }


def test_build_cv_pool_is_deterministic(files):
    """Rule-based selection: running it twice gives exactly the same pool."""
    assert build_cv_pool(*files) == build_cv_pool(*files)


def test_shortlist_prefers_full_time_detailed_postings():
    jobs = [
        {"id": 1, "role": "ml_engineer", "title": "ML Engineer Intern", "employment_type": "Internship", "desc_len": 4500},
        {"id": 2, "role": "ml_engineer", "title": "ML Engineer", "employment_type": "Full-time", "desc_len": 9000},
        {"id": 3, "role": "ml_engineer", "title": "Senior ML Engineer", "employment_type": "Full-time", "desc_len": 4400},
        {"id": 4, "role": None, "title": "Designer", "employment_type": "Full-time", "desc_len": 4500},
        {"id": 5, "role": "devops_cloud", "title": "DevOps Intern", "employment_type": "Internship", "desc_len": 3000},
    ]
    shortlist = shortlist_jobs(jobs, per_role=2)
    assert [j["id"] for j in shortlist["ml_engineer"]] == [3, 2]
    assert [j["id"] for j in shortlist["devops_cloud"]] == [5]  # internships allowed when nothing else exists
    assert set(shortlist) == {"ml_engineer", "devops_cloud"}


# --------------------------------------------------------------------------- frozen job descriptions


def test_one_eval_job_per_target_role():
    roles = sorted(p.stem for p in EVAL_JOBS.glob("*.json"))
    assert roles == sorted(ROLE_PATTERNS)


@pytest.mark.parametrize("path", sorted(EVAL_JOBS.glob("*.json")), ids=lambda p: p.stem)
def test_eval_job_is_valid_and_well_formed(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    job = JobDescription.model_validate(raw)
    ids = [r.id for r in job.requirements]

    assert raw["role"] == path.stem and raw["source"]["job_id"]
    assert len(job.description) > 1000  # the full original posting is kept (RAG needs it)
    assert len(ids) == len(set(ids)), "requirement ids must be unique"
    # M1, M2... are must-haves ("required"); N1, N2... are nice-to-haves
    for r in job.requirements:
        assert re.fullmatch(r"[MN]\d+", r.id), r.id
        assert (r.id[0] == "M") == (r.requirement_type == "required"), r.id
    assert sum(i.startswith("M") for i in ids) >= 3
