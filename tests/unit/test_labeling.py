"""Tests for labeling-set selection and label storage (no Streamlit needed)."""

import json
from pathlib import Path

from recruitment_ai.evaluation.eval_set import (
    PAIR_PLAN,
    read_rows,
    sample_extraction_checks,
    select_pairs,
    synthetic_split,
)
from recruitment_ai.evaluation.labeling import (
    MATCH_COLUMNS,
    match_row,
    upsert,
    validate_match,
)


def _pool() -> list[dict[str, str]]:
    """A fake pool with 4 dev + 6 test CVs in every category used by PAIR_PLAN."""
    categories = {c for plan in PAIR_PLAN.values() for c, _ in plan["real"]}
    return [
        {"resume_id": f"{cat[:3]}_{split}_{i}", "category": cat, "eval_split": split}
        for cat in sorted(categories)
        for split, n in (("dev", 4), ("test", 6))
        for i in range(n)
    ]


def test_select_pairs_follows_the_plan():
    pairs = select_pairs(_pool())
    expected = sum(sum(n for _, n in p["real"]) + len(p["synthetic"]) for p in PAIR_PLAN.values())
    assert len(pairs) == expected
    assert [p["pair_id"] for p in pairs] == [f"P{i:02d}" for i in range(1, expected + 1)]
    for role, plan in PAIR_PLAN.items():
        job = [p for p in pairs if p["job_role"] == role]
        assert sorted(p["cv_id"] for p in job if p["cv_source"] == "synthetic") == sorted(plan["synthetic"])
        assert any(p["eval_split"] == "dev" for p in job) and any(p["eval_split"] == "test" for p in job)


def test_select_pairs_uses_each_real_cv_once_and_is_reproducible():
    pairs = select_pairs(_pool())
    real = [p["cv_id"] for p in pairs if p["cv_source"] == "real"]
    assert len(real) == len(set(real))
    assert select_pairs(_pool()) == pairs  # same seed -> same pairs
    assert select_pairs(_pool(), seed=7) != pairs


def test_synthetic_split_is_deterministic():
    assert synthetic_split("syn04") == "dev" and synthetic_split("syn05") == "test"


def test_extraction_sample_is_spread_over_categories():
    sample = sample_extraction_checks(_pool(), n=20)
    assert len(sample) == 20 and all(s["eval_split"] == "test" for s in sample)
    counts = {}
    for s in sample:
        counts[s["category"]] = counts.get(s["category"], 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1  # round-robin over categories


def test_validate_match():
    assert validate_match(2, "Strong Java, but only 3 years of experience") == []
    assert validate_match(None, "Strong Java, but only 3 years") == ["Choose a grade (0-3)."]
    assert len(validate_match(5, "short")) == 2


def test_upsert_relabel_replaces_only_the_same_annotator(tmp_path: Path):
    path = tmp_path / "matches.csv"
    upsert(path, match_row("P01", "AB", 2, "First opinion about this pair", {}), ("pair_id", "annotator"), MATCH_COLUMNS)
    upsert(path, match_row("P01", "CD", 3, "Second annotator's opinion", {}), ("pair_id", "annotator"), MATCH_COLUMNS)
    upsert(path, match_row("P01", "AB", 1, "Changed my mind after rereading", {"M1": "missing"}),
           ("pair_id", "annotator"), MATCH_COLUMNS)

    rows = {(r["pair_id"], r["annotator"]): r for r in read_rows(path)}
    assert len(rows) == 2  # AB's label replaced, CD's kept (inter-annotator agreement stays possible)
    assert rows[("P01", "AB")]["grade"] == "1"
    assert json.loads(rows[("P01", "AB")]["requirements"]) == {"M1": "missing"}
    assert rows[("P01", "CD")]["grade"] == "3"


def test_match_row_keeps_only_assessed_requirements():
    row = match_row("P02", " AB ", 3, "  Excellent fit  ", {"M1": "met", "M2": "—", "N1": "partial"})
    assert row["annotator"] == "AB" and row["reason"] == "Excellent fit"
    assert json.loads(row["requirements"]) == {"M1": "met", "N1": "partial"}


def test_auto_check_rates_each_section_by_values_found_in_the_text():
    from recruitment_ai.evaluation.labeling import AUTO_CHECK_ANNOTATOR, auto_check

    key = {
        "full_name": "Lindsay Beer", "email": None, "phone": "+1 (555) 753 8128",
        "experience": [{"title": "Python Developer", "company": "Kris and Sons"}],
        "education": [{"degree": "Bachelor's", "institution": "Invented University"}],
        "skills": [{"name": n} for n in ("Python", "Docker", "Kafka", "Git", "SQL", "Rust", "Go", "Java", "C", "Perl")],
    }
    text = ("Lindsay Beer +1 (555) 753-8128. Python Developer at Kris and Sons. Bachelor's degree. "
            "Python, Docker, Kafka, Git, SQL, Go, Java, C, Perl")
    row = auto_check(key, text, "r1")
    assert row["annotator"] == AUTO_CHECK_ANNOTATOR and row["resume_id"] == "r1"
    assert row["identity"] == "correct" and row["experience"] == "correct"
    assert row["education"] == "major_errors"   # 1 of 2 values found (50 %)
    assert row["skills"] == "correct"           # 9 of 10 found (90 %)
    assert "Invented University" in row["notes"] and "Rust" in row["notes"]


def test_auto_check_without_values_is_not_applicable():
    from recruitment_ai.evaluation.labeling import auto_check

    row = auto_check({"experience": [], "education": [], "skills": []}, "some text", "r2")
    assert row["identity"] == row["experience"] == row["education"] == row["skills"] == "not_applicable"
