"""Tests for freezing and verifying the evaluation set."""

import json
from pathlib import Path

import pytest

from recruitment_ai.evaluation.freeze import (
    GRADE_VALUES,
    agreement,
    cohen_kappa,
    completeness,
    label_summary,
    reference_matches,
    sha256,
    verify,
    write_manifest,
)


def _match(pair: str, annotator: str, grade: int, reqs: dict | None = None) -> dict[str, str]:
    return {"pair_id": pair, "annotator": annotator, "grade": str(grade), "reason": "because",
            "requirements": json.dumps(reqs or {}), "labeled_at": "t"}


def test_kappa_reference_values():
    assert cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3], GRADE_VALUES) == 1.0          # perfect agreement
    assert cohen_kappa([0, 1, 0, 1], [0, 1, 1, 0], GRADE_VALUES) == pytest.approx(0.0)  # chance level
    assert cohen_kappa([0, 3], [3, 0], GRADE_VALUES, weighted=True) == pytest.approx(-1.0)  # opposite


def test_weighted_kappa_forgives_near_misses():
    a = [3, 2, 1, 0, 3, 1]
    near = [2, 2, 1, 0, 3, 2]   # off by one grade twice
    far = [0, 2, 1, 0, 3, 3]    # off by three and two grades
    assert cohen_kappa(a, near, GRADE_VALUES, weighted=True) > cohen_kappa(a, far, GRADE_VALUES, weighted=True)


def test_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1], GRADE_VALUES)


def test_agreement_on_common_pairs_only():
    matches = [_match("P01", "AB", 3), _match("P02", "AB", 1), _match("P03", "AB", 0),
               _match("P01", "CD", 3), _match("P02", "CD", 2), _match("P09", "CD", 0)]
    [result] = agreement(matches)
    assert result["annotators"] == ["AB", "CD"] and result["common_pairs"] == 2
    assert result["exact_agreement"] == 0.5 and result["within_one_grade"] == 1.0


def test_reference_annotator_is_the_one_with_most_labels():
    matches = [_match("P01", "CD", 1), _match("P01", "AB", 3), _match("P02", "AB", 2)]
    primary, reference = reference_matches(matches)
    assert primary == "AB" and [m["pair_id"] for m in reference] == ["P01", "P02"]


def test_completeness_and_summary():
    pairs = [{"pair_id": "P01", "job_role": "ml_engineer", "eval_split": "dev"},
             {"pair_id": "P02", "job_role": "ml_engineer", "eval_split": "test"}]
    matches = [_match("P01", "AB", 3, {"M1": "met", "M2": "partial"})]
    sample = [{"resume_id": "r1"}, {"resume_id": "r2"}]
    checks = [{"resume_id": "r1", "annotator": "AB", "identity": "correct", "experience": "minor_errors",
               "education": "correct", "skills": "correct", "notes": "", "labeled_at": "t"}]
    assert completeness(pairs, matches, sample, checks) == {"pairs_without_grade": ["P02"], "cvs_without_check": ["r2"]}

    summary = label_summary(pairs, matches, checks)
    assert summary["grades_by_job"] == {"ml_engineer": {"3": 1}}
    assert summary["requirement_labels"] == {"met": 1, "partial": 1}
    assert summary["extraction_check_results"]["experience"] == {"minor_errors": 1}
    assert summary["agreement"] == []  # one annotator: nothing to compare


def test_verify_detects_changed_and_missing_files(tmp_path: Path):
    kept, edited, deleted = (tmp_path / n for n in ("kept.txt", "edited.txt", "deleted.txt"))
    for f in (kept, edited, deleted):
        f.write_text("original", encoding="utf-8")
    rows = [{"path": f.name, "kind": "x", "in_git": "yes", "bytes": 8, "sha256": sha256(f)} for f in (kept, edited, deleted)]
    write_manifest(rows, tmp_path / "manifest.csv")

    assert verify(tmp_path / "manifest.csv", tmp_path) == {"changed": [], "missing": []}
    edited.write_text("modified", encoding="utf-8")
    deleted.unlink()
    assert verify(tmp_path / "manifest.csv", tmp_path) == {"changed": ["edited.txt"], "missing": ["deleted.txt"]}
