"""Data access and storage for the human labeling app (frontend/labeling_app.py).

Two labeling tasks over the evaluation set:
- match grading: a human grades each CV-job pair 0-3 with the rubric KB-RR-002
  (optionally requirement by requirement) -> labels/matches.csv
- extraction check: a human compares a real CV image with its converted answer key
  -> labels/extraction_checks.csv

Labels are keyed by (item, annotator), so several people can label the same items and
their agreement can be measured. No model output is ever shown (no anchoring).
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recruitment_ai.evaluation.eval_set import read_rows

GRADES = {3: "Strong fit", 2: "Partial fit", 1: "Weak fit", 0: "No fit"}
REQUIREMENT_STATUSES = ["met", "partial", "missing"]
CHECK_SECTIONS = ["identity", "experience", "education", "skills"]
CHECK_VALUES = ["correct", "minor_errors", "major_errors", "not_applicable"]

MATCH_COLUMNS = ["pair_id", "annotator", "grade", "reason", "requirements", "labeled_at"]
CHECK_COLUMNS = ["resume_id", "annotator", *CHECK_SECTIONS, "notes", "labeled_at"]
MIN_REASON_CHARS = 15

_SPLIT_DIRS = {"dev": "validation", "test": "test"}  # our split -> Resume Parser split folder


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert(path: Path, row: dict[str, Any], keys: tuple[str, ...], columns: list[str]) -> None:
    """Insert ``row``, or replace the existing row with the same key values (a re-label)."""
    rows = [r for r in read_rows(path) if any(r[k] != str(row[k]) for k in keys)]
    rows.append({c: row.get(c, "") for c in columns})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)  # atomic: a crash never leaves a half-written label file


def validate_match(grade: int | None, reason: str) -> list[str]:
    """Problems that prevent saving a match label (empty list = valid)."""
    errors = []
    if grade not in GRADES:
        errors.append("Choose a grade (0-3).")
    if len(reason.strip()) < MIN_REASON_CHARS:
        errors.append(f"Write a reason of at least {MIN_REASON_CHARS} characters.")
    return errors


def match_row(pair_id: str, annotator: str, grade: int, reason: str, requirements: dict[str, str]) -> dict[str, Any]:
    assessed = {k: v for k, v in requirements.items() if v in REQUIREMENT_STATUSES}
    return {
        "pair_id": pair_id, "annotator": annotator.strip(), "grade": grade, "reason": reason.strip(),
        "requirements": json.dumps(assessed, ensure_ascii=False), "labeled_at": now(),
    }


AUTO_CHECK_ANNOTATOR = "auto-groundedness"


def _section_values(key: dict[str, Any]) -> dict[str, list[str]]:
    """Answer-key values per checked section (dates are normalised, so they are left out)."""
    return {
        "identity": [v for v in (key.get("full_name"), key.get("email"), key.get("phone")) if v],
        "experience": [v for x in key.get("experience", []) for v in (x.get("title"), x.get("company")) if v],
        "education": [v for e in key.get("education", []) for v in (e.get("degree"), e.get("institution")) if v],
        "skills": [s["name"] for s in key.get("skills", [])],
    }


def auto_check(key: dict[str, Any], cv_text: str, resume_id: str) -> dict[str, Any]:
    """Automatic extraction check: are the answer key's values present in the CV's own text?

    Per section, the share of values found verbatim (case/punctuation-insensitive) gives
    correct (>= 90 %), minor_errors (>= 70 %) or major_errors; values not found go to notes.
    """
    from recruitment_ai.data.normalization import text_key

    text = f" {text_key(cv_text)} "
    row: dict[str, Any] = {"resume_id": resume_id, "annotator": AUTO_CHECK_ANNOTATOR}
    notes = []
    for section, values in _section_values(key).items():
        values = [v for v in values if text_key(v)]
        if not values:
            row[section] = "not_applicable"
            continue
        missing = [v for v in values if f" {text_key(v)} " not in text]
        found = 1 - len(missing) / len(values)
        row[section] = "correct" if found >= 0.9 else "minor_errors" if found >= 0.7 else "major_errors"
        if missing:
            notes.append(f"{section} {found:.0%} found; not in text: {', '.join(missing[:5])}")
    row["notes"] = " | ".join(notes)
    row["labeled_at"] = now()
    return row


class LabelingData:
    """Read-only access to everything the labeling app shows."""

    def __init__(
        self,
        eval_dir: Path = Path("data/processed/eval_v1"),
        interim_dir: Path = Path("data/interim/resume_parser"),
        raw_pdf_dir: Path = Path("data/raw/resumes/resume_parser/hf/raw"),
    ) -> None:
        self.eval_dir, self.interim_dir, self.raw_pdf_dir = eval_dir, interim_dir, raw_pdf_dir
        self.labels_dir = eval_dir / "labels"
        self._pool = {r["resume_id"]: r for r in read_rows(eval_dir / "cv_pool.csv")}

    @property
    def matches_path(self) -> Path:
        return self.labels_dir / "matches.csv"

    @property
    def checks_path(self) -> Path:
        return self.labels_dir / "extraction_checks.csv"

    def pairs(self) -> list[dict[str, str]]:
        return read_rows(self.labels_dir / "pairs.csv")

    def extraction_sample(self) -> list[dict[str, str]]:
        return read_rows(self.labels_dir / "extraction_sample.csv")

    def job(self, role: str) -> dict[str, Any]:
        return json.loads((self.eval_dir / "jobs" / f"{role}.json").read_text(encoding="utf-8"))

    def cv_pdf(self, cv_id: str, source: str) -> Path:
        if source == "synthetic":
            return self.eval_dir / "synthetic" / "files" / f"{cv_id}.pdf"
        return self.raw_pdf_dir / self._pool[cv_id]["category"] / f"{cv_id}.pdf"

    def answer_key(self, cv_id: str, source: str, eval_split: str) -> dict[str, Any]:
        if source == "synthetic":
            path = self.eval_dir / "synthetic" / "profiles" / f"{cv_id}.json"
            return json.loads(path.read_text(encoding="utf-8"))["profile"]
        path = self.interim_dir / _SPLIT_DIRS[eval_split] / f"{cv_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def labels(self, path: Path, annotator: str, key: str) -> dict[str, dict[str, str]]:
        """This annotator's existing labels, by item id."""
        return {r[key]: r for r in read_rows(path) if r["annotator"] == annotator.strip()}
