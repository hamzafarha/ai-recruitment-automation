"""Compare two extraction evaluation runs (e.g. before and after fine-tuning).

Reads what scripts/eval_extraction.py writes in data/evaluation/extraction/<run>/: metrics.json
(overall and per-group scores) and results.csv (one row per document). Used by the app's
"Evaluation" page; no model or GPU involved.
"""

import csv
import json
from pathlib import Path
from typing import Any

FINETUNED_SUFFIX = "-finetuned"  # added to the run name by eval_extraction.py when an adapter is used

# (label, path in metrics["overall"]); "lower is better" metrics are flagged separately
METRICS: list[tuple[str, tuple[str, ...]]] = [
    ("Overall F1", ("overall_f1",)),
    ("Experience F1", ("experience", "f1")),
    ("Dates exact", ("experience", "date_exact")),
    ("Education F1", ("education", "f1")),
    ("Skills precision", ("skills", "precision")),
    ("Skills recall", ("skills", "recall")),
    ("Skills F1", ("skills", "f1")),
    ("Identity accuracy", ("identity", "accuracy")),
]


def list_runs(root: Path) -> list[str]:
    """Names of the runs that have a metrics.json, sorted."""
    if not root.is_dir():
        return []
    return sorted(p.parent.name for p in root.glob("*/metrics.json"))


def load_run(root: Path, name: str) -> dict[str, Any]:
    """metrics.json plus the per-document rows of results.csv."""
    folder = root / name
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    if (folder / "results.csv").exists():
        with (folder / "results.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    return {"name": name, "metrics": metrics, "rows": rows}


def default_pair(names: list[str]) -> tuple[str | None, str | None]:
    """(before, after): the fine-tuned run and the same run without the adapter, if both exist."""
    after = next((n for n in names if FINETUNED_SUFFIX in n), None)
    if after is None:
        return (names[0] if names else None), None
    before = after.replace(FINETUNED_SUFFIX, "")
    return (before if before in names else next((n for n in names if n != after), None)), after


def value(summary: dict[str, Any], path: tuple[str, ...]) -> float | None:
    for key in path:
        if not isinstance(summary, dict) or summary.get(key) is None:
            return None
        summary = summary[key]
    return float(summary)


def compare_metrics(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per metric: before, after and the difference (fractions, 0.05 = 5 points)."""
    rows = []
    for label, path in METRICS:
        b, a = value(before["metrics"]["overall"], path), value(after["metrics"]["overall"], path)
        rows.append({"metric": label, "before": b, "after": a,
                     "delta": a - b if a is not None and b is not None else None})
    return rows


def compare_groups(before: dict[str, Any], after: dict[str, Any], group: str) -> list[dict[str, Any]]:
    """Overall F1 per value of a group (input_type, source, language), before and after."""
    by_before = before["metrics"].get("by", {}).get(group, {})
    by_after = after["metrics"].get("by", {}).get(group, {})
    rows = []
    for key in sorted(set(by_before) | set(by_after)):
        b = value(by_before.get(key, {}), ("overall_f1",))
        a = value(by_after.get(key, {}), ("overall_f1",))
        rows.append({group: key, "documents": (by_after.get(key) or by_before.get(key) or {}).get("cvs"),
                     "before": b, "after": a, "delta": a - b if a is not None and b is not None else None})
    return rows


def _document_f1(row: dict[str, str]) -> float | None:
    """Mean of the document's experience, education and skills F1 (the report's overall F1)."""
    values = [float(row[k]) for k in ("exp_f1", "edu_f1", "skills_f1") if row.get(k) not in (None, "")]
    return sum(values) / len(values) if values else None


def compare_documents(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Documents present in both runs, sorted from the largest improvement to the largest drop."""
    after_rows = {r["doc_id"]: r for r in after["rows"]}
    rows = []
    for b in before["rows"]:
        a = after_rows.get(b["doc_id"])
        if a is None:
            continue
        fb, fa = _document_f1(b), _document_f1(a)
        rows.append({"doc_id": b["doc_id"], "input_type": b["input_type"], "language": b["language"],
                     "before": fb, "after": fa, "delta": fa - fb if fa is not None and fb is not None else None,
                     "error_before": b["error"] == "True", "error_after": a["error"] == "True"})
    return sorted(rows, key=lambda r: r["delta"] if r["delta"] is not None else 0.0, reverse=True)
