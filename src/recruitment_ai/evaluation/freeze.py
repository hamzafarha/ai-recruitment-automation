"""Freeze the evaluation set (Step 1c, last stage) and verify it later.

``freeze``  checks that every item is labeled, picks the reference labels, measures
            inter-annotator agreement, writes ``manifest.csv`` (one checksum per file,
            including the git-ignored real CVs and answer keys) and ``labels/summary.json``.
``verify``  recomputes the checksums: any changed or missing file means results are no
            longer comparable with earlier runs.
"""

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Collection
from itertools import combinations
from pathlib import Path
from typing import Any

from recruitment_ai.evaluation.eval_set import read_rows

GRADE_VALUES = [0, 1, 2, 3]
MANIFEST_COLUMNS = ["path", "kind", "in_git", "bytes", "sha256"]
_SPLIT_DIRS = {"dev": "validation", "test": "test"}


# --------------------------------------------------------------------------- agreement


def cohen_kappa(a: list[int], b: list[int], labels: list[int], weighted: bool = False) -> float:
    """Cohen's kappa between two annotators' labels on the same items.

    ``weighted=True`` gives quadratic-weighted kappa, suited to ordinal grades: a 3-vs-2
    disagreement costs less than a 3-vs-0 one. 1 = perfect, 0 = chance level.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of the same length")
    k, n = len(labels), len(a)
    index = {lab: i for i, lab in enumerate(labels)}
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[index[x]][index[y]] += 1 / n
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    def weight(i: int, j: int) -> float:
        if weighted:
            return (i - j) ** 2 / (k - 1) ** 2
        return 0.0 if i == j else 1.0

    disagreement_obs = sum(weight(i, j) * observed[i][j] for i in range(k) for j in range(k))
    disagreement_exp = sum(weight(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    if disagreement_exp == 0:
        return 1.0
    return 1 - disagreement_obs / disagreement_exp


def agreement(matches: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Agreement statistics for every pair of annotators who graded common pairs."""
    by_annotator: dict[str, dict[str, int]] = {}
    for m in matches:
        by_annotator.setdefault(m["annotator"], {})[m["pair_id"]] = int(m["grade"])
    results = []
    for first, second in combinations(sorted(by_annotator), 2):
        common = sorted(set(by_annotator[first]) & set(by_annotator[second]))
        if len(common) < 2:
            continue
        a = [by_annotator[first][p] for p in common]
        b = [by_annotator[second][p] for p in common]
        results.append({
            "annotators": [first, second],
            "common_pairs": len(common),
            "exact_agreement": round(sum(x == y for x, y in zip(a, b, strict=True)) / len(common), 3),
            "within_one_grade": round(sum(abs(x - y) <= 1 for x, y in zip(a, b, strict=True)) / len(common), 3),
            "cohen_kappa": round(cohen_kappa(a, b, GRADE_VALUES), 3),
            "quadratic_weighted_kappa": round(cohen_kappa(a, b, GRADE_VALUES, weighted=True), 3),
        })
    return results


# --------------------------------------------------------------------------- reference labels


def reference_matches(matches: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
    """The reference grade of each pair comes from the primary annotator (the one with most
    labels); the other annotators are only used to measure agreement."""
    counts = Counter(m["annotator"] for m in matches)
    if not counts:
        return None, []
    primary = sorted(counts, key=lambda a: (-counts[a], a))[0]
    return primary, [m for m in matches if m["annotator"] == primary]


def completeness(pairs: list[dict[str, str]], matches: list[dict[str, str]],
                 sample: list[dict[str, str]], checks: list[dict[str, str]]) -> dict[str, list[str]]:
    """Items still missing a label (by the primary annotator for pairs, by anyone for checks)."""
    _, reference = reference_matches(matches)
    graded = {m["pair_id"] for m in reference}
    checked = {c["resume_id"] for c in checks}
    return {
        "pairs_without_grade": [p["pair_id"] for p in pairs if p["pair_id"] not in graded],
        "cvs_without_check": [s["resume_id"] for s in sample if s["resume_id"] not in checked],
    }


def label_summary(pairs: list[dict[str, str]], matches: list[dict[str, str]],
                  checks: list[dict[str, str]]) -> dict[str, Any]:
    primary, reference = reference_matches(matches)
    pair_info = {p["pair_id"]: p for p in pairs}
    grades_by_job: dict[str, Counter] = {}
    grades_by_split: dict[str, Counter] = {}
    requirement_status: Counter = Counter()
    for m in reference:
        p = pair_info.get(m["pair_id"])
        if p is None:
            continue
        grades_by_job.setdefault(p["job_role"], Counter())[m["grade"]] += 1
        grades_by_split.setdefault(p["eval_split"], Counter())[m["grade"]] += 1
        requirement_status.update(json.loads(m["requirements"] or "{}").values())
    check_values: dict[str, Counter] = {}
    for c in checks:
        for section in ("identity", "experience", "education", "skills"):
            check_values.setdefault(section, Counter())[c[section]] += 1
    return {
        "primary_annotator": primary,
        "annotators": sorted({m["annotator"] for m in matches}),
        "graded_pairs": len(reference),
        "grades_by_job": {k: dict(sorted(v.items())) for k, v in sorted(grades_by_job.items())},
        "grades_by_split": {k: dict(sorted(v.items())) for k, v in sorted(grades_by_split.items())},
        "requirement_labels": dict(requirement_status),
        "pairs_with_requirement_labels": sum(1 for m in reference if json.loads(m["requirements"] or "{}")),
        "extraction_checks": len(checks),
        "extraction_check_results": {k: dict(v) for k, v in check_values.items()},
        "agreement": agreement(matches),
    }


# --------------------------------------------------------------------------- manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_files(eval_dir: Path, interim_dir: Path, raw_pdf_dir: Path, root: Path) -> list[dict[str, Any]]:
    """Every file the evaluation depends on, with its checksum (relative paths from ``root``)."""
    entries: list[tuple[Path, str, bool]] = []
    for path in sorted(eval_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.csv", "README.md"} and path.suffix != ".tmp":
            kind = path.relative_to(eval_dir).parts[0]
            entries.append((path, kind, True))
    for row in read_rows(eval_dir / "cv_pool.csv"):
        entries.append((raw_pdf_dir / row["category"] / f"{row['resume_id']}.pdf", "real_cv_pdf", False))
        entries.append((interim_dir / _SPLIT_DIRS[row["eval_split"]] / f"{row['resume_id']}.json",
                        "real_cv_answer_key", False))
    return [
        {
            "path": path.resolve().relative_to(root.resolve()).as_posix(), "kind": kind,
            "in_git": "yes" if in_git else "no", "bytes": path.stat().st_size, "sha256": sha256(path),
        }
        for path, kind, in_git in entries
    ]


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def verify(manifest_path: Path, root: Path, skip_kinds: Collection[str] = ()) -> dict[str, list[str]]:
    """Compare the files on disk with the frozen manifest.

    ``skip_kinds``: kinds of files the caller never reads (e.g. "real_cv_pdf" for a job that uses
    the real CVs' dataset text: their PDFs are not uploaded to Kaggle).
    """
    changed, missing = [], []
    for row in read_rows(manifest_path):
        if row["kind"] in skip_kinds:
            continue
        path = root / row["path"]
        if not path.exists():
            missing.append(row["path"])
        elif sha256(path) != row["sha256"]:
            changed.append(row["path"])
    return {"changed": changed, "missing": missing}


# --------------------------------------------------------------------------- datasheet


def _table(counter: dict[str, Any]) -> str:
    return ", ".join(f"{k}: {v}" for k, v in counter.items()) or "—"


def datasheet(eval_dir: Path, summary: dict[str, Any], manifest: list[dict[str, Any]],
              status: str, frozen_at: str) -> str:
    """Markdown datasheet of the evaluation set (written to eval_dir/README.md)."""
    pool = read_rows(eval_dir / "cv_pool.csv")
    pairs = read_rows(eval_dir / "labels" / "pairs.csv")
    synth = [json.loads(p.read_text(encoding="utf-8"))["meta"]
             for p in sorted((eval_dir / "synthetic" / "profiles").glob("*.json"))]
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((eval_dir / "jobs").glob("*.json"))]
    pool_split = Counter(r["eval_split"] for r in pool)
    pair_split = Counter(p["eval_split"] for p in pairs)
    kinds = Counter(m["kind"] for m in manifest)
    agreement_lines = [
        f"- {a['annotators'][0]} vs {a['annotators'][1]} on {a['common_pairs']} pairs: exact agreement "
        f"{a['exact_agreement']:.0%}, within one grade {a['within_one_grade']:.0%}, Cohen's κ {a['cohen_kappa']:.2f}, "
        f"quadratic-weighted κ {a['quadratic_weighted_kappa']:.2f}"
        for a in summary["agreement"]
    ] or ["- Not measured: a single (model) annotator. Adding a human annotator on ≥ 15 common pairs is the main way "
          "to strengthen these labels."]
    job_lines = [
        f"| {j['role']} | [{j['source']['job_id']}] {j['title']} ({j['source']['company']}) | "
        f"{sum(r['id'].startswith('M') for r in j['requirements'])} | "
        f"{sum(r['id'].startswith('N') for r in j['requirements'])} |"
        for j in jobs
    ]
    return f"""# Evaluation set eval_v1 — datasheet

**Status:** {status} · **frozen at:** {frozen_at} · **files under checksum:** {len(manifest)}
Generated by `python scripts/build_eval_set.py freeze`; verify with `python scripts/build_eval_set.py verify`.

## Purpose

A fixed "exam with an answer key" for the AI Recruitment Automation pipeline:

- **Extraction (DSO1)**: CV document → `CandidateProfile`, compared field by field with the answer keys.
- **Matching (DSO3)**: CV–job pairs graded 0–3 with the rubric KB-RR-002 (reference labels: see "Labels"), compared with the system's grades,
  overall and requirement by requirement.
- **Splits**: tune prompts and settings on **dev** only; report final numbers on **test** only.

## Composition

| Part | Content |
|---|---|
| Real CVs (`cv_pool.csv`) | {len(pool)} scanned PDF CVs from the Resume Parser dataset: dev {pool_split['dev']}, test {pool_split['test']} |
| Synthetic CVs (`synthetic/`) | {len(synth)} profiles ({sum(m['language'] == 'fr' for m in synth)} in French), rendered as {sum(len(m['formats']) for m in synth)} documents (text PDF in 3 layouts, DOCX) |
| Job descriptions (`jobs/`) | {len(jobs)} real postings from Tech Jobs 2026 with structured requirements |
| Graded pairs (`labels/`) | {len(pairs)} CV–job pairs: dev {pair_split['dev']}, test {pair_split['test']} |
| Extraction checks (`labels/`) | {summary['extraction_checks']} real CVs whose answer keys were checked automatically against the CV text |

| Role | Posting | Must-haves | Nice-to-haves |
|---|---|---|---|
{chr(10).join(job_lines)}

## How it was built

1. **Real CVs**: tech-category CVs of the dataset's validation (→ dev) and test (→ test) splits, minus CVs whose text
   also appears in the train split (leakage), corrupted texts, CVs without a tech job title (misfiled categories, e.g.
   science teachers in "Data Science") and duplicate texts. Selection is rule-based and reproducible.
2. **Answer keys of real CVs**: the dataset's labels converted to `CandidateProfile` (dates normalised, junk values
   removed). They are LLM-made "silver" labels (96 % of short values found verbatim in the CV text); a sample is
   checked by hand against the images.
3. **Synthetic CVs**: written as answer keys first, then rendered, so the reference is exact by construction. They
   cover what the real data lacks: NLP/LLM profiles, French CVs, digital PDF and DOCX inputs, and deliberate traps
   (keyword stuffing, synonyms only, equivalent cloud, other language stack, career change, overqualification,
   adjacent role, missing dates). No expected grade is stored.
4. **Job descriptions**: one posting per target role, requirements taken from the posting's own wording
   (must-have = required, nice-to-have = preferred); legal checks (work authorisation, background checks) are not scored.
5. **Pairs**: per job, real CVs from the same, adjacent and unrelated categories plus synthetic CVs targeting that job
   (fixed seed).

## Labels

- Annotators: {', '.join(summary['annotators']) or '—'} (reference annotator: {summary['primary_annotator'] or '—'}).
- **Provenance of the match grades: model-made "silver" labels.** The reference grades were produced by Claude Opus
  5.5 applying the written rubric KB-RR-002 (v1.1) requirement by requirement (met / partial / missing per
  must-have, importance-weighted coverage, grade definitions), with a written reason per pair. The system under
  evaluation uses Qwen2.5, a different model, so it is not graded against its own outputs. Human annotators can add
  labels later in the labeling app (`frontend/labeling_app.py`); agreement with the reference is then measured below.
- **Provenance of the extraction checks: automatic.** Each checked answer key is compared with the dataset's own CV
  text: share of identity, experience, education and skill values found verbatim (≥ 90 % correct, ≥ 70 % minor
  errors, otherwise major errors).
- Reference grades by job: {_table(summary['grades_by_job'])}
- Reference grades by split: {_table(summary['grades_by_split'])}
- Requirement-level labels: {summary['pairs_with_requirement_labels']} pairs; statuses {_table(summary['requirement_labels'])}
- Extraction checks: {_table(summary['extraction_check_results'])}

**Inter-annotator agreement**

{chr(10).join(agreement_lines)}

## Known limitations

- Real CVs are English, single-page scans from US résumé templates; 55 % of tech CVs share a template identity
  ("Jessica Claire"), so identity fields are only scored on the others.
- Real answer keys are LLM-made (silver); they were verified automatically on a sample (text groundedness), not by
  a human looking at the images.
- Match grades are model-made (silver), not human judgments; no human agreement has been measured yet. Results on
  matching should be read as "agreement with a rubric-following reference", not with recruiters.
- {len(pairs)} graded pairs is small: confidence intervals on matching scores are wide; report them.
- Synthetic CVs are written by the project team and may be easier or more "textbook" than real CVs.
- The data posting (internship) has an adapted education requirement (see its `adaptations` field).
- An existing human-labeled ranking dataset (Vanetik & Kogan, 2023) was considered and not used: its two annotators
  agree at chance level (Kendall τ = 0.14) on five near-identical software-developer postings.

## Privacy and licences

- Real CVs contain personal data: PDFs and answer keys stay in git-ignored folders (`data/raw/**/hf/`,
  `data/interim/`); this folder only stores ids, checksums, job postings, synthetic data and labels.
  Kinds of files under checksum: {_table(dict(kinds))}.
- Sources: Resume Parser dataset (research use; source terms apply), Tech Jobs 2026 (academic use only),
  ESCO v1.2.1 (European Commission). Synthetic CVs use fictional names and reserved example contact details.
"""
