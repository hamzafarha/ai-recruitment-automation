"""Build the evaluation set (Step 1c): CV pool, PDF download, job-description shortlist.

The evaluation set lives in data/processed/eval_v1/ and is versioned in git, but it
only stores ids, quality flags, checksums, job descriptions and human labels. CV
texts, PDFs and converted labels stay in git-ignored folders (data/raw/**/hf/,
data/interim/) and are regenerated from data/raw/ by scripts/build_eval_set.py.
"""

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from recruitment_ai.data.normalization import text_hash
from recruitment_ai.data.resume_parser import iter_split

logger = logging.getLogger(__name__)

RESUME_REPO = "md-zain/resume-parser-dataset"
# The dataset's own splits map to ours: tune prompts on "dev", report final numbers on "test"
EVAL_SPLITS = {"validation": "dev", "test": "test"}

POOL_COLUMNS = [
    "resume_id", "eval_split", "category", "template_identity", "grounded",
    "pages", "pdf_sha256", "job_titles",
]


def _is_true(value: str) -> bool:
    return value == "True"


def build_cv_pool(
    index_path: Path, curated_dir: Path
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select tech-category CVs usable for evaluation; returns (pool rows, exclusion counts).

    Excluded: CVs whose text also appears in train (leakage), corrupted texts, CVs
    whose job titles are not tech titles (misfiled, e.g. science teachers in "Data
    Science"; rule in resume_parser._TECH_TITLE), and duplicate texts (the first
    occurrence is kept, dev before test).
    """
    with index_path.open(encoding="utf-8") as f:
        candidates = [
            r for r in csv.DictReader(f)
            if r["split"] in EVAL_SPLITS and _is_true(r["is_tech_category"])
        ]
    wanted = {r["resume_id"] for r in candidates}
    hashes = {
        row["resume_id"]: text_hash(row["resume_text"])
        for split in EVAL_SPLITS
        for row in iter_split(curated_dir / f"{split}.jsonl")
        if row["resume_id"] in wanted
    }

    stats = {
        "tech_category": len(candidates), "leak_train": 0, "corrupted": 0,
        "not_tech_title": 0, "duplicate_text": 0,
    }
    pool, seen = [], set()
    for r in sorted(candidates, key=lambda r: (r["split"] != "validation", r["resume_id"])):
        if _is_true(r["leak_train"]):
            stats["leak_train"] += 1
        elif _is_true(r["corrupted"]):
            stats["corrupted"] += 1
        elif not _is_true(r["tech_title_match"]):
            stats["not_tech_title"] += 1
        elif hashes[r["resume_id"]] in seen:
            stats["duplicate_text"] += 1
        else:
            seen.add(hashes[r["resume_id"]])
            pool.append({
                "resume_id": r["resume_id"],
                "eval_split": EVAL_SPLITS[r["split"]],
                "category": r["category"],
                "template_identity": _is_true(r["template_identity"]),
                "grounded": r["grounded"],
                "pages": "",
                "pdf_sha256": "",
                "job_titles": r["job_titles"],
            })
    stats["pool"] = len(pool)
    return pool, stats


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_pdfs(pool: list[dict[str, Any]], raw_root: Path) -> dict[str, int]:
    """Fetch each pool CV's original PDF (cached), recording its page count and checksum."""
    import pymupdf
    from huggingface_hub import hf_hub_download

    stats = {"ok": 0, "failed": 0}
    for i, row in enumerate(pool, 1):
        try:
            path = Path(hf_hub_download(
                RESUME_REPO, f"raw/{row['category']}/{row['resume_id']}.pdf",
                repo_type="dataset", local_dir=raw_root,
            ))
            with pymupdf.open(path) as doc:
                row["pages"] = doc.page_count
            row["pdf_sha256"] = _sha256(path)
            stats["ok"] += 1
        except Exception as exc:  # keep going: one missing PDF must not stop the build
            logger.warning("PDF %s failed: %s", row["resume_id"], exc)
            stats["failed"] += 1
        if i % 25 == 0:
            logger.info("downloaded %d/%d PDFs", i, len(pool))
    return stats


def write_pool(pool: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=POOL_COLUMNS)
        writer.writeheader()
        writer.writerows(pool)


# --------------------------------------------------------------------------- job descriptions


def shortlist_jobs(jobs: list[dict[str, Any]], per_role: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Up to ``per_role`` candidate postings per target role, most suitable first.

    Preference: not an internship, then a description close to 4,500 characters
    (detailed requirements, not just boilerplate).
    """
    by_role: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        if job.get("role"):
            by_role.setdefault(job["role"], []).append(job)

    def rank(job: dict[str, Any]) -> tuple[bool, int]:
        internship = job.get("employment_type") == "Internship" or "intern" in job["title"].lower()
        return internship, abs(job["desc_len"] - 4500)

    return {role: sorted(items, key=rank)[:per_role] for role, items in sorted(by_role.items())}


def write_shortlist(shortlist: dict[str, list[dict[str, Any]]], path: Path, preview_chars: int = 2500) -> None:
    """Readable markdown of the shortlist, so a human can choose one posting per role."""
    lines = ["# Job description shortlist — choose one posting per role", ""]
    for role, jobs in shortlist.items():
        lines += [f"## {role}", ""]
        for job in jobs:
            lines += [
                f"### id {job['id']} — {job['title']} ({job.get('company')})",
                f"*{job.get('employment_type')} · {job.get('workplace_type')} · {job.get('location')} · "
                f"{job['desc_len']} chars*",
                "",
                "```text",
                job["description"][:preview_chars].strip(),
                "```",
                "",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_jobs(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- labeling sets

# For each job: real CVs drawn by category (same, adjacent and unrelated fields, to cover the
# whole range of fit) + synthetic CVs that target that job's traps. Expected grades are never
# stored: only the human labels decide.
PAIR_PLAN: dict[str, dict[str, list]] = {
    "backend_developer": {
        "real": [("Java Developer", 2), ("Python Developer", 1), ("React Developer", 1)],
        "synthetic": ["syn07", "syn08", "syn12", "syn04"],
    },
    "data_analyst_scientist": {
        "real": [("Data Science", 1), ("Database", 1), ("Sql Developer", 1), ("Information Technology", 1)],
        "synthetic": ["syn09", "syn10", "syn03"],
    },
    "devops_cloud": {
        "real": [("DevOps Engineer", 2), ("Information Technology", 1), ("Database", 1)],
        "synthetic": ["syn05", "syn06", "syn04"],
    },
    "ml_engineer": {
        "real": [("Data Science", 2), ("Python Developer", 1)],
        "synthetic": ["syn03", "syn11", "syn01", "syn04"],
    },
    "nlp_llm_engineer": {
        "real": [("Data Science", 1), ("Python Developer", 1)],
        "synthetic": ["syn01", "syn02", "syn03", "syn11", "syn04"],
    },
}
PAIR_COLUMNS = ["pair_id", "job_role", "cv_id", "cv_source", "eval_split"]
EXTRACTION_COLUMNS = ["resume_id", "category", "eval_split"]


def synthetic_split(cv_id: str) -> str:
    """Deterministic dev/test assignment of synthetic CVs: even ids -> dev, odd ids -> test."""
    return "dev" if int(cv_id[-2:]) % 2 == 0 else "test"


def select_pairs(pool: list[dict[str, Any]], seed: int = 42) -> list[dict[str, str]]:
    """CV-job pairs to grade by hand, following PAIR_PLAN.

    Each real CV is used once across all jobs (more diversity); the first real CV of each
    job comes from dev, the others from test. Same seed -> same pairs.
    """
    import random

    rng = random.Random(seed)
    used: set[str] = set()
    pairs: list[dict[str, str]] = []
    for role, plan in PAIR_PLAN.items():
        job_pairs = []
        for category, count in plan["real"]:
            for _ in range(count):
                split = "dev" if not any(p["eval_split"] == "dev" and p["cv_source"] == "real" for p in job_pairs) else "test"
                options = sorted(
                    r["resume_id"] for r in pool
                    if r["category"] == category and r["eval_split"] == split and r["resume_id"] not in used
                )
                if not options:  # fall back to the other split rather than skipping the category
                    split = "test" if split == "dev" else "dev"
                    options = sorted(
                        r["resume_id"] for r in pool
                        if r["category"] == category and r["eval_split"] == split and r["resume_id"] not in used
                    )
                cv_id = rng.choice(options)
                used.add(cv_id)
                job_pairs.append({"job_role": role, "cv_id": cv_id, "cv_source": "real", "eval_split": split})
        job_pairs += [
            {"job_role": role, "cv_id": cid, "cv_source": "synthetic", "eval_split": synthetic_split(cid)}
            for cid in plan["synthetic"]
        ]
        rng.shuffle(job_pairs)  # real and synthetic CVs interleaved, so they are not graded as two blocks
        pairs += job_pairs
    for i, pair in enumerate(pairs, 1):
        pair["pair_id"] = f"P{i:02d}"
    return pairs


def sample_extraction_checks(pool: list[dict[str, Any]], n: int = 20, seed: int = 42) -> list[dict[str, str]]:
    """Real test CVs whose answer keys a human checks against the image, spread over categories."""
    import random

    rng = random.Random(seed)
    by_category: dict[str, list[str]] = {}
    for r in pool:
        if r["eval_split"] == "test":
            by_category.setdefault(r["category"], []).append(r["resume_id"])
    for ids in by_category.values():
        ids.sort()
        rng.shuffle(ids)
    sample, categories = [], sorted(by_category)
    while len(sample) < n and any(by_category.values()):
        for category in categories:  # round-robin: every category gets its share
            if by_category[category] and len(sample) < n:
                sample.append({"resume_id": by_category[category].pop(), "category": category, "eval_split": "test"})
    return sample


def write_rows(rows: list[dict[str, Any]], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: r[c] for c in columns} for r in rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- synthetic CVs


SYNTHETIC_COLUMNS = ["cv_id", "language", "layout", "format", "file", "sha256", "case"]


def render_synthetic(synthetic_dir: Path) -> list[dict[str, str]]:
    """Render every synthetic profile (``profiles/*.json``) into its documents (``files/``).

    The profile is the answer key; each document is generated from it. Returns one
    index row per rendered file (with checksum), also written to ``index.csv``.
    """
    from recruitment_ai.evaluation.render_cv import render_docx, render_pdf
    from recruitment_ai.extraction.schemas import CandidateProfile

    rows = []
    for path in sorted((synthetic_dir / "profiles").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        meta, profile = data["meta"], CandidateProfile.model_validate(data["profile"])
        for fmt in meta["formats"]:
            out = synthetic_dir / "files" / f"{meta['id']}.{fmt}"
            if fmt == "pdf":
                render_pdf(profile, out, lang=meta["language"], layout=meta["layout"])
            elif fmt == "docx":
                render_docx(profile, out, lang=meta["language"])
            else:
                raise ValueError(f"{path.name}: unsupported format {fmt!r}")
            rows.append({
                "cv_id": meta["id"], "language": meta["language"],
                "layout": meta["layout"] if fmt == "pdf" else "docx",
                "format": fmt, "file": out.relative_to(synthetic_dir).as_posix(),
                "sha256": _sha256(out), "case": meta["case"],
            })

    with (synthetic_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SYNTHETIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
