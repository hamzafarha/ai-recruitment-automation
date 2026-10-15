"""Clean the Tech Jobs 2026 postings (Sundaydream/tech-jobs-dataset-2026).

Inspection found 144 duplicate postings (same title + company), 81 duplicate
descriptions, 9 near-empty descriptions and 1 posting with broken characters.
Descriptions are kept full-length on purpose: handling company boilerplate is
part of the RAG pipeline's job.
"""

import json
import re
from pathlib import Path
from typing import Any

from recruitment_ai.data.normalization import text_hash

MIN_DESCRIPTION_CHARS = 500

# Our 5 target roles, matched on the job title (same patterns as the inspection notebook)
ROLE_PATTERNS = {
    "ml_engineer": r"machine learning|\bml\b|mlops",
    "nlp_llm_engineer": r"\bnlp\b|natural language|\bllm\b|generative|genai",
    "data_analyst_scientist": r"data analyst|data scien|analytics",
    "backend_developer": r"backend|back-end|back end|python developer|java developer",
    "devops_cloud": r"devops|\bsre\b|site reliability|cloud engineer|platform engineer",
}


def assign_role(title: str) -> str | None:
    """First target role whose pattern matches the title, else None (still useful for RAG)."""
    for role, pattern in ROLE_PATTERNS.items():
        if re.search(pattern, title or "", re.IGNORECASE):
            return role
    return None


def clean_jobs(jobs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop unusable and duplicate postings; add ``role`` and ``desc_len``.

    Duplicates: same description text, or same (title, company) — the longest
    description is kept. Returns (clean jobs, counts of what was removed).
    """
    stats = {"input": len(jobs), "too_short": 0, "broken_chars": 0, "duplicates": 0}
    usable = []
    for job in jobs:
        description = job.get("description") or ""
        if len(description) < MIN_DESCRIPTION_CHARS:
            stats["too_short"] += 1
        elif "�" in description:
            stats["broken_chars"] += 1
        else:
            usable.append(job)

    usable.sort(key=lambda j: len(j["description"]), reverse=True)  # keep the longest version
    seen_text, seen_key, clean = set(), set(), []
    for job in usable:
        h = text_hash(job["description"])
        key = ((job.get("title") or "").strip().lower(), (job.get("company") or "").strip().lower())
        is_duplicate = h in seen_text or key in seen_key
        # Record both keys even for duplicates, so a third copy matching either one is caught
        seen_text.add(h)
        seen_key.add(key)
        if is_duplicate:
            stats["duplicates"] += 1
            continue
        clean.append({**job, "role": assign_role(job.get("title", "")), "desc_len": len(job["description"])})

    clean.sort(key=lambda j: j["id"])
    stats["output"] = len(clean)
    return clean, stats


def prepare_tech_jobs(
    raw_path: Path = Path("data/raw/job_descriptions/tech_jobs_2026/hf/jobs.json"),
    out_path: Path = Path("data/interim/tech_jobs/jobs_clean.json"),
) -> dict[str, int]:
    """Read raw postings, clean them, write ``jobs_clean.json``; returns the cleaning stats."""
    jobs, stats = clean_jobs(json.loads(raw_path.read_text(encoding="utf-8")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(jobs, indent=1, ensure_ascii=False), encoding="utf-8")
    for role in ROLE_PATTERNS:
        stats[f"role:{role}"] = sum(j["role"] == role for j in jobs)
    return stats
