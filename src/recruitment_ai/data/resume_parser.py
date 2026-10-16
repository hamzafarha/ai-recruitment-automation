"""Prepare the Resume Parser dataset (md-zain/resume-parser-dataset) for evaluation.

Two jobs:
1. ``convert_label``: map the dataset's ``target_json`` (its own ResumeOutput schema)
   to our ``CandidateProfile``, so extraction output and reference can be compared.
2. ``tag_cv``: flag the quality problems found in notebooks/00_data_inspection.ipynb
   (train/test leakage, corrupted text, template identities, non-tech categories).

Raw files are never modified: outputs go to data/interim/resume_parser/.
"""

import csv
import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from recruitment_ai.data.normalization import (
    PRESENT,
    clean_value,
    normalize_date,
    text_hash,
    text_key,
)
from recruitment_ai.extraction.schemas import (
    CandidateProfile,
    Certification,
    Education,
    Experience,
    Language,
    Project,
    Skill,
)

logger = logging.getLogger(__name__)

TECH_CATEGORIES = {
    "Data Science", "Python Developer", "DevOps Engineer", "Java Developer",
    "React Developer", "Sql Developer", "Information Technology", "Database",
}

# Their skill groups -> our Skill.category
SKILL_GROUPS = {
    "technical": "Technical",
    "tools_and_software": "Tool",
    "domain": "Domain",
    "soft": "Soft",
}

# Job titles that really indicate a tech profile. Kept strict on purpose:
# "Boiler Plant Operator/REACT" must not match "react", and "Science Teacher" must not
# match "Data Science" (the dataset's categories were assigned by keyword, so its
# "Data Science" category is full of science teachers).
_TECH_TITLE = re.compile(
    r"\b("
    r"software|developer|programmer|devops|sre|site reliability|dba|"
    r"data (scientist|analyst|engineer|architect)|data science|machine learning|\bml\b|\bai\b engineer|"
    r"database (administrator|administration|developer|engineer|analyst|manager|specialist|support)|"
    r"information technology|information systems?|"
    r"(it|i\.t\.) (specialist|support|manager|analyst|consultant|technician|administrator|director)|"
    r"systems? (administrator|engineer|analyst|architect)|network (engineer|administrator|services)|"
    r"web (developer|designer)|front[- ]?end|back[- ]?end|full[- ]?stack|"
    r"(qa|test|automation|cloud|platform|solutions?|integration|etl|bi) (engineer|architect|developer|analyst)|"
    r"business intelligence|technical lead|tech lead"
    r")\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- conversion


def _skill_list(names: list[Any], category: str, seen: set[str]) -> list[Skill]:
    skills = []
    for name in names or []:
        name = clean_value(name)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            skills.append(Skill(name=name, category=category))
    return skills


def _language(raw: str) -> Language | None:
    """'English (Native)' / 'French - fluent' / 'Spanish' -> Language."""
    raw = clean_value(raw)
    if not raw:
        return None
    if m := re.fullmatch(r"(.+?)\s*[(\-–:]\s*([^)]+?)\)?", raw):
        return Language(language=m[1].strip(), proficiency=m[2].strip())
    return Language(language=raw)


def _experience(job: dict[str, Any]) -> Experience | None:
    title, company = clean_value(job.get("job_title")), clean_value(job.get("company"))
    if not title and not company:
        return None
    end = normalize_date(job.get("end_date"))
    lines = [clean_value(job.get("description"))]
    lines += [f"- {x}" for x in (job.get("responsibilities") or []) + (job.get("achievements") or []) if x]
    description = "\n".join(line for line in lines if line) or None
    return Experience(
        title=title,
        company=company,
        location=clean_value(job.get("location")),
        start_date=normalize_date(job.get("start_date")),
        end_date=end,
        is_current=bool(job.get("is_current")) or end == PRESENT,
        description=description,
    )


def _education(edu: dict[str, Any]) -> Education | None:
    institution, degree = clean_value(edu.get("institution")), clean_value(edu.get("degree"))
    if not institution and not degree:
        return None
    grade_parts = [clean_value(edu.get("gpa"))] + [clean_value(h) for h in edu.get("honors") or []]
    return Education(
        institution=institution,
        degree=degree,
        field_of_study=clean_value(edu.get("field_of_study")),
        start_date=normalize_date(edu.get("start_date")),
        end_date=normalize_date(edu.get("end_date") or edu.get("graduation_date")),
        grade="; ".join(p for p in grade_parts if p) or None,
    )


def convert_label(target: dict[str, Any], resume_id: str | None = None) -> CandidateProfile:
    """Map a Resume Parser ``target_json`` to our ``CandidateProfile``.

    Kept: identity, summary, experience, education, skills (+ skills used in jobs),
    certifications, projects, spoken languages.
    Dropped (no field in our schema, not used for screening): awards, publications,
    volunteering, memberships, references, links, additional_sections.
    """
    pi = target.get("personal_information") or {}
    location = ", ".join(
        v for v in (clean_value(pi.get(k)) for k in ("city", "state_or_region", "country")) if v
    )

    seen: set[str] = set()
    skills: list[Skill] = []
    groups = target.get("skills") or {}
    for group, category in SKILL_GROUPS.items():
        skills += _skill_list(groups.get(group), category, seen)
    for job in target.get("work_experience") or []:  # skills only named inside a job
        skills += _skill_list(job.get("skills_used"), "Technical", seen)

    certifications = [
        Certification(
            name=name,
            issuer=clean_value(c.get("issuer")),
            issue_date=normalize_date(c.get("issue_date")),
            expiration_date=normalize_date(c.get("expiry_date")),
            credential_id=clean_value(c.get("credential_id")),
        )
        for c in target.get("certifications") or []
        if (name := clean_value(c.get("name")))
    ]
    projects = [
        Project(
            title=title,
            description=clean_value(p.get("description")),
            technologies=[t for t in (clean_value(t) for t in p.get("technologies") or []) if t],
            url=clean_value(p.get("url")),
        )
        for p in target.get("projects") or []
        if (title := clean_value(p.get("name")))
    ]
    phones = [p for p in (clean_value(x) for x in pi.get("phone_numbers") or []) if p]

    return CandidateProfile(
        id=resume_id,
        full_name=clean_value(pi.get("full_name")),
        email=clean_value(pi.get("email")),
        phone=phones[0] if phones else None,
        location=location or None,
        summary=clean_value(target.get("professional_summary")) or clean_value(target.get("career_objective")),
        experience=[e for e in map(_experience, target.get("work_experience") or []) if e],
        education=[e for e in map(_education, target.get("education") or []) if e],
        skills=skills,
        certifications=certifications,
        projects=projects,
        languages=[lang for lang in map(_language, groups.get("spoken_languages") or []) if lang],
    )


# --------------------------------------------------------------------------- tagging


def _short_values(obj: Any, max_len: int = 40) -> Iterator[str]:
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _short_values(v, max_len)
    elif isinstance(obj, list):
        for v in obj:
            yield from _short_values(v, max_len)
    elif isinstance(obj, str) and 2 <= len(obj) <= max_len and text_key(obj):
        yield obj


def groundedness(resume_text: str, target: dict[str, Any]) -> float:
    """Share of short label values found word-for-word in the CV text (1.0 = all found)."""
    text = f" {text_key(resume_text)} "
    values = list(_short_values(target))
    if not values:
        return 1.0
    return sum(f" {text_key(v)} " in text for v in values) / len(values)


def tag_cv(row: dict[str, Any], target: dict[str, Any], train_hashes: set[str]) -> dict[str, Any]:
    """Quality flags for one CV; the evaluation-set builder decides what to keep."""
    text = row["resume_text"]
    titles = [j.get("job_title") or "" for j in target.get("work_experience") or []]
    professional_title = (target.get("personal_information") or {}).get("professional_title") or ""
    return {
        "resume_id": row["resume_id"],
        "category": row["category"],
        "is_tech_category": row["category"] in TECH_CATEGORIES,
        # decides whether a tech-category CV is really a tech profile (evaluation pool filter)
        "tech_title_match": bool(_TECH_TITLE.search(" | ".join(titles + [professional_title]))),
        "leak_train": text_hash(text) in train_hashes,
        "corrupted": bool(re.search(r"[a-z]Claire[a-z]", text)),
        "template_identity": "resumesample@example.com" in text.lower(),
        "grounded": round(groundedness(text, target), 3),
        "text_len": len(text),
        "job_titles": " | ".join(t for t in titles if t)[:200],
    }


# --------------------------------------------------------------------------- pipeline


def iter_split(path: Path) -> Iterator[dict[str, Any]]:
    """Stream rows of a curated .jsonl split with ``target_json`` parsed to a dict."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            target = row["target_json"]
            row["target"] = json.loads(target) if isinstance(target, str) else target
            yield row


def prepare_resume_parser(
    raw_dir: Path = Path("data/raw/resumes/resume_parser/hf/curated"),
    out_dir: Path = Path("data/interim/resume_parser"),
    splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, int]:
    """Convert labels and tag CVs of the given splits; returns counts per split.

    Writes ``<out_dir>/<split>/<resume_id>.json`` (CandidateProfile reference) and
    ``<out_dir>/index.csv`` (one row of quality flags per CV).
    """
    train_hashes = {text_hash(r["resume_text"]) for r in iter_split(raw_dir / "train.jsonl")}
    logger.info("Hashed %d train CVs for leakage detection", len(train_hashes))

    index_rows, counts = [], {}
    for split in splits:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        counts[split] = 0
        for row in iter_split(raw_dir / f"{split}.jsonl"):
            profile = convert_label(row["target"], resume_id=row["resume_id"])
            (split_dir / f"{row['resume_id']}.json").write_text(
                profile.model_dump_json(indent=2, exclude={"created_at"}), encoding="utf-8"
            )
            index_rows.append({"split": split, **tag_cv(row, row["target"], train_hashes)})
            counts[split] += 1

    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    return counts
