"""Evidence material for matching: CV snippets, experience timeline, excerpt grounding.

- ``candidate_snippets``: a profile as small, citable pieces (one per position, project, degree...),
  each tagged with its CV section: the rubric weighs a skill seen in a job or project
  ("demonstrated") more than one seen only in a skills list ("listed").
- ``positions`` / ``total_years``: experience counted with the rules of KB-RR-003: overlapping
  positions counted once, internships at 50 % and at most one year in total, "2018 - 2020"
  counted from July 2018 to June 2020, "present" = today.
- ``is_grounded``: is an excerpt quoted by the LLM really in the CV? (KB-RR-001: never invent).
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from recruitment_ai.data.normalization import PRESENT, text_key
from recruitment_ai.matching.schemas import RequirementAssessment

_INTERNSHIP = re.compile(r"\b(intern|internship|stagiaire|stage|apprenti|apprentice|apprenticeship|alternance|"
                         r"alternant|trainee|work[- ]study)\b", re.IGNORECASE)
MAX_SNIPPET_CHARS = 1200
SKILLS_PER_SNIPPET = 30


@dataclass
class Position:
    title: str
    company: str
    start: str | None
    end: str | None
    months: set[int] = field(default_factory=set)
    internship: bool = False

    @property
    def years(self) -> float | None:
        return round(len(self.months) / 12, 1) if self.months else None


@dataclass
class Snippet:
    ref: str            # "experience:0", "project:2", "skills:0"...
    section: str        # experience | project | education | certification | languages | skills | summary
    label: str          # short human label, shown as the evidence source
    text: str           # what is embedded and shown to the reranker
    position: Position | None = None


def _month_index(value: str | None, is_end: bool, today: date) -> int | None:
    if not value:
        return None
    value = value.strip().lower()
    if value == PRESENT:
        return today.year * 12 + today.month - 1
    if m := re.fullmatch(r"(\d{4})-(\d{2})", value):
        return int(m[1]) * 12 + int(m[2]) - 1
    if m := re.fullmatch(r"(\d{4})", value):  # year only: July of the first year to June of the last
        return int(m[1]) * 12 + (5 if is_end else 6)
    return None


def positions(profile: dict[str, Any], today: date | None = None) -> list[Position]:
    today = today or date.today()
    out = []
    for job in profile.get("experience") or []:
        title, company = job.get("title") or "", job.get("company") or ""
        end_value = job.get("end_date") or (PRESENT if job.get("is_current") else None)
        start, end = _month_index(job.get("start_date"), False, today), _month_index(end_value, True, today)
        months = set(range(start, end + 1)) if start is not None and end is not None and end >= start else set()
        out.append(Position(title=title, company=company, start=job.get("start_date"), end=end_value, months=months,
                            internship=bool(_INTERNSHIP.search(title))))
    return out


def total_years(items: list[Position]) -> float | None:
    """Years of experience (KB-RR-003): overlaps once, internships 50 % and at most one year."""
    regular: set[int] = set().union(*(p.months for p in items if not p.internship))
    intern: set[int] = set().union(*(p.months for p in items if p.internship)) - regular
    if not regular and not intern:
        return None  # no readable dates: "duration unknown", never guessed
    return round((len(regular) + min(len(intern) * 0.5, 12)) / 12, 1)


def _clip(text: str, limit: int = MAX_SNIPPET_CHARS) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def candidate_snippets(profile: dict[str, Any], today: date | None = None) -> list[Snippet]:
    """A profile as citable evidence snippets."""
    snippets: list[Snippet] = []
    if profile.get("summary"):
        snippets.append(Snippet("summary:0", "summary", "Summary", _clip(f"Summary: {profile['summary']}")))
    for i, (job, pos) in enumerate(zip(profile.get("experience") or [], positions(profile, today), strict=False)):
        span = f"{pos.start or '?'} → {pos.end or '?'}" + (f", {pos.years} years" if pos.years is not None else "")
        label = f"{pos.title or 'Position'} at {pos.company or '?'} ({span}{', internship' if pos.internship else ''})"
        snippets.append(Snippet(f"experience:{i}", "experience", label,
                                _clip(f"Experience: {label}. {job.get('description') or ''}"), pos))
    for i, project in enumerate(profile.get("projects") or []):
        tech = ", ".join(project.get("technologies") or [])
        label = f"Project: {project.get('title') or '?'}"
        snippets.append(Snippet(f"project:{i}", "project", label,
                                _clip(f"{label}. {project.get('description') or ''}"
                                      + (f" Technologies: {tech}." if tech else ""))))
    for i, edu in enumerate(profile.get("education") or []):
        parts = [edu.get("degree"), edu.get("field_of_study")]
        label = " in ".join(p for p in parts if p) or "Education"
        label += f", {edu['institution']}" if edu.get("institution") else ""
        label += f" ({edu['end_date']})" if edu.get("end_date") else ""
        snippets.append(Snippet(f"education:{i}", "education", f"Education: {label}", f"Education: {label}"))
    for i, cert in enumerate(profile.get("certifications") or []):
        name = cert.get("name") if isinstance(cert, dict) else str(cert)
        if name:
            snippets.append(Snippet(f"certification:{i}", "certification", f"Certification: {name}",
                                    f"Certification: {name}"))
    skills = [s.get("name") if isinstance(s, dict) else str(s) for s in profile.get("skills") or []]
    skills = [s for s in skills if s]
    for i in range(0, len(skills), SKILLS_PER_SNIPPET):
        group = skills[i:i + SKILLS_PER_SNIPPET]
        snippets.append(Snippet(f"skills:{i // SKILLS_PER_SNIPPET}", "skills", "Skills list",
                                "Skills list: " + ", ".join(group)))
    languages = [f"{x.get('language')}" + (f" ({x['proficiency']})" if x.get("proficiency") else "")
                 for x in profile.get("languages") or [] if isinstance(x, dict) and x.get("language")]
    if languages:
        snippets.append(Snippet("languages:0", "languages", "Languages", "Languages: " + ", ".join(languages)))
    return snippets


def profile_text(profile: dict[str, Any]) -> str:
    """A readable CV-like text of a profile (used when the original CV text is not available)."""
    lines = [profile.get("full_name") or "Candidate"]
    if profile.get("location"):
        lines.append(profile["location"])
    for snippet in candidate_snippets(profile):
        lines.append(snippet.text)
    return "\n".join(lines)


def is_grounded(excerpt: str | None, source: str, threshold: float = 0.6) -> bool:
    """True when most words of the excerpt appear in the CV (a quote, or a light paraphrase of it)."""
    words = text_key(excerpt or "").split()
    if not words:
        return False
    source_key = text_key(source)
    if " ".join(words) in source_key:
        return True
    vocabulary = set(source_key.split())
    return sum(w in vocabulary for w in words) / len(words) >= threshold


def requirement_templates(job: dict[str, Any]) -> list[RequirementAssessment]:
    """The job's requirements as empty assessments (level "not_evidenced" until assessed)."""
    out = []
    for i, req in enumerate(job.get("requirements") or [], 1):
        text = req.get("requirement_text") or req.get("text") or ""
        if not text:
            continue
        kind = "must" if (req.get("requirement_type") or "required") == "required" else "nice"
        importance = req.get("importance") if req.get("importance") in ("high", "medium", "low") else "medium"
        out.append(RequirementAssessment(
            id=req.get("id") or f"{'M' if kind == 'must' else 'N'}{i}", text=text, kind=kind, importance=importance,
            min_years=req.get("min_years_experience"), hard=bool(req.get("hard"))))
    return out
