"""Job description extraction: a free-text job ad -> structured requirements (M1.. must-haves, N1.. nice-to-haves).

The requirements are what the matching scores against (rubric KB-RR-002): each one is a
must-have or a nice-to-have, with an importance (high/medium/low) and, when the ad states it, a
minimum number of years. Uses the base Qwen model (the fine-tuned adapter is for CVs).
"""

import re
from typing import Any

from pydantic import Field, field_validator

from recruitment_ai.extraction.extractor import _Lenient

MAX_JOB_CHARS = 12000
KIND_ALIASES = {"must": "must", "must-have": "must", "must_have": "must", "required": "must", "mandatory": "must",
                "nice": "nice", "nice-to-have": "nice", "nice_to_have": "nice", "preferred": "nice",
                "optional": "nice", "plus": "nice", "bonus": "nice"}


class ExtractedRequirement(_Lenient):
    text: str
    kind: str = Field(default="must", description="must | nice")
    importance: str = Field(default="medium", description="high | medium | low")
    min_years: float | None = Field(default=None, description="Only when the ad states a number of years")

    @field_validator("min_years", mode="before")
    @classmethod
    def _years(cls, value: Any) -> Any:
        if isinstance(value, str):
            match = re.search(r"\d+(\.\d+)?", value)
            return float(match.group(0)) if match else None
        return value


class JobExtraction(_Lenient):
    title: str | None = None
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None
    requirements: list[ExtractedRequirement] = Field(default_factory=list, max_length=16)

    @field_validator("requirements", mode="before")
    @classmethod
    def _lenient(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, list):
            return [{"text": v} if isinstance(v, str) else v for v in value if v][:16]
        return value


JOB_SYSTEM_PROMPT = """You turn a job advertisement into the structured requirements a recruitment team scores candidates against.

Rules:
1. title, company, location (city/country), employment_type: as written; null if absent.
2. requirements: 5 to 14 items, each ONE checkable skill, knowledge, experience or qualification, at most 20 words. Split combined items ("Python and Docker" gives two requirements only if they are different skills that matter separately).
3. kind: "must" for what the ad requires (required, must, mandatory, X+ years); "nice" for what it presents as a plus (preferred, bonus, nice to have, ideally).
4. importance: "high" for the core skills of the role (named in the title or first, or repeated); "medium" for important but secondary ones; "low" for minor ones (soft skills, degree when experience can replace it, tools of minor importance).
5. min_years: only when the ad gives a number of years for that requirement, else null.
6. Use only the ad: never invent requirements. Write requirements in English, even for a French ad.

Return only the JSON object."""


def extract_job(text: str, llm: Any) -> dict[str, Any]:
    """Structured job (title, company, ..., requirements with ids) from a job advertisement."""
    ad = text.strip()[:MAX_JOB_CHARS]
    raw = llm.generate_structured(f"JOB ADVERTISEMENT:\n{ad}", JobExtraction, system_prompt=JOB_SYSTEM_PROMPT,
                                  max_tokens=1200, use_adapter=False)
    requirements, counters = [], {"must": 0, "nice": 0}
    for r in raw.requirements:
        if not r.text.strip():
            continue
        kind = KIND_ALIASES.get(str(r.kind).strip().lower(), "must")
        counters[kind] += 1
        importance = str(r.importance).strip().lower()
        requirements.append({
            "id": f"{'M' if kind == 'must' else 'N'}{counters[kind]}",
            "requirement_text": r.text.strip(),
            "requirement_type": "required" if kind == "must" else "preferred",
            "importance": importance if importance in ("high", "medium", "low") else "medium",
            "min_years_experience": r.min_years if r.min_years and r.min_years > 0 else None,
        })
    first_line = next((line.strip() for line in ad.splitlines() if line.strip()), "Untitled job")
    return {"title": (raw.title or first_line[:120]).strip(), "company": raw.company, "location": raw.location,
            "employment_type": raw.employment_type, "description": ad, "requirements": requirements}
