"""CV extraction (DSO1): CV document -> CandidateProfile, with a local LLM (Qwen2.5 via Ollama).

Two paths, compared in the evaluation:
- ``extract_document``: parse the file to text (PyMuPDF / OCR / python-docx), then a text LLM
  structures it;
- ``extract_images``: a vision-language model (Qwen2.5-VL) reads the page images directly.

The LLM fills a compact schema (``CVExtraction``); code then normalizes dates, removes
duplicates and converts it to ``CandidateProfile``.
"""

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from recruitment_ai.data.normalization import PRESENT, clean_value, normalize_date
from recruitment_ai.extraction.schemas import (
    CandidateProfile,
    Certification,
    Education,
    Experience,
    Language,
    Skill,
)
from recruitment_ai.ingestion.parser import ParsedDocument, UnifiedDocumentParser
from recruitment_ai.llm.ollama_adapter import OllamaAdapter

# Prompt budget: with num_ctx 8192, keep the CV text well below the window (output needs room too)
MAX_INPUT_CHARS = 16000


# --------------------------------------------------------------------------- LLM-facing schema


class ExtractedJob(BaseModel):
    title: str | None = Field(default=None, description="Job title as written")
    company: str | None = Field(default=None, description="Employer name as written, null if absent")
    location: str | None = None
    start_date: str | None = Field(default=None, description="Start date as written, e.g. 'Mar 2019' or '2019'")
    end_date: str | None = Field(default=None, description="End date as written, or 'Present' for a current job")
    description: str | None = Field(default=None, description="One or two sentences summarising the role")


class ExtractedEducation(BaseModel):
    degree: str | None = None
    field_of_study: str | None = None
    institution: str | None = None
    start_date: str | None = None
    end_date: str | None = Field(default=None, description="End or graduation date as written")


class ExtractedLanguage(BaseModel):
    language: str
    proficiency: str | None = None


class CVExtraction(BaseModel):
    """What the LLM must return (kept small: every field costs generation time)."""

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = Field(default=None, description="City / country of the candidate")
    summary: str | None = Field(default=None, description="The CV's own summary or objective, shortened")
    # max_length becomes "maxItems" in the JSON schema, which Ollama enforces while generating:
    # a small model stuck repeating list items is cut off at once instead of running for minutes
    experience: list[ExtractedJob] = Field(default_factory=list, max_length=15)
    education: list[ExtractedEducation] = Field(default_factory=list, max_length=8)
    skills: list[str] = Field(default_factory=list, max_length=60,
                              description="Every skill, tool and technology, as written")
    certifications: list[str] = Field(default_factory=list, max_length=15)
    languages: list[ExtractedLanguage] = Field(default_factory=list, max_length=8, description="Spoken languages")


SYSTEM_PROMPT = """You extract structured information from a candidate's CV for a recruitment team.
The CV may be in English or French; keep values in the language of the CV.

Rules:
1. Use only information written in the CV. Never invent, guess or complete a value.
2. If a field is absent, return null (or an empty list). Do not write "N/A" or "Unknown".
3. full_name, email, phone, location: the candidate's own details, usually in the first lines of the CV.
   location = city and region/country only, never the street address.
4. experience: one entry for EVERY position in the CV (CVs usually list 2 to 6), most recent first.
   Do not stop after the first one. Copy title, company and dates as written; a current position ends
   with "Present". description = one or two short sentences, not the full bullet list.
5. education: one entry per degree or diploma. degree = the diploma name ("Bachelor's", "MSc", "Master");
   field_of_study = the subject ("Computer Science"); institution = the school or university.
6. skills: short skill names of 1 to 4 words, never whole sentences. Take every technical skill, tool,
   technology and method mentioned anywhere (skills section, job descriptions, projects), plus soft skills.
   Split lists: "Python, Docker and AWS" gives "Python", "Docker", "AWS". No duplicates.
7. certifications: certification names only. languages: spoken languages only (not programming languages).

Example (shortened):
CV:
  Jane Doe | Lyon, France | jane.doe@mail.com | +33 6 00 00 00 00
  Data Engineer - Acme Corp (Jan 2021 - Present)
  - Built Spark pipelines on AWS; strong experience with Airflow and dbt
  MSc in Computer Science, Université de Lyon, 2020
  Skills: Python, SQL, team leadership
JSON:
  {"full_name": "Jane Doe", "email": "jane.doe@mail.com", "phone": "+33 6 00 00 00 00",
   "location": "Lyon, France", "summary": null,
   "experience": [{"title": "Data Engineer", "company": "Acme Corp", "location": null,
     "start_date": "Jan 2021", "end_date": "Present", "description": "Builds Spark data pipelines on AWS."}],
   "education": [{"degree": "MSc", "field_of_study": "Computer Science", "institution": "Université de Lyon",
     "start_date": null, "end_date": "2020"}],
   "skills": ["Spark", "AWS", "Airflow", "dbt", "Python", "SQL", "Team leadership"],
   "certifications": [], "languages": []}

Return only the JSON object."""

USER_PROMPT = "Extract the information from this CV.\n\n=== CV ===\n{cv_text}\n=== END OF CV ==="
VISION_PROMPT = "Extract the information from this CV. The CV is the attached image (one image per page)."


# --------------------------------------------------------------------------- skill phrases

MAX_SKILL_WORDS = 4


def split_skill_phrase(phrase: str, normalizer: Any) -> list[str]:
    """Known skills inside a sentence-like "skill", using the EN/FR skill lexicon.

    Small LLMs often copy a CV's skill bullets verbatim ("Knowledge modern DevOps services
    (Kubernetes services, Prometheus, etc.)"). Every 1- to 4-word window is looked up, longest
    first; single lowercase words are ignored (too generic: "design", "services"), since tech
    names are almost always capitalised. Returns the matched words as written in the phrase.
    """
    words = [w.strip(".,;:()[]{}\"'") for w in phrase.split()]
    words = [w for w in words if w]
    found: list[str] = []
    used = [False] * len(words)
    for size in range(MAX_SKILL_WORDS, 0, -1):
        for start in range(len(words) - size + 1):
            if any(used[start:start + size]):
                continue
            candidate = " ".join(words[start:start + size])
            if size == 1 and not (candidate[:1].isupper() or any(c in candidate for c in "+#./0123456789")):
                continue
            if len(candidate) >= 2 and normalizer.normalize(candidate) is not None:
                found.append(candidate)
                used[start:start + size] = [True] * size
    order = {w: phrase.find(w) for w in found}
    return sorted(found, key=lambda w: order[w])


def expand_skills(skills: list[str], normalizer: Any | None) -> list[str]:
    """Replace sentence-like skills by the known skills they contain (kept as-is without a lexicon)."""
    if normalizer is None:
        return skills
    out: list[str] = []
    for skill in skills:
        if len(skill.split()) > MAX_SKILL_WORDS:
            out += split_skill_phrase(skill, normalizer)
        else:
            out.append(skill)
    return out


def default_normalizer(path: Path = Path("data/interim/skills/skill_lexicon.json")) -> Any | None:
    """The skill lexicon built by scripts/prepare_data.py, or None if it is not built yet."""
    if not path.exists():
        return None
    from recruitment_ai.data.skill_lexicon import SkillNormalizer

    return SkillNormalizer.from_file(path)


# --------------------------------------------------------------------------- conversion


def _date(value: str | None) -> str | None:
    return normalize_date(value) if value else None


def to_profile(raw: CVExtraction, cv_id: str | None = None, normalizer: Any | None = None) -> CandidateProfile:
    """Convert the LLM output to a CandidateProfile: clean values, normalize dates, split
    sentence-like skills, dedupe."""
    experience = []
    for job in raw.experience:
        title, company = clean_value(job.title), clean_value(job.company)
        if not title and not company:
            continue
        end = _date(job.end_date)
        experience.append(Experience(
            title=title, company=company, location=clean_value(job.location),
            start_date=_date(job.start_date), end_date=end, is_current=end == PRESENT,
            description=clean_value(job.description),
        ))
    education = [
        Education(degree=clean_value(e.degree), field_of_study=clean_value(e.field_of_study),
                  institution=clean_value(e.institution), start_date=_date(e.start_date),
                  end_date=_date(e.end_date))
        for e in raw.education if clean_value(e.degree) or clean_value(e.institution)
    ]
    seen: set[str] = set()
    skills = []
    for name in expand_skills(raw.skills, normalizer):
        name = clean_value(name)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            skills.append(Skill(name=name))
    certifications = [Certification(name=n) for n in dict.fromkeys(filter(None, map(clean_value, raw.certifications)))]
    languages = [Language(language=x.language.strip(), proficiency=clean_value(x.proficiency))
                 for x in raw.languages if clean_value(x.language)]
    return CandidateProfile(
        id=cv_id, full_name=clean_value(raw.full_name), email=clean_value(raw.email),
        phone=clean_value(raw.phone), location=clean_value(raw.location), summary=clean_value(raw.summary),
        experience=experience, education=education, skills=skills,
        certifications=certifications, languages=languages,
    )


# --------------------------------------------------------------------------- extractor


class CVExtractor:
    """Extract a CandidateProfile from a CV file, its text, or its page images."""

    def __init__(self, llm: OllamaAdapter | None = None, parser: UnifiedDocumentParser | None = None,
                 vision_llm: OllamaAdapter | None = None, normalizer: Any | None = None) -> None:
        self.llm = llm or OllamaAdapter()
        self.parser = parser or UnifiedDocumentParser()
        self._vision_llm = vision_llm
        self.normalizer = normalizer if normalizer is not None else default_normalizer()

    @property
    def vision_llm(self) -> OllamaAdapter:
        if self._vision_llm is None:
            self._vision_llm = OllamaAdapter(model=self.llm.settings.ollama_vision_model)
        return self._vision_llm

    def extract_text(self, cv_text: str, cv_id: str | None = None) -> tuple[CandidateProfile, dict[str, Any]]:
        text = cv_text.strip()
        truncated = len(text) > MAX_INPUT_CHARS
        raw = self.llm.generate_structured(USER_PROMPT.format(cv_text=text[:MAX_INPUT_CHARS]), CVExtraction,
                                           system_prompt=SYSTEM_PROMPT)
        stats = {**self.llm.last_stats, "input_chars": len(text), "input_truncated": truncated}
        return to_profile(raw, cv_id, self.normalizer), stats

    def extract_document(self, path: Path, cv_id: str | None = None) -> tuple[CandidateProfile, dict[str, Any]]:
        """Parse a CV file (OCR when needed) then structure its text."""
        start = time.perf_counter()
        parsed: ParsedDocument = self.parser.parse(Path(path))
        parse_seconds = round(time.perf_counter() - start, 2)
        profile, stats = self.extract_text(parsed.raw_text, cv_id)
        return profile, {**stats, "parse_seconds": parse_seconds, "input_type": parsed.metadata.get("input_type"),
                         "parser": parsed.metadata.get("engine")}

    def extract_images(self, images: list[bytes], cv_id: str | None = None) -> tuple[CandidateProfile, dict[str, Any]]:
        """Vision path: the VLM reads PNG page images directly (no OCR)."""
        raw = self.vision_llm.generate_structured(VISION_PROMPT, CVExtraction, system_prompt=SYSTEM_PROMPT,
                                                  images=images)
        return to_profile(raw, cv_id, self.normalizer), dict(self.vision_llm.last_stats)


def pdf_page_images(path: Path, max_side: int = 1400) -> list[bytes]:
    """PNG bytes of each PDF page, longest side about ``max_side`` px (a VLM's sweet spot)."""
    import pymupdf

    images = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            zoom = max_side / max(page.rect.width, page.rect.height)
            images.append(page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png"))
    return images
