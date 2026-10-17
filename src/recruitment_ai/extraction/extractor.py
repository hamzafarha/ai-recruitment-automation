"""CV extraction (DSO1): CV document -> CandidateProfile, with a local LLM (Qwen2.5, Transformers).

Two paths, compared in the evaluation:
- ``extract_document``: parse the file to text (PyMuPDF / OCR / python-docx), then a text LLM
  structures it;
- ``extract_images``: a vision-language model (Qwen2.5-VL) reads the page images directly.

The LLM fills a compact schema (``CVExtraction``); code then normalizes dates, removes
duplicates and converts it to ``CandidateProfile``.
"""

import re
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from recruitment_ai.data.normalization import PRESENT, clean_value, normalize_date
from recruitment_ai.extraction.schemas import (
    CandidateProfile,
    Certification,
    Education,
    Experience,
    Language,
    Project,
    Skill,
)
from recruitment_ai.ingestion.parser import ParsedDocument, UnifiedDocumentParser
from recruitment_ai.llm.huggingface_adapter import HuggingFaceAdapter

# About 4,000 tokens of CV text: longer CVs are cut (on CPU, reading the prompt costs time too)
MAX_INPUT_CHARS = 16000
# Answer budget: a normal CV needs about 1,000 tokens; a model stuck repeating itself stops here
MAX_OUTPUT_TOKENS = 2048
# A CV too long for the GPU memory (seen on Kaggle's T4s) is retried once, cut to this length
OOM_RETRY_CHARS = 8000


# --------------------------------------------------------------------------- LLM-facing schema


class _Lenient(BaseModel):
    # The model's JSON is not schema-constrained: accept numbers where text is expected ("end_date": 2020)
    model_config = ConfigDict(coerce_numbers_to_str=True)


class ExtractedJob(_Lenient):
    title: str | None = Field(default=None, description="Job title as written")
    company: str | None = Field(default=None, description="Employer organisation as written (not a project "
                                                          "name), null if absent")
    location: str | None = None
    start_date: str | None = Field(default=None, description="Start date as written, e.g. 'Mar 2019' or '2019'")
    end_date: str | None = Field(default=None, description="End date as written, or 'Present' for a current job")
    description: str | None = Field(default=None, description="The role in one sentence, at most 20 words")


class ExtractedEducation(_Lenient):
    degree: str | None = None
    field_of_study: str | None = None
    institution: str | None = None
    start_date: str | None = None
    end_date: str | None = Field(default=None, description="End or graduation date as written")


class ExtractedProject(_Lenient):
    name: str
    description: str | None = Field(default=None, description="One sentence, at most 20 words")
    technologies: list[str] = Field(default_factory=list, description="Tools and technologies used")


class ExtractedLanguage(_Lenient):
    language: str
    proficiency: str | None = None


LIST_LIMITS = {"experience": 15, "education": 8, "projects": 10, "skills": 60, "certifications": 15,
               "languages": 8}


class CVExtraction(_Lenient):
    """What the LLM must return (kept small: every field costs generation time)."""

    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    location: str | None = Field(default=None, description="City / country of the candidate")
    summary: str | None = Field(default=None, description="The CV's own summary or objective, shortened")
    # max_length is shown to the model as "maxItems" in the JSON schema; longer lists are trimmed.
    # Order matters: the model writes the fields in this order, so the short lists come before
    # the long ones (a long skills list must not crowd out the certifications or languages).
    experience: list[ExtractedJob] = Field(default_factory=list, max_length=LIST_LIMITS["experience"])
    education: list[ExtractedEducation] = Field(default_factory=list, max_length=LIST_LIMITS["education"])
    certifications: list[str] = Field(default_factory=list, max_length=LIST_LIMITS["certifications"])
    languages: list[ExtractedLanguage] = Field(default_factory=list, max_length=LIST_LIMITS["languages"],
                                               description="Spoken languages")
    projects: list[ExtractedProject] = Field(default_factory=list, max_length=LIST_LIMITS["projects"],
                                             description="Personal, academic or team projects")
    skills: list[str] = Field(default_factory=list, max_length=LIST_LIMITS["skills"],
                              description="Every skill, tool and technology, as written")

    @field_validator(*LIST_LIMITS, mode="before")
    @classmethod
    def _lenient_list(cls, value: Any, info: ValidationInfo) -> Any:
        """null -> empty list, null items dropped, "English" accepted for {"language": "English"},
        and only the first N items kept (a small model can repeat items until the token limit)."""
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        items = [v for v in value if v is not None]
        if info.field_name == "languages":
            items = [{"language": v} if isinstance(v, str) else v for v in items]
        if info.field_name == "projects":
            items = [{"name": v} if isinstance(v, str) else v for v in items]
        return items[: LIST_LIMITS[info.field_name]]


SYSTEM_PROMPT = """You extract structured information from a candidate's CV for a recruitment team.
The CV may be in English or French; keep values in the language of the CV.

Rules:
1. Use only information written in the CV. Never invent, guess or complete a value.
2. If a field is absent, return null (or an empty list). Do not write "N/A" or "Unknown".
3. full_name, email, phone, location: the candidate's own details, usually in the first lines of the CV.
   location = city and region/country only, never the street address.
4. experience: one entry for EVERY position in the CV (CVs usually list 2 to 6), most recent first.
   Do not stop after the first one. Copy title, company and dates as written; a current position ends
   with "Present". description = one sentence of at most 20 words, not the bullet list.
   company = the employer organisation. In a heading like "Intern - Fraud Detection Platform" followed
   by "Acme Bank", the project name is not the company: company = "Acme Bank" (and location = the
   place only).
5. education: one entry per degree or diploma. degree = the diploma name ("Bachelor's", "MSc", "Master");
   field_of_study = the subject ("Computer Science"); institution = the school or university.
6. skills: short skill names of 1 to 4 words, never whole sentences. Take every technical skill, tool,
   technology and method mentioned ANYWHERE: the skills section, job descriptions, "Keywords" or
   "Technologies" lines, and projects; plus soft skills. Split lists: "Python, Docker and AWS" gives
   "Python", "Docker", "AWS". Group headings are not skills: in "Cloud & DevOps: Docker, AWS", keep
   "Docker" and "AWS", not "Cloud & DevOps". No duplicates.
7. projects: one entry per personal, academic or team project: its name, one sentence of at most 20
   words, and the technologies it lists.
8. certifications: certification names only. languages: spoken languages only (not programming languages).

Example (shortened):
CV:
  Jane Doe | Lyon, France | jane.doe@mail.com | +33 6 00 00 00 00
  Data Engineer - Customer Analytics Platform                    Jan 2021 - Present
  Acme Corp                                                      Lyon
  - Built Spark pipelines on AWS; strong experience with Airflow and dbt
  Keywords: Kafka, Terraform
  MSc in Computer Science, Université de Lyon, 2020
  Projects: Movie Recommender - collaborative filtering web app (PyTorch, FastAPI)
  Skills: Programming: Python, SQL | Soft skills: team leadership
  Certifications: AWS Certified Cloud Practitioner
  Languages: French (native), English (C1)
JSON:
  {"full_name": "Jane Doe", "email": "jane.doe@mail.com", "phone": "+33 6 00 00 00 00",
   "location": "Lyon, France", "summary": null,
   "experience": [{"title": "Data Engineer", "company": "Acme Corp", "location": "Lyon",
     "start_date": "Jan 2021", "end_date": "Present", "description": "Builds Spark data pipelines on AWS."}],
   "education": [{"degree": "MSc", "field_of_study": "Computer Science", "institution": "Université de Lyon",
     "start_date": null, "end_date": "2020"}],
   "certifications": ["AWS Certified Cloud Practitioner"],
   "languages": [{"language": "French", "proficiency": "native"}, {"language": "English", "proficiency": "C1"}],
   "projects": [{"name": "Movie Recommender", "description": "Collaborative filtering web app.",
     "technologies": ["PyTorch", "FastAPI"]}],
   "skills": ["Spark", "AWS", "Airflow", "dbt", "Kafka", "Terraform", "PyTorch", "FastAPI", "Python", "SQL",
     "Team leadership"]}

Return only the JSON object."""

USER_PROMPT = "Extract the information from this CV.\n\n=== CV ===\n{cv_text}\n=== END OF CV ==="
VISION_PROMPT = "Extract the information from this CV. The CV is the attached image (one image per page)."


def text_prompt(cv_text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """The user prompt for a CV text (cut at ``max_chars``). Shared with fine-tuning."""
    return USER_PROMPT.format(cv_text=cv_text.strip()[:max_chars])


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


def is_heading_only(skill: str, cv_text: str) -> bool:
    """True when the CV writes this "skill" only as a group heading ("Databases: MySQL, Redis").

    The model sometimes copies group headings into the skill list; a real skill also appears
    without a colon right after it.
    """
    pattern = re.compile(rf"(?<!\w){re.escape(skill)}(?!\w)(\s*:)?", re.IGNORECASE)
    found = [m.group(1) is not None for m in pattern.finditer(cv_text)]
    return bool(found) and all(found)


def to_profile(raw: CVExtraction, cv_id: str | None = None, normalizer: Any | None = None,
               cv_text: str | None = None) -> CandidateProfile:
    """Convert the LLM output to a CandidateProfile: clean values, normalize dates, split
    sentence-like skills, add the projects' technologies to the skills, drop group headings
    copied as skills (needs ``cv_text``), dedupe."""
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
    projects = [
        Project(title=name, description=clean_value(p.description),
                technologies=[t for t in map(clean_value, p.technologies) if t])
        for p in raw.projects if (name := clean_value(p.name))
    ]
    seen: set[str] = set()
    skills = []
    for name in expand_skills(raw.skills + [t for p in projects for t in p.technologies], normalizer):
        name = clean_value(name)
        if not name or name.lower() in seen or (cv_text and is_heading_only(name, cv_text)):
            continue
        seen.add(name.lower())
        skills.append(Skill(name=name))
    certifications = [Certification(name=n) for n in dict.fromkeys(filter(None, map(clean_value, raw.certifications)))]
    languages = [Language(language=x.language.strip(), proficiency=clean_value(x.proficiency))
                 for x in raw.languages if clean_value(x.language)]
    return CandidateProfile(
        id=cv_id, full_name=clean_value(raw.full_name), email=clean_value(raw.email),
        phone=clean_value(raw.phone), location=clean_value(raw.location), summary=clean_value(raw.summary),
        experience=experience, education=education, skills=skills, projects=projects,
        certifications=certifications, languages=languages,
    )


# --------------------------------------------------------------------------- extractor


class CVExtractor:
    """Extract a CandidateProfile from a CV file, its text, or its page images."""

    def __init__(self, llm: HuggingFaceAdapter | None = None, parser: UnifiedDocumentParser | None = None,
                 vision_llm: HuggingFaceAdapter | None = None, normalizer: Any | None = None) -> None:
        self.llm = llm or HuggingFaceAdapter()
        self.parser = parser or UnifiedDocumentParser()
        self._vision_llm = vision_llm
        self.normalizer = normalizer if normalizer is not None else default_normalizer()

    @property
    def vision_llm(self) -> HuggingFaceAdapter:
        if self._vision_llm is None:
            self._vision_llm = HuggingFaceAdapter(model=self.llm.settings.multimodal_model, vision=True)
        return self._vision_llm

    def extract_text(self, cv_text: str, cv_id: str | None = None) -> tuple[CandidateProfile, dict[str, Any]]:
        text = cv_text.strip()
        try:
            return self._extract_text(text, cv_id, MAX_INPUT_CHARS)
        except Exception as exc:  # torch is not imported here: recognise the error by its name
            if type(exc).__name__ != "OutOfMemoryError":
                raise
        # Out of GPU memory (a very long CV): free it, outside the except block so the failed
        # attempt's tensors can be released, then retry once with a shorter text
        self.free_memory()
        profile, stats = self._extract_text(text, cv_id, OOM_RETRY_CHARS)
        return profile, {**stats, "retried_after_out_of_memory": True}

    def _extract_text(self, text: str, cv_id: str | None, max_chars: int) -> tuple[CandidateProfile, dict[str, Any]]:
        raw = self.llm.generate_structured(text_prompt(text, max_chars), CVExtraction, system_prompt=SYSTEM_PROMPT,
                                           max_tokens=MAX_OUTPUT_TOKENS)
        stats = {**self.llm.last_stats, "input_chars": len(text), "input_truncated": len(text) > max_chars}
        return to_profile(raw, cv_id, self.normalizer, cv_text=text), stats

    def free_memory(self) -> None:
        """Release the GPU memory cached by the models (after an out-of-memory error)."""
        for llm in (self.llm, self._vision_llm):
            if llm is not None and hasattr(llm, "free_memory"):
                llm.free_memory()

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
                                                  images=images, max_tokens=MAX_OUTPUT_TOKENS)
        return to_profile(raw, cv_id, self.normalizer), dict(self.vision_llm.last_stats)


def pdf_page_images(path: Path, max_side: int = 1400) -> list[bytes]:
    """PNG bytes of each PDF page, longest side about ``max_side`` px (the adapter then caps the
    pixel count at VLM_MAX_PIXELS)."""
    import pymupdf

    images = []
    with pymupdf.open(str(path)) as doc:
        for page in doc:
            zoom = max_side / max(page.rect.width, page.rect.height)
            images.append(page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png"))
    return images
