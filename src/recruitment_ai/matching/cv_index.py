"""One candidate's CV as searchable evidence: the screening agent's "CV Search Tool" (diagram part 5).

The CV text is cut into short passages (one per bullet or short block), each tagged with the CV
section it comes from, found from the section headings ("Experience", "Skills", "Projets",
"Compétences"...). A passage is what the analysis may quote word for word, and its section tells a
skill used in a position or project (it can be "demonstrated") from one only named in a skills
section or a summary ("listed"), as the fit scoring rubric KB-RR-002 requires.
Without CV text, the structured profile's snippets (one per position, project, degree...) are used.

Search: BGE-M3 cosine similarity over every passage, then the BGE cross-encoder on the best
``POOL`` passages of each query (all queries of a round in one embedding and one reranking request).
"""

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from recruitment_ai.matching.evidence import candidate_snippets, is_grounded

POOL = 8                 # passages given to the cross-encoder for each query
MAX_PASSAGE_CHARS = 320  # a passage is short enough to be quoted
USAGE_SECTIONS = frozenset({"experience", "project", "achievement"})  # where a skill is used for real
LISTING_SECTIONS = frozenset({"skills", "summary"})                   # where a skill is only named

SECTION_LABELS = {"experience": "Experience", "project": "Project", "achievement": "Achievement",
                  "skills": "Skills", "education": "Education", "certification": "Certification",
                  "languages": "Languages", "summary": "Summary", "other": "Other", "header": "CV header"}
_HEADINGS = {
    "experience": ["experience", "experiences", "professional experience", "professional experiences",
                   "work experience", "work history", "employment", "employment history", "career history",
                   "professional background", "relevant experience", "internships", "internship",
                   "experience professionnelle", "experiences professionnelles", "parcours professionnel",
                   "stages", "stage"],
    "project": ["projects", "project", "key projects", "academic projects", "personal projects", "projets",
                "projet", "projets academiques", "projets personnels", "projets realises", "realisations"],
    "achievement": ["accomplishments", "achievements", "key achievements", "awards and achievements"],
    "skills": ["skills", "skill", "technical skills", "key skills", "core skills", "it skills", "computer skills",
               "skills and tools", "tools and technologies", "technologies", "tools", "tech stack", "stack",
               "competences", "competences techniques", "competences cles", "outils", "outils et technologies",
               "highlights", "skill highlights", "core qualifications", "qualifications", "areas of expertise",
               "expertise", "competencies", "core competencies", "hard skills", "soft skills", "strengths",
               "summary of skills", "technical proficiencies"],
    "education": ["education", "education and training", "academic background", "academic qualification",
                  "academic qualifications", "educational qualification", "educational qualifications",
                  "formation", "formations", "diplomes", "etudes", "cursus", "parcours academique"],
    "certification": ["certifications", "certification", "certificates", "certificats", "licenses",
                      "licenses and certifications", "training", "trainings", "courses"],
    "languages": ["languages", "language", "language skills", "langues"],
    "summary": ["summary", "professional summary", "executive summary", "profile", "professional profile",
                "personal profile", "career profile", "profil", "about me", "a propos", "objective",
                "career objective", "objectif", "resume"],
    "other": ["interests", "hobbies", "centres d interet", "loisirs", "references", "activities",
              "extracurricular activities", "volunteering", "volunteer experience", "vie associative",
              "additional information", "personal details", "personal information", "informations personnelles",
              "declaration", "affiliations", "professional affiliations", "publications", "contact"],
}
HEADINGS = {name: section for section, names in _HEADINGS.items() for name in names}
BULLET_CHARS = frozenset("•-*▪◦●‣⁃–—►✓✔➢➤○■□·")
BULLET = re.compile(r"^(?:[•\-*▪◦●‣⁃–—►✓✔➢➤○■□·]|\d{1,2}[.)])\s+")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_SENTENCE_END = re.compile(r"(?<=[.;!?])\s+")


def _norm(text: str) -> str:
    """Lowercase, no accents, letters only: "Expérience Professionnelle :" -> "experience professionnelle"."""
    text = unicodedata.normalize("NFKD", text.lower().replace("&", " and "))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]+", " ", text).strip()


def section_of_heading(line: str) -> str | None:
    """The section a heading line opens ("SKILLS", "Expérience professionnelle :"), None for other lines."""
    if len(line) > 50:
        return None
    return HEADINGS.get(_norm(line))


@dataclass
class Passage:
    section: str     # experience | project | achievement | skills | education | certification | languages |
                     # summary | other | header (the top of the CV, before any heading)
    text: str        # what is embedded, reranked and may be quoted
    context: str = ""  # the position or project line above it (experience and project sections)

    @property
    def label(self) -> str:
        name = SECTION_LABELS.get(self.section, self.section.title())
        return f"{name} — {self.context[:80]}" if self.context else name


def _pack(text: str, limit: int = MAX_PASSAGE_CHARS) -> list[str]:
    """A long block cut at sentence ends into pieces of at most ``limit`` characters (when possible)."""
    if len(text) <= limit:
        return [text]
    pieces, current = [], ""
    for sentence in _SENTENCE_END.split(text):
        if current and len(current) + len(sentence) + 1 > limit:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def cv_passages(cv_text: str) -> list[Passage]:
    """The CV text as passages tagged with their section; a bullet starts a new passage."""
    out: list[Passage] = []
    section, context = "header", ""
    lines: list[str] = []
    bullet_block = pending_bullet = False

    def flush(block_section: str | None = None) -> None:
        nonlocal context
        text = " ".join(lines).strip()
        lines.clear()
        if not text:
            return
        where = block_section or section
        if block_section is None and where in ("experience", "project") and not bullet_block and len(text) <= 160:
            context = text  # a position or project header: "Backend Developer — Acme, 2020 - 2024"
            out.append(Passage(where, text))
            return
        out.extend(Passage(where, piece, context if where == section else "") for piece in _pack(text))

    for raw in (cv_text or "").splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            flush()
            continue
        if heading := section_of_heading(line.rstrip(":").strip()):
            flush()
            section, context = heading, ""
            continue
        head, colon, rest = line.partition(":")
        if colon and rest.strip() and (inline := section_of_heading(head.strip())):
            # "Languages: English" is one languages line; inside a position or a project, "Technologies: ..." is
            # the stack it used and stays part of it
            flush()
            lines.append(rest.strip())
            flush(section if section in USAGE_SECTIONS and inline == "skills" else inline)
            continue
        if line in BULLET_CHARS:  # a bullet alone on its line (PDF text): its text is on the next line
            flush()
            pending_bullet = True
            continue
        if match := BULLET.match(line):
            flush()
            bullet_block, line = True, line[match.end():].strip()
        elif pending_bullet:
            bullet_block = True
        elif not lines:
            bullet_block = False
        elif bullet_block and section in ("experience", "project") and YEAR.search(line):
            flush()  # after a position's bullets, the next position's header (it has dates)
            bullet_block = False
        pending_bullet = False
        if lines and len(" ".join(lines)) + len(line) + 1 > MAX_PASSAGE_CHARS:
            flush()
        if line:
            lines.append(line)
    flush()
    return out


def profile_passages(profile: dict[str, Any]) -> list[Passage]:
    """Without CV text: the profile's snippets (positions, projects, degrees, skills...) as passages."""
    return [Passage(s.section, s.text) for s in candidate_snippets(profile)]


@dataclass
class EvidenceHit:
    section: str
    label: str
    text: str
    score: float      # cross-encoder relevance (0..1), or cosine similarity without a reranker
    query: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"section": self.section, "label": self.label, "text": self.text, "score": round(self.score, 4),
                "query": self.query}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class CVEvidenceIndex:
    """Search one candidate's CV (embedded once, in memory: a CV is a few dozen passages)."""

    def __init__(self, embedder: Any, reranker: Any | None = None, cv_text: str | None = None,
                 profile: dict[str, Any] | None = None) -> None:
        self.embedder, self.reranker = embedder, reranker
        self.from_cv_text = bool((cv_text or "").strip())
        self.passages = cv_passages(cv_text or "") if self.from_cv_text else profile_passages(profile or {})
        self._vectors: list[list[float]] | None = None
        self.reranked: bool | None = None  # known after the first search: cross-encoder scores, or cosine

    def _passage_vectors(self) -> list[list[float]]:
        if self._vectors is None:
            self._vectors = self.embedder.embed_documents([p.text for p in self.passages]) if self.passages else []
        return self._vectors

    def search_many(self, queries: list[str], top_k: int = 3) -> list[list[EvidenceHit]]:
        """The best ``top_k`` passages for each query, best first."""
        if not queries:
            return []
        if not self.passages:
            return [[] for _ in queries]
        vectors = self._passage_vectors()
        query_vectors = self.embedder.embed_documents(queries)
        pools, pairs = [], []
        for query, query_vector in zip(queries, query_vectors, strict=True):
            cosines = [_cosine(query_vector, v) for v in vectors]
            order = sorted(range(len(vectors)), key=lambda i, c=cosines: c[i], reverse=True)[:POOL]
            pools.append((order, cosines))
            pairs += [[query, self.passages[i].text] for i in order]
        scores = self.reranker.score_pairs(pairs) if self.reranker is not None and pairs else None
        self.reranked = scores is not None
        out, k = [], 0
        for query, (order, cosines) in zip(queries, pools, strict=True):
            hits = []
            for i in order:
                p = self.passages[i]
                hits.append(EvidenceHit(p.section, p.label, p.text, float(scores[k] if scores is not None
                                                                          else cosines[i]), query))
                k += 1
            out.append(sorted(hits, key=lambda h: h.score, reverse=True)[:top_k])
        return out

    def search(self, query: str, top_k: int = 3) -> list[EvidenceHit]:
        return self.search_many([query], top_k)[0]

    def relevance(self, pairs: list[tuple[str, str]]) -> list[float] | None:
        """Cross-encoder relevance of (requirement, excerpt) pairs; None without a cross-encoder."""
        if not pairs or self.reranker is None:
            return None
        return self.reranker.score_pairs([[a, b] for a, b in pairs])

    def sections_of(self, excerpt: str | None) -> set[str]:
        """The CV sections whose passages contain the excerpt (a quote or a light paraphrase of it)."""
        if not excerpt:
            return set()
        return {p.section for p in self.passages if is_grounded(excerpt, p.text)}
