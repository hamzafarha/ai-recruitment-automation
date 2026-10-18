"""Test doubles for the application layer: tiny deterministic models, no GPU, download or network.

- ``WordEmbedder``: bag-of-words vectors (texts sharing words are close), like BGE-M3 in spirit;
- ``OverlapReranker``: share of the query's words found in the document, like the cross-encoder;
- ``ScriptedLLM``: Qwen replaced by prepared answers, in order;
- ``FakeExtractor``: CV extraction returning a given profile;
- ``FakeContainer``: the application's container with those fakes, a real SearchIndex on an
  embedded in-memory Qdrant and the real matching code.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from recruitment_ai.config.settings import Settings
from recruitment_ai.extraction.schemas import CandidateProfile
from recruitment_ai.llm.base import LLMResponse
from recruitment_ai.reranking.reranker import RerankResult

WORD = re.compile(r"[a-z0-9+#]+")
STOP = {"and", "or", "the", "of", "in", "with", "a", "an", "to", "for", "on", "at", "is", "skills", "list"}


def words(text: str | None) -> list[str]:
    return WORD.findall((text or "").lower())


def overlap(query: str, document: str) -> float:
    asked = {w for w in words(query) if w not in STOP}
    return len(asked & set(words(document))) / len(asked) if asked else 0.0


class WordEmbedder:
    DIM = 256

    def require_model(self) -> None:
        pass

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIM
        for word in words(text):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM] += 1.0
        vector[0] += 1e-3  # never a zero vector
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_text(self, text: str) -> list[float]:
        return self._vector(text)


class OverlapReranker:
    def score_pairs(self, pairs: list[list[str]]) -> list[float]:
        return [overlap(q, d) for q, d in pairs]

    def rerank(self, query: str, documents: list[Any], top_n: int = 3) -> list[RerankResult]:
        ranked = sorted(documents, key=lambda d: overlap(query, d.text), reverse=True)[:top_n]
        return [RerankResult(id=d.id, text=d.text, relevance_score=overlap(query, d.text), metadata=d.metadata)
                for d in ranked]


class ScriptedLLM:
    """``structured``: answers of generate_structured, in order (a dict, an exception, or a function of the
    prompt), or {schema name: answers in order} when several kinds of calls interleave (the screening
    agent's plans and analyses). ``text``: the answer of generate (free text)."""

    model = "fake-qwen"

    def __init__(self, structured: list[Any] | dict[str, list[Any]] | None = None,
                 text: Any = "Answer from the tool results.") -> None:
        self.structured = ({k: list(v) for k, v in structured.items()} if isinstance(structured, dict)
                           else list(structured or []))
        self.text = text
        self.calls: list[dict[str, Any]] = []
        self.last_stats: dict[str, Any] = {}

    def _next(self, schema: type[BaseModel]) -> Any:
        queue = self.structured.get(schema.__name__, []) if isinstance(self.structured, dict) else self.structured
        return queue.pop(0) if queue else {}

    def generate_structured(self, prompt: str, schema: type[BaseModel], system_prompt: str | None = None,
                            images: list[bytes] | None = None, max_tokens: int = 2048,
                            use_adapter: bool = True) -> BaseModel:
        self.calls.append({"kind": "structured", "schema": schema.__name__, "prompt": prompt,
                           "system": system_prompt, "use_adapter": use_adapter})
        answer: Any = self._next(schema)
        if isinstance(answer, Exception):
            raise answer
        if callable(answer):
            answer = answer(prompt)
        result = schema.model_validate(answer)
        # like the real adapters: what the model wrote, and its token counts
        self.last_stats = {"model": self.model, "seconds": 0.1, "prompt_tokens": len(prompt) // 4,
                           "output_tokens": 20, "raw_answer": json.dumps(answer, ensure_ascii=False)}
        return result

    def generate(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 1024, use_adapter: bool = True) -> LLMResponse:
        self.calls.append({"kind": "text", "prompt": prompt, "system": system_prompt, "use_adapter": use_adapter})
        if isinstance(self.text, Exception):
            raise self.text
        self.last_stats = {"model": self.model, "seconds": 0.1, "prompt_tokens": len(prompt) // 4,
                           "output_tokens": len(self.text) // 4}
        return LLMResponse(content=self.text, model=self.model)


ANN = {
    "full_name": "Ann Lee", "email": "ann.lee@example.com", "location": "Tunis",
    "summary": "Backend developer building Python APIs.",
    "experience": [
        {"title": "Backend Developer", "company": "Acme", "start_date": "2020-01", "end_date": "2024-12",
         "description": "Built REST APIs in Python with FastAPI and PostgreSQL; Docker deployments."},
        {"title": "Software Intern", "company": "Beta", "start_date": "2019-02", "end_date": "2019-07",
         "description": "Wrote Python scripts."},
    ],
    "education": [{"degree": "Engineering degree", "field_of_study": "Computer Science", "institution": "ESPRIT",
                   "end_date": "2019"}],
    "skills": [{"name": "Python"}, {"name": "FastAPI"}, {"name": "PostgreSQL"}, {"name": "Docker"},
               {"name": "Kubernetes"}],
    "languages": [{"language": "English", "proficiency": "C1"}],
}
BOB = {
    "full_name": "Bob Stone", "location": "Paris",
    "experience": [{"title": "Graphic Designer", "company": "Studio", "start_date": "2021-01", "end_date": "present",
                    "description": "Designed brand identities with Photoshop and Illustrator."}],
    "skills": [{"name": "Photoshop"}, {"name": "Illustrator"}],
}
JOB = {
    "title": "Python Backend Developer", "company": "Acme", "location": "Tunis", "role": "backend_developer",
    "description": "We build APIs.\n\nYou will design REST services.",
    "requirements": [
        {"requirement_text": "Python programming", "requirement_type": "required", "importance": "high",
         "min_years_experience": 3},
        {"requirement_text": "REST APIs with FastAPI", "requirement_type": "required", "importance": "high"},
        {"requirement_text": "PostgreSQL databases", "requirement_type": "required", "importance": "medium"},
        {"requirement_text": "Kubernetes clusters", "requirement_type": "preferred", "importance": "low"},
    ],
}


DESIGN_JOB = {
    "title": "Graphic Designer", "company": "Studio", "location": "Paris", "description": "Brand design.",
    "requirements": [
        {"requirement_text": "Photoshop and Illustrator", "requirement_type": "required", "importance": "high"},
        {"requirement_text": "Brand identity projects", "requirement_type": "required", "importance": "medium"},
    ],
}
# the CVs as their text (what the document processing step reads), with section headings
ANN_CV = """Ann Lee
Backend developer, Tunis

Experience
Backend Developer — Acme (2020 - 2024)
• Built REST APIs in Python with FastAPI and PostgreSQL
• Docker deployments

Skills
Python, FastAPI, PostgreSQL, Docker, Kubernetes
"""
BOB_CV = """Bob Stone
Paris

Experience
Graphic Designer — Studio (2021 - present)
• Designed brand identities with Photoshop and Illustrator

Skills
Photoshop, Illustrator
"""


class FakeParser:
    """Reads the uploaded bytes as the CV text: one page, from the text layer."""

    def parse(self, path: Any) -> Any:
        from recruitment_ai.ingestion.parser import ParsedDocument, ParsedPage

        text = Path(path).read_bytes().decode("utf-8", "ignore")
        return ParsedDocument(filename=Path(path).name, file_type="pdf", total_pages=1, raw_text=text,
                              pages=[ParsedPage(page_number=1, text=text,
                                                layout_info={"method": "text_layer", "ocr_confidence": None})],
                              metadata={"engine": "PyMuPDF", "input_type": "text_pdf"})


# what the base model (without the fine-tuned adapter) extracts from Ann's CV in the fakes: less
BASE_ANN = {**ANN, "experience": ANN["experience"][:1], "skills": [{"name": "Python"}, {"name": "FastAPI"},
                                                                  {"name": "Teamwork"}]}


class FakeExtractor:
    """``extract_file`` returns the next prepared profile (default: Ann Lee) and the file's text;
    ``extract_text`` (the application pipeline's extraction step) returns the next of ``text_profiles``
    (a profile or an exception), else Ann's profile, smaller without the adapter (``use_adapter=False``)."""

    def __init__(self, profiles: list[dict[str, Any]] | None = None,
                 text_profiles: list[Any] | None = None) -> None:
        self.profiles = list(profiles or [])
        self.text_profiles = list(text_profiles or [])
        self.seen: list[tuple[Any, bool]] = []
        self.parser = FakeParser()
        self.adapter_calls: list[bool] = []

    def extract_text(self, cv_text: str, cv_id: str | None = None,
                     use_adapter: bool = True) -> tuple[CandidateProfile, dict[str, Any]]:
        self.adapter_calls.append(use_adapter)
        data = ANN if use_adapter else BASE_ANN
        if use_adapter and self.text_profiles:
            data = self.text_profiles.pop(0)
            if isinstance(data, Exception):
                raise data
        stats = {"model": "fake-qwen", "prompt_tokens": 900, "output_tokens": 300,
                 "raw_answer": json.dumps({"full_name": data["full_name"], "skills": [s["name"] for s in data["skills"]]})}
        return CandidateProfile.model_validate(data), stats

    def extract_file(self, path: Any, cv_id: str | None = None) -> tuple[CandidateProfile, dict[str, Any], str]:
        self.seen.append((path, path.exists()))
        data = self.profiles.pop(0) if self.profiles else ANN
        if isinstance(data, Exception):
            raise data
        text = path.read_bytes().decode("utf-8", "ignore")
        stats = {"input_type": "text_pdf", "model": "fake-qwen", "adapter": "lora", "seconds": 0.1,
                 "raw_answer": '{"full_name": "..."}'}
        return CandidateProfile.model_validate(data), stats, text


class FakeContainer:
    """The application's wiring with fakes for the models; search, matching and analysis are the real code."""

    def __init__(self, llm: ScriptedLLM | None = None, extractor: FakeExtractor | None = None) -> None:
        from recruitment_ai.matching.analyzer import EvidenceAnalyzer
        from recruitment_ai.matching.semantic import SemanticMatcher
        from recruitment_ai.rag.search_index import SearchIndex
        from recruitment_ai.vectorstore.qdrant import QdrantVectorStore

        self.settings = Settings(QDRANT_LOCATION=":memory:", LLM_PROVIDER="mock", EMBEDDING_PROVIDER="local",
                                 N8N_EVENTS_ENABLED="false")
        self.embedder, self.reranker, self.normalizer = WordEmbedder(), OverlapReranker(), None
        self.store = QdrantVectorStore(self.settings)
        self.index = SearchIndex(self.store, self.embedder, self.reranker, WordEmbedder.DIM)
        self.extractor = extractor or FakeExtractor()
        self.semantic = SemanticMatcher(self.embedder, self.reranker, None)
        self.llm = llm or ScriptedLLM()
        self.analyzer = EvidenceAnalyzer(self.llm, knowledge_search=self.knowledge_search, semantic=self.semantic)

    def knowledge_search(self, query: str, role: str | None = None) -> list[RerankResult]:
        return self.index.search_knowledge(query, role=role, top_k=3)
