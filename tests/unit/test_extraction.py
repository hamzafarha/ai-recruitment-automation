"""Tests for Step 2 (CV extraction): parsing, OCR line rebuilding, conversion, scoring.

No GPU, OCR model or LLM needed: those are replaced by small fakes.
"""

import json
from pathlib import Path

import httpx
import pytest

from recruitment_ai.evaluation.extraction_eval import (
    score_entries,
    score_identity,
    score_profile,
    score_skills,
    similarity,
    summarize,
)
from recruitment_ai.extraction.extractor import (
    CVExtraction,
    ExtractedJob,
    expand_skills,
    split_skill_phrase,
    to_profile,
)
from recruitment_ai.ingestion.ocr import OCRResult, merge_lines
from recruitment_ai.ingestion.parser import (
    PdfParser,
    UnifiedDocumentParser,
    normalize_text,
)

SYNTHETIC = Path(__file__).resolve().parents[2] / "data" / "processed" / "eval_v1" / "synthetic"


class FakeOCR:
    """Stands in for RapidOCR: records calls and returns a fixed text."""

    def __init__(self) -> None:
        self.calls = 0

    def image_to_text(self, image) -> OCRResult:
        self.calls += 1
        return OCRResult(text="Jane Doe\nData Engineer  2021 - Present", line_count=2, mean_confidence=0.97)


class FakeNormalizer:
    """Tiny lexicon: knows a handful of skills, case-insensitively."""

    KNOWN = {"python 3", "python", "devops", "kubernetes", "prometheus", "big data", "jenkins", "react"}

    def normalize(self, raw: str):
        key = raw.strip().lower()
        if key in {"reactjs", "react.js"}:
            key = "react"
        return type("Concept", (), {"id": key})() if key in self.KNOWN else None


# --------------------------------------------------------------------------- parsing


def test_normalize_text_unfolds_ligatures_and_keeps_accents():
    assert normalize_text("Certiﬁed in Snowﬂake") == "Certified in Snowflake"
    assert normalize_text("Ingénieure à Montréal") == "Ingénieure à Montréal"


def test_text_pdf_uses_the_text_layer_without_ocr():
    ocr = FakeOCR()
    doc = PdfParser(ocr).parse(SYNTHETIC / "files" / "syn01.pdf")
    assert ocr.calls == 0
    assert doc.metadata["input_type"] == "text_pdf" and "Priya Raman" in doc.raw_text
    assert "Certified" in doc.raw_text  # ligature unfolded (the PDF draws "fi" as one glyph)


def test_scanned_pdf_page_goes_to_ocr(tmp_path: Path):
    import pymupdf

    path = tmp_path / "scan.pdf"
    with pymupdf.open() as doc:  # one page with an image only: what a scanned CV looks like
        page = doc.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 50, 50), False)
        pix.clear_with(200)
        page.insert_image(page.rect, pixmap=pix)
        doc.save(path)
    ocr = FakeOCR()
    parsed = PdfParser(ocr).parse(path)
    assert ocr.calls == 1
    assert parsed.metadata["input_type"] == "scanned_pdf" and "Jane Doe" in parsed.raw_text


def test_docx_is_parsed():
    doc = UnifiedDocumentParser(FakeOCR()).parse(SYNTHETIC / "files" / "syn10.docx")
    assert doc.metadata["input_type"] == "docx" and "Karim Haddad" in doc.raw_text


def test_merge_lines_joins_boxes_on_the_same_visual_line():
    box = lambda x, y, text: ([[x, y], [x + 100, y], [x + 100, y + 20], [x, y + 20]], text, 0.99)  # noqa: E731
    lines = merge_lines([box(400, 101, "2019 - Present"), box(10, 100, "Data Engineer"), box(10, 140, "Acme")])
    assert lines == ["Data Engineer  2019 - Present", "Acme"]


# --------------------------------------------------------------------------- conversion


def test_to_profile_normalizes_dates_and_dedupes_skills():
    raw = CVExtraction(
        full_name="Jane Doe", email="N/A",
        experience=[ExtractedJob(title="Data Engineer", company="Acme", start_date="Jan 2021", end_date="Present"),
                    ExtractedJob(title=None, company=None)],
        skills=["Python", "python", "Docker"],
    )
    profile = to_profile(raw, "cv1")
    job = profile.experience[0]
    assert (job.start_date, job.end_date, job.is_current) == ("2021-01", "present", True)
    assert len(profile.experience) == 1  # the empty job is dropped
    assert [s.name for s in profile.skills] == ["Python", "Docker"]
    assert profile.id == "cv1"


def test_sentence_like_skills_are_split_with_the_lexicon():
    phrase = "Knowledge modern DevOps services (Kubernetes services, Prometheus, etc.)"
    assert split_skill_phrase(phrase, FakeNormalizer()) == ["DevOps", "Kubernetes", "Prometheus"]
    skills = expand_skills(["Python 3", phrase, "Interest is Big Data and willing to collaborate"], FakeNormalizer())
    assert skills == ["Python 3", "DevOps", "Kubernetes", "Prometheus", "Big Data"]
    assert expand_skills([phrase], None) == [phrase]  # without a lexicon nothing changes


def test_list_limits_are_in_the_json_schema():
    props = CVExtraction.model_json_schema()["properties"]
    assert props["experience"]["maxItems"] == 15 and props["skills"]["maxItems"] == 60


# --------------------------------------------------------------------------- scoring


def test_similarity():
    assert similarity("Senior Python Developer", "senior python developer!") == 1.0
    assert similarity("San Francisco, CA", "1735 Veta Mountains, San Francisco, CA") == 0.9
    assert similarity(None, "x") == 0.0


def test_identity_scoring():
    gold = {"full_name": "Lindsay Beer", "email": None, "phone": "+1 (555) 753 8128", "location": "San Francisco, CA"}
    pred = {"full_name": "Lindsay Beer", "email": "made.up@x.com", "phone": "555-753-8128", "location": None}
    assert score_identity(gold, pred) == {"correct": 2, "wrong": 0, "missed": 1, "hallucinated": 1, "gold_fields": 3}


def test_experience_pairing_and_dates():
    gold = [{"title": "Backend Developer", "company": "Infosys", "start_date": "2018-01", "end_date": "present"},
            {"title": "Python Developer", "company": "BDO", "start_date": "2016-04", "end_date": "2018-01"}]
    pred = [{"title": "Backend Web Developer", "company": "Infosys Ltd", "start_date": "2018", "end_date": "present"}]
    counts = score_entries(gold, pred, "experience")
    assert counts["gold"] == 2 and counts["pred"] == 1 and counts["matched"] == 1
    assert counts["dates"] == 2 and counts["dates_exact"] == 1 and counts["dates_year"] == 2


def test_skills_use_lexicon_concepts():
    gold = [{"name": "React", "category": "Tool"}, {"name": "Teamwork", "category": "Soft"}]
    pred = [{"name": "ReactJS"}, {"name": "Cooking"}]
    counts = score_skills(gold, pred, FakeNormalizer())
    assert counts["gold_found"] == 1 and counts["pred_correct"] == 1
    assert counts["gold_technical"] == 1 and counts["technical_found"] == 1


def test_summary_micro_averages():
    gold = {"experience": [{"title": "Dev", "company": "A"}], "education": [], "skills": [{"name": "Python"}]}
    perfect = score_profile(gold, gold, FakeNormalizer(), score_identity_fields=False)
    empty = score_profile(gold, {}, FakeNormalizer(), score_identity_fields=False)
    summary = summarize([perfect, empty])
    assert summary["experience"]["recall"] == 0.5 and summary["experience"]["precision"] == 1.0
    assert summary["identity"]["cvs"] == 0 and summary["identity"]["accuracy"] is None


# --------------------------------------------------------------------------- Ollama adapter


def test_ollama_adapter_sends_context_window_schema_and_images(monkeypatch):
    from recruitment_ai.llm.ollama_adapter import OllamaAdapter

    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent.update(url=url, payload=json)
        body = {"message": {"content": '{"full_name": "Jane Doe"}'}, "prompt_eval_count": 100, "eval_count": 20}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    llm = OllamaAdapter(model="qwen2.5:3b")
    result = llm.generate_structured("Extract", CVExtraction, system_prompt="rules", images=[b"png-bytes"])
    payload = sent["payload"]
    assert sent["url"].endswith("/api/chat") and payload["model"] == "qwen2.5:3b"
    assert payload["options"]["num_ctx"] == llm.settings.ollama_num_ctx and payload["options"]["num_predict"] == 3000
    assert payload["format"]["properties"]["experience"]["maxItems"] == 15
    assert payload["messages"][1]["images"] and result.full_name == "Jane Doe"
    assert llm.last_stats["output_tokens"] == 20 and not llm.last_stats["truncated_prompt"]


@pytest.mark.parametrize("content", ["", "   "])
def test_ollama_adapter_rejects_empty_answers(monkeypatch, content):
    from recruitment_ai.llm.ollama_adapter import OllamaAdapter

    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: httpx.Response(
        200, json={"message": {"content": content}}, request=httpx.Request("POST", url)))
    with pytest.raises(ValueError):
        OllamaAdapter().generate_structured("x", CVExtraction)


def test_json_roundtrip_of_profile():
    profile = to_profile(CVExtraction(full_name="A", skills=["Python"]))
    assert json.loads(profile.model_dump_json())["skills"][0]["name"] == "Python"
