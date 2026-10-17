"""Tests for Step 2 (CV extraction): parsing, OCR line rebuilding, conversion, scoring.

No GPU, OCR model or LLM needed: those are replaced by small fakes.
"""

import io
import json
import time
from pathlib import Path

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


# --------------------------------------------------------------------------- Transformers adapter


class FakeTokenizer:
    """Stands in for a Transformers tokenizer / processor: records what it receives."""

    def __init__(self, answer: str) -> None:
        self.answer, self.messages, self.images = answer, None, None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        return "<prompt>"

    def __call__(self, text=None, images=None, return_tensors=None):
        import torch
        from transformers import BatchEncoding

        self.images = images
        return BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})

    def decode(self, ids, skip_special_tokens=True):
        return self.answer


class FakeModel:
    """Stands in for Qwen: answers two new tokens after the prompt's three."""

    def generate(self, input_ids, max_new_tokens, **kwargs):
        import torch

        self.max_new_tokens = max_new_tokens
        return torch.cat([input_ids, torch.tensor([[7, 8]])], dim=1)


def fake_llm(answer: str, model: str = "Qwen/Qwen2.5-3B-Instruct"):
    from recruitment_ai.llm.huggingface_adapter import HuggingFaceAdapter

    llm = HuggingFaceAdapter(model=model)
    llm.settings = llm.settings.model_copy(update={"llm_device": "cpu"})  # whatever .env says
    llm._processor, llm._model = FakeTokenizer(answer), FakeModel()
    return llm


def test_hf_adapter_asks_for_the_schema_and_validates_the_answer():
    llm = fake_llm('Here it is:\n```json\n{"full_name": "Jane Doe", "skills": ["Python"]}\n```')
    result = llm.generate_structured("Extract", CVExtraction, system_prompt="rules", max_tokens=500)
    system, user = llm._processor.messages
    assert system == {"role": "system", "content": "rules"}
    assert user["content"].startswith("Extract") and '"maxItems":15' in user["content"]
    assert result.full_name == "Jane Doe" and result.skills == ["Python"]
    assert llm._model.max_new_tokens == 500
    assert llm.last_stats["prompt_tokens"] == 3 and llm.last_stats["output_tokens"] == 2
    assert not llm.last_stats["truncated_output"]


def test_hf_adapter_sends_page_images_to_vision_models():
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2000, 1000), "white").save(buffer, format="PNG")
    llm = fake_llm('{"full_name": "Jane Doe"}', model="Qwen/Qwen2.5-VL-3B-Instruct")
    assert llm.vision
    llm.generate_structured("Read", CVExtraction, system_prompt="rules", images=[buffer.getvalue()] * 2)
    user = llm._processor.messages[-1]
    assert [part["type"] for part in user["content"]] == ["image", "image", "text"]
    assert len(llm._processor.images) == 2
    assert all(img.width * img.height <= llm.settings.vlm_max_pixels for img in llm._processor.images)


@pytest.mark.parametrize("answer", ["", "   "])
def test_hf_adapter_rejects_empty_answers(answer):
    with pytest.raises(ValueError):
        fake_llm(answer).generate_structured("x", CVExtraction)


def test_parse_json_object_repairs_an_answer_cut_by_the_token_limit():
    from recruitment_ai.llm.huggingface_adapter import parse_json_object

    cut = '{"full_name": "Jane", "skills": ["Python", "Docker", "Kube'
    assert parse_json_object(cut) == {"full_name": "Jane", "skills": ["Python", "Docker"]}
    cut = '{"full_name": "Jane", "experience": [{"title": "Dev", "company": "A"}, {"title": "Le'
    assert parse_json_object(cut) == {"full_name": "Jane", "experience": [{"title": "Dev", "company": "A"}]}
    assert parse_json_object('Sure! {"a": 1} Hope it helps {"b": 2}') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("no JSON here")


def test_cv_extraction_accepts_loose_json():
    raw = CVExtraction.model_validate({
        "full_name": "Jane", "phone": 33600000000, "education": None,
        "skills": ["Python", None] + [f"S{i}" for i in range(80)],
        "languages": ["English", {"language": "French", "proficiency": "B2"}],
        "experience": [{"title": "Dev", "start_date": 2019, "end_date": "Present"}],
    })
    assert raw.phone == "33600000000" and raw.education == []
    assert len(raw.skills) == 60 and raw.skills[:2] == ["Python", "S0"]
    assert [x.language for x in raw.languages] == ["English", "French"]
    assert raw.experience[0].start_date == "2019"


# --------------------------------------------------------------------------- evaluation runs


def test_spread_groups_keeps_the_mix_in_any_prefix():
    from recruitment_ai.evaluation.extraction_run import spread_groups

    docs = ([{"doc_id": f"s{i}", "input_type": "scanned_pdf"} for i in range(8)]
            + [{"doc_id": f"d{i}", "input_type": "docx"} for i in range(2)])
    order = [d["doc_id"] for d in spread_groups(docs)]
    assert sorted(order) == sorted(d["doc_id"] for d in docs)
    assert "d0" in order[:5] and "d1" in order[5:]


def test_run_extraction_starts_no_document_after_the_deadline(tmp_path):
    from recruitment_ai.evaluation.extraction_run import partial_note, run_extraction

    gold = {"experience": [], "education": [], "skills": []}
    docs = [{"doc_id": d, "cv_id": d, "path": Path(f"{d}.pdf"), "gold": gold, "score_identity": False,
             "input_type": "text_pdf"} for d in ("a", "b")]
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "a.json").write_text(json.dumps({"prediction": {}, "stats": {}, "error": None}),
                                                     encoding="utf-8")

    class NoNewDocument:
        normalizer = FakeNormalizer()

        def extract_document(self, *args):
            raise AssertionError("no document may start after the deadline")

    rows = run_extraction(docs, "ocr_llm", tmp_path, NoNewDocument(), deadline=time.monotonic() - 1,
                          log=lambda *_: None)
    assert [r["doc"]["doc_id"] for r in rows] == ["a"]  # the saved prediction is still used
    assert partial_note(2, 2) == [] and "1 of 2" in partial_note(1, 2)[0]


def test_out_of_memory_cv_is_retried_once_with_a_shorter_text():
    from recruitment_ai.extraction.extractor import OOM_RETRY_CHARS, CVExtractor

    class OutOfMemoryError(RuntimeError):  # same name as torch's CUDA error
        pass

    class LongCVLLM:
        def __init__(self) -> None:
            self.prompts, self.freed, self.last_stats = [], 0, {}

        def generate_structured(self, prompt, schema, system_prompt=None, max_tokens=None):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                raise OutOfMemoryError("CUDA out of memory")
            return CVExtraction(full_name="Ann Lee")

        def free_memory(self):
            self.freed += 1

    llm = LongCVLLM()
    profile, stats = CVExtractor(llm=llm, normalizer=object()).extract_text("Ann Lee " + "x" * 20000)
    assert profile.full_name == "Ann Lee" and stats["retried_after_out_of_memory"]
    assert llm.freed == 1 and len(llm.prompts[1]) < len(llm.prompts[0])
    assert len(llm.prompts[1]) < OOM_RETRY_CHARS + 200  # cut to the retry length (+ the prompt's words)


def test_projects_are_kept_and_their_technologies_become_skills():
    raw = CVExtraction.model_validate({
        "skills": ["Python", "Machine Learning & AI", "Databases", "MySQL"],
        "projects": [{"name": "QoSentry", "description": "Network anomaly detection.",
                      "technologies": ["SHAP", "LangGraph", "Python"]}, "Image Captioning"],
    })
    cv_text = ("Keywords: Python\nMachine Learning & AI: Scikit-learn, PyTorch\nDatabases: MySQL, Redis\n"
               "QoSentry (SHAP, LangGraph)")
    profile = to_profile(raw, cv_text=cv_text)
    assert [p.title for p in profile.projects] == ["QoSentry", "Image Captioning"]
    assert profile.projects[0].technologies == ["SHAP", "LangGraph", "Python"]
    # group headings dropped (they only appear before a colon), project technologies added, no duplicates
    assert [s.name for s in profile.skills] == ["Python", "MySQL", "SHAP", "LangGraph"]


def test_projects_accept_null_technologies():
    raw = CVExtraction.model_validate({"projects": [{"name": "A", "technologies": None},
                                                    {"name": "B", "technologies": ["Python", None]}]})
    assert [p.technologies for p in raw.projects] == [[], ["Python"]]


def test_heading_detection_keeps_real_skills():
    from recruitment_ai.extraction.extractor import is_heading_only

    text = "Python: 5 years\nSkills: Python, Docker\nCloud: AWS"
    assert not is_heading_only("Python", text)  # also written without a colon
    assert is_heading_only("Cloud", text) and not is_heading_only("Kubernetes", text)


def test_json_roundtrip_of_profile():
    profile = to_profile(CVExtraction(full_name="A", skills=["Python"]))
    assert json.loads(profile.model_dump_json())["skills"][0]["name"] == "Python"
