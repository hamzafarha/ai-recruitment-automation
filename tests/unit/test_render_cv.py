"""Tests for CV rendering and the synthetic part of the evaluation set."""

import json
import re
import unicodedata
from pathlib import Path

import pytest

from recruitment_ai.evaluation.render_cv import (
    LAYOUTS,
    display_date,
    render_docx,
    render_pdf,
)
from recruitment_ai.extraction.schemas import CandidateProfile

SYNTHETIC = Path(__file__).resolve().parents[2] / "data" / "processed" / "eval_v1" / "synthetic"
PROFILES = sorted((SYNTHETIC / "profiles").glob("*.json"))


@pytest.mark.parametrize(
    ("value", "lang", "expected"),
    [
        ("2021-03", "en", "Mar 2021"),
        ("2021-03", "fr", "mars 2021"),
        ("2022-09", "fr", "sept. 2022"),
        ("present", "en", "Present"),
        ("present", "fr", "aujourd'hui"),
        ("2016", "en", "2016"),
        (None, "en", ""),
    ],
)
def test_display_date(value, lang, expected):
    assert display_date(value, lang) == expected


def _load(path: Path) -> tuple[dict, CandidateProfile]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["meta"], CandidateProfile.model_validate(data["profile"])


def _pdf_text(path: Path) -> str:
    import pymupdf

    with pymupdf.open(path) as doc:
        return " ".join(page.get_text() for page in doc)


def _docx_text(path: Path) -> str:
    from docx import Document

    return " ".join(p.text for p in Document(str(path)).paragraphs)


def _facts(profile: CandidateProfile, lang: str) -> list[str]:
    """Every value of the answer key that a reader must be able to find in the document."""
    facts = [profile.full_name, profile.email, profile.phone]
    for x in profile.experience:
        facts += [x.title, x.company, display_date(x.start_date, lang), display_date(x.end_date, lang)]
    for ed in profile.education:
        facts += [ed.degree, ed.institution]
    facts += [s.name for s in profile.skills] + [c.name for c in profile.certifications]
    facts += [lang_.language for lang_ in profile.languages]
    return [f for f in facts if f]


def _normalize(text: str) -> str:
    """Comparable form of extracted text.

    - NFKC: PDFs draw "fi"/"fl" as ligature glyphs, extracted as "ﬁ"/"ﬂ" ("Certiﬁed");
      real CVs do the same, so the Step 2 extraction pipeline must apply NFKC too.
    - Whitespace-insensitive, and a line wrapped after a hyphen ("Retrieval-\\nAugmented") is re-joined.
    """
    text = " ".join(unicodedata.normalize("NFKC", text).split())
    return re.sub(r"-\s+", "-", text)


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.stem)
def test_every_answer_key_fact_is_in_the_rendered_documents(path: Path, tmp_path: Path):
    meta, profile = _load(path)
    for fmt in meta["formats"]:
        out = tmp_path / f"cv.{fmt}"
        if fmt == "pdf":
            render_pdf(profile, out, lang=meta["language"], layout=meta["layout"])
            text = _normalize(_pdf_text(out))
        else:
            render_docx(profile, out, lang=meta["language"])
            text = _normalize(_docx_text(out))
        missing = [f for f in _facts(profile, meta["language"]) if _normalize(f) not in text]
        assert not missing, f"{path.stem}.{fmt} does not show: {missing}"


def test_synthetic_set_composition():
    metas = [_load(p)[0] for p in PROFILES]
    assert len(metas) >= 10
    assert [m["id"] for m in metas] == [p.stem for p in PROFILES]  # file name = id
    assert sum(m["language"] == "fr" for m in metas) >= len(metas) / 4  # French is well represented
    assert {m["layout"] for m in metas} == set(LAYOUTS)  # every layout is used
    assert sum("docx" in m["formats"] for m in metas) >= 4
    for m in metas:
        assert m["case"] and "grade" not in m["case"].lower()  # no expected grade: labelers must not be anchored


def test_render_pdf_rejects_unknown_layout(tmp_path: Path):
    _, profile = _load(PROFILES[0])
    with pytest.raises(ValueError):
        render_pdf(profile, tmp_path / "x.pdf", layout="fancy")
