"""Unit tests for document ingestion parser."""

from pathlib import Path

from recruitment_ai.ingestion.parser import UnifiedDocumentParser


def test_unified_document_parser_text_file(tmp_path: Path):
    sample_file = tmp_path / "sample_resume.txt"
    sample_file.write_text(
        "John Doe\nSoftware Engineer\nPython, FastAPI", encoding="utf-8"
    )

    parser = UnifiedDocumentParser()
    doc = parser.parse(sample_file)

    assert doc.filename == "sample_resume.txt"
    assert doc.file_type == "txt"
    assert "John Doe" in doc.raw_text
    assert doc.total_pages == 1
