"""Unit tests for text cleaning and normalizer."""

from recruitment_ai.document_processing.cleaner import DocumentCleaner


def test_clean_text_normalization():
    cleaner = DocumentCleaner()
    dirty_text = "Senior   Developer \t\t\n\n\n\nwith  Python\u00a0skills.\r\n"
    cleaned = cleaner.clean_text(dirty_text)
    assert "Senior Developer" in cleaned
    assert "Python skills." in cleaned
    assert "\t" not in cleaned
    assert "\u00a0" not in cleaned


def test_extract_metadata_heuristics():
    cleaner = DocumentCleaner()
    sample = """
    EXPERIENCE
    Worked at TechCorp from 2020-2023.
    Contact: recruiter@example.com or phone +1-555-123-4567.

    SKILLS
    Python, SQL, Docker
    """
    metadata = cleaner.extract_metadata(sample)
    assert "recruiter@example.com" in metadata.detected_emails
    assert len(metadata.detected_phones) > 0
    assert "EXPERIENCE" in metadata.estimated_sections
    assert "SKILLS" in metadata.estimated_sections
    assert metadata.word_count > 10
