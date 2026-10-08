"""Text cleaning, normalization, and structural layout extraction."""

import re
import unicodedata

from pydantic import BaseModel, Field


class ExtractedMetadata(BaseModel):
    """Metadata extracted during text cleaning."""

    detected_emails: list[str] = Field(default_factory=list)
    detected_phones: list[str] = Field(default_factory=list)
    word_count: int = 0
    character_count: int = 0
    estimated_sections: list[str] = Field(default_factory=list)


class DocumentCleaner:
    """Cleans and normalizes extracted resume and job description text."""

    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    PHONE_REGEX = re.compile(
        r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
    )
    KNOWN_SECTIONS = [
        "EXPERIENCE",
        "WORK EXPERIENCE",
        "PROFESSIONAL EXPERIENCE",
        "EDUCATION",
        "SKILLS",
        "TECHNICAL SKILLS",
        "PROJECTS",
        "CERTIFICATIONS",
        "LANGUAGES",
        "SUMMARY",
        "PROFILE",
    ]

    def clean_text(self, text: str) -> str:
        """Normalize unicode, strip non-printable characters, and collapse excessive whitespace."""
        if not text:
            return ""

        # Normalize unicode characters (e.g. smart quotes, accented chars)
        normalized = unicodedata.normalize("NFKD", text)

        # Replace non-breaking spaces and tabs with standard spaces
        normalized = normalized.replace("\u00a0", " ").replace("\t", " ")

        # Remove control characters except newlines
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", normalized)

        # Normalize multiple spaces per line
        lines = [re.sub(r" {2,}", " ", line.strip()) for line in cleaned.splitlines()]

        # Collapse more than two consecutive empty lines
        result_lines: list[str] = []
        consecutive_empty = 0
        for line in lines:
            if not line:
                consecutive_empty += 1
                if consecutive_empty <= 2:
                    result_lines.append("")
            else:
                consecutive_empty = 0
                result_lines.append(line)

        return "\n".join(result_lines).strip()

    def extract_metadata(self, text: str) -> ExtractedMetadata:
        """Extract basic contact heuristics and section headers from text."""
        cleaned = self.clean_text(text)
        emails = list(set(self.EMAIL_REGEX.findall(cleaned)))
        raw_phones = list(set(self.PHONE_REGEX.findall(cleaned)))
        phones = [p.strip() for p in raw_phones if len(re.sub(r"\D", "", p)) >= 7]

        # Detect resume sections
        detected_sections: list[str] = []
        for line in cleaned.splitlines():
            line_upper = line.strip().upper()
            if (
                line_upper in self.KNOWN_SECTIONS
                and line_upper not in detected_sections
            ):
                detected_sections.append(line_upper)

        words = cleaned.split()
        return ExtractedMetadata(
            detected_emails=emails,
            detected_phones=phones,
            word_count=len(words),
            character_count=len(cleaned),
            estimated_sections=detected_sections,
        )
