"""Shared normalization helpers for dataset preparation (dates, junk values, text keys)."""

import hashlib
import re

PRESENT = "present"

_PRESENT_WORDS = {
    "present", "current", "currently", "now", "today", "to date", "till date", "ongoing",
    "aujourd'hui", "aujourd hui", "présent", "actuel", "actuellement", "en cours",
}

_MONTHS = {
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
    # French
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "décembre": 12, "decembre": 12, "janv": 1, "févr": 2, "fevr": 2, "juil": 7,
}

# Section headings that the Resume Parser labeller sometimes stored as values
# (found by the groundedness check in notebooks/00_data_inspection.ipynb).
JUNK_VALUES = {
    "certification", "certifications", "professional certification", "license", "licenses",
    "copyright notice", "declaration", "personal details", "education details",
    "skills", "experience", "education", "summary", "accomplishments", "highlights",
}


def _valid(year: int, month: int | None = None) -> bool:
    return 1950 <= year <= 2035 and (month is None or 1 <= month <= 12)


def normalize_date(raw: str | None) -> str | None:
    """Normalize a CV date to 'YYYY-MM', 'YYYY', 'present', or None if unreadable.

    Handles '07/2018', '2018-07', 'August 2014', 'Aug. 2014', 'juin 2021',
    '07/15/2018' (US order), '2018', 'Present', 'Current', 'aujourd'hui'.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text.strip(" .") in _PRESENT_WORDS:
        return PRESENT

    if m := re.fullmatch(r"(\d{4})[-/.](\d{1,2})", text):
        year, month = int(m[1]), int(m[2])
        return f"{year}-{month:02d}" if _valid(year, month) else None
    if m := re.fullmatch(r"(\d{1,2})[-/.](\d{4})", text):
        month, year = int(m[1]), int(m[2])
        return f"{year}-{month:02d}" if _valid(year, month) else None
    if m := re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", text):
        first, second, year = int(m[1]), int(m[2]), int(m[3])
        month = first if first <= 12 else second  # US order first, else DD/MM
        return f"{year}-{month:02d}" if _valid(year, month) else None

    for m in re.finditer(r"([a-zéèûô]+)\.?\s*,?\s*(\d{4})", text):
        month = _MONTHS.get(m[1])
        if month and _valid(int(m[2]), month):
            return f"{int(m[2])}-{month:02d}"

    if m := re.search(r"\b(19[5-9]\d|20[0-3]\d)\b", text):
        return m[1]
    # e.g. "to present", "since now" (word boundaries: "unknown" must not match "now")
    if re.search(r"\b(present|current|now|ongoing|aujourd'hui|en cours)\b", text):
        return PRESENT
    return None


def clean_value(value: str | None) -> str | None:
    """Strip a label value; return None for empty strings and known junk headings."""
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower().strip(" :.") in JUNK_VALUES:
        return None
    return value


def text_key(text: str) -> str:
    """Lowercase alphanumeric form of a text, used for groundedness and duplicate checks."""
    return re.sub(r"[^a-z0-9+#]+", " ", text.lower()).strip()


def text_hash(text: str) -> str:
    """Stable hash of a text's normalized form (detects the same CV under different ids)."""
    return hashlib.md5(text_key(text).encode("utf-8")).hexdigest()
