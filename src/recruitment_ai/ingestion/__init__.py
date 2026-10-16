"""Document ingestion package."""

from recruitment_ai.ingestion.ocr import OCREngine, OCRResult
from recruitment_ai.ingestion.parser import (
    DocumentParser,
    DocxParser,
    ImageOCRParser,
    ParsedDocument,
    PdfParser,
    UnifiedDocumentParser,
)

__all__ = [
    "ParsedDocument",
    "DocumentParser",
    "PdfParser",
    "DocxParser",
    "ImageOCRParser",
    "OCREngine",
    "OCRResult",
    "UnifiedDocumentParser",
]
