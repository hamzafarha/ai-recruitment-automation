"""Document ingestion and parsing abstractions."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from recruitment_ai.utils.logging import get_logger

logger = get_logger(__name__)


class ParsedPage(BaseModel):
    """Extracted content from an individual document page."""

    page_number: int
    text: str
    layout_info: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Complete parsed document output."""

    filename: str
    file_type: str
    total_pages: int
    raw_text: str
    pages: list[ParsedPage] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentParser(ABC):
    """Abstract interface for file parsers."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse given document file and return structured text."""
        pass


class PdfParser(DocumentParser):
    """PDF parser using PyMuPDF (fitz)."""

    def parse(self, file_path: Path) -> ParsedDocument:
        path = Path(file_path)
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            pages: list[ParsedPage] = []
            full_text_chunks: list[str] = []

            for page_idx, page in enumerate(doc):
                text = page.get_text("text") or ""
                pages.append(
                    ParsedPage(
                        page_number=page_idx + 1,
                        text=text,
                        layout_info={"rect": list(page.rect)},
                    )
                )
                full_text_chunks.append(text)

            total_pages = len(doc)
            doc.close()

            return ParsedDocument(
                filename=path.name,
                file_type="pdf",
                total_pages=total_pages,
                raw_text="\n\n".join(full_text_chunks),
                pages=pages,
                metadata={"engine": "PyMuPDF"},
            )
        except ImportError:
            logger.warning(
                "PyMuPDF (fitz) is not installed. Using raw text fallback parser."
            )
            text = path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(
                filename=path.name,
                file_type="pdf",
                total_pages=1,
                raw_text=text,
                pages=[ParsedPage(page_number=1, text=text)],
                metadata={"engine": "fallback"},
            )


class DocxParser(DocumentParser):
    """DOCX parser using python-docx."""

    def parse(self, file_path: Path) -> ParsedDocument:
        path = Path(file_path)
        try:
            import docx

            doc = docx.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(
                        cell.text.strip() for cell in row.cells if cell.text.strip()
                    )
                    if row_text:
                        paragraphs.append(row_text)

            full_text = "\n\n".join(paragraphs)
            return ParsedDocument(
                filename=path.name,
                file_type="docx",
                total_pages=1,
                raw_text=full_text,
                pages=[ParsedPage(page_number=1, text=full_text)],
                metadata={"engine": "python-docx"},
            )
        except ImportError:
            logger.warning("python-docx is not installed. Using raw text fallback.")
            text = path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(
                filename=path.name,
                file_type="docx",
                total_pages=1,
                raw_text=text,
                pages=[ParsedPage(page_number=1, text=text)],
                metadata={"engine": "fallback"},
            )


class ImageOCRParser(DocumentParser):
    """Image / scanned document parser with OCR."""

    def parse(self, file_path: Path) -> ParsedDocument:
        path = Path(file_path)
        try:
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(use_angle_cls=True, lang="en")
            result = ocr.ocr(str(path), cls=True)
            lines: list[str] = []
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0]
                    lines.append(text)
            full_text = "\n".join(lines)
            return ParsedDocument(
                filename=path.name,
                file_type=path.suffix.lstrip("."),
                total_pages=1,
                raw_text=full_text,
                pages=[ParsedPage(page_number=1, text=full_text)],
                metadata={"engine": "PaddleOCR"},
            )
        except ImportError:
            logger.warning(
                "PaddleOCR not installed. Returning placeholder OCR parsed document."
            )
            return ParsedDocument(
                filename=path.name,
                file_type=path.suffix.lstrip("."),
                total_pages=1,
                raw_text="[Scanned image document - PaddleOCR required for text extraction]",
                pages=[
                    ParsedPage(
                        page_number=1,
                        text="[Scanned image document - PaddleOCR required for text extraction]",
                    )
                ],
                metadata={"engine": "placeholder"},
            )


class UnifiedDocumentParser(DocumentParser):
    """Router parser that chooses appropriate parser based on file extension."""

    def __init__(self) -> None:
        self.pdf_parser = PdfParser()
        self.docx_parser = DocxParser()
        self.ocr_parser = ImageOCRParser()

    def parse(self, file_path: Path) -> ParsedDocument:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".pdf":
            return self.pdf_parser.parse(path)
        elif ext in [".docx", ".doc"]:
            return self.docx_parser.parse(path)
        elif ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            return self.ocr_parser.parse(path)
        elif ext in [".txt", ".md"]:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return ParsedDocument(
                filename=path.name,
                file_type=ext.lstrip("."),
                total_pages=1,
                raw_text=text,
                pages=[ParsedPage(page_number=1, text=text)],
                metadata={"engine": "plain_text"},
            )
        else:
            raise ValueError(f"Unsupported document format: {ext}")
