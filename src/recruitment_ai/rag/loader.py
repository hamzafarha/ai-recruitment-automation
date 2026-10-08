"""Document loading utilities for RAG ingestion."""

from pathlib import Path

from recruitment_ai.ingestion.parser import ParsedDocument, UnifiedDocumentParser


class DocumentLoader:
    """Loads documents from disk and parses them into standardized ParsedDocument representations."""

    def __init__(self, parser: UnifiedDocumentParser | None = None) -> None:
        self.parser = parser or UnifiedDocumentParser()

    def load_file(self, file_path: Path) -> ParsedDocument:
        """Load and parse single file."""
        return self.parser.parse(file_path)

    def load_directory(
        self, directory_path: Path, glob_pattern: str = "*.*"
    ) -> list[ParsedDocument]:
        """Load and parse all matching documents in a directory."""
        results: list[ParsedDocument] = []
        for file_path in directory_path.glob(glob_pattern):
            if file_path.is_file() and not file_path.name.startswith("."):
                try:
                    results.append(self.load_file(file_path))
                except Exception:
                    continue
        return results
