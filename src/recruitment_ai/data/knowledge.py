"""HR Knowledge Base loader for RAG policies, job profiles, skill definitions, and recruitment rules."""

from pathlib import Path

from pydantic import BaseModel


class KnowledgeDocument(BaseModel):
    """Structured knowledge document for RAG indexing."""

    category: str  # hr_policies, job_profiles, skill_definitions, recruitment_rules
    title: str
    content: str
    file_path: str


class KnowledgeBaseLoader:
    """Loads and organizes HR knowledge sources for RAG retrieval."""

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path = base_path or Path("./data/knowledge_base")

    def load_all(self) -> list[KnowledgeDocument]:
        """Load all markdown documents across knowledge base subcategories."""
        documents: list[KnowledgeDocument] = []
        categories = [
            "hr_policies",
            "job_profiles",
            "skill_definitions",
            "recruitment_rules",
        ]

        for category in categories:
            cat_dir = self.base_path / category
            if not cat_dir.exists():
                continue
            for file_path in cat_dir.glob("*.md"):
                content = file_path.read_text(encoding="utf-8")
                documents.append(
                    KnowledgeDocument(
                        category=category,
                        title=file_path.stem.replace("_", " ").title(),
                        content=content,
                        file_path=str(file_path),
                    )
                )
        return documents
