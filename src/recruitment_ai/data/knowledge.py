"""HR Knowledge Base loader for RAG policies, job profiles, skill definitions, and recruitment rules."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

CATEGORIES = ["hr_policies", "job_profiles", "skill_definitions", "recruitment_rules"]


class KnowledgeDocument(BaseModel):
    """Structured knowledge document for RAG indexing."""

    category: str  # hr_policies, job_profiles, skill_definitions, recruitment_rules
    title: str
    content: str  # markdown body, without the metadata header
    file_path: str
    doc_id: str | None = None  # stable id cited in analyses, e.g. "KB-RR-002"
    language: str = "en"
    roles: list[str] = Field(default_factory=lambda: ["all"])  # role keys or ["all"]
    version: str | None = None

    def applies_to(self, role: str) -> bool:
        """True when the document is relevant to the given role key (e.g. "ml_engineer")."""
        return "all" in self.roles or role in self.roles


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Separate a leading '---' YAML metadata block from the markdown body."""
    text = text.lstrip("﻿")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return yaml.safe_load(parts[1]) or {}, parts[2].lstrip("\n")
    return {}, text


class KnowledgeBaseLoader:
    """Loads and organizes HR knowledge sources for RAG retrieval."""

    def __init__(self, base_path: Path | None = None) -> None:
        self.base_path = base_path or Path("./data/knowledge_base")

    def load_all(self) -> list[KnowledgeDocument]:
        """Load all markdown documents across knowledge base subcategories."""
        documents: list[KnowledgeDocument] = []
        for category in CATEGORIES:
            cat_dir = self.base_path / category
            if not cat_dir.exists():
                continue
            for file_path in sorted(cat_dir.glob("*.md")):
                meta, body = split_front_matter(file_path.read_text(encoding="utf-8"))
                documents.append(
                    KnowledgeDocument(
                        category=meta.get("category", category),
                        title=meta.get("title") or file_path.stem.replace("_", " ").title(),
                        content=body,
                        file_path=str(file_path),
                        doc_id=meta.get("doc_id"),
                        language=meta.get("language", "en"),
                        roles=meta.get("roles") or ["all"],
                        version=str(meta["version"]) if "version" in meta else None,
                    )
                )
        return documents
