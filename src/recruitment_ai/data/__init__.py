"""Data layer package for dataset ingestion, ontologies, synthetic generation, and knowledge sources."""

from recruitment_ai.data.knowledge import KnowledgeBaseLoader
from recruitment_ai.data.loader import DatasetCatalog, DatasetLoader
from recruitment_ai.data.ontology import (
    ESCONormalizer,
    ONetNormalizer,
    SkillOntologyManager,
)
from recruitment_ai.data.synthetic_generator import SyntheticDataGenerator

__all__ = [
    "DatasetCatalog",
    "DatasetLoader",
    "ESCONormalizer",
    "ONetNormalizer",
    "SkillOntologyManager",
    "SyntheticDataGenerator",
    "KnowledgeBaseLoader",
]
