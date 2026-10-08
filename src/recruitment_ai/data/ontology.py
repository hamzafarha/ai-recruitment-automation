"""Skill and occupation ontology normalization based on ESCO v1.2.1 and O*NET 31.0."""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OntologySkillConcept(BaseModel):
    """Standardized ontology skill entity."""

    concept_uri: str
    preferred_label: str
    alt_labels: list[str] = Field(default_factory=list)
    skill_type: str = "knowledge"  # knowledge, skill/competence, attitude
    source: str = "ESCO"  # ESCO or ONET


class ESCONormalizer:
    """Normalizes candidate/job skills and occupations against ESCO v1.2.1."""

    def __init__(self, ontology_path: Path | None = None) -> None:
        self.ontology_path = ontology_path or Path("./data/raw/ontologies/esco")
        self._skills_map: dict[str, OntologySkillConcept] = {}
        self._load_local_ontology()

    def _load_local_ontology(self) -> None:
        """Load local ESCO JSON/CSV ontology if present, else seed defaults."""
        skills_file = self.ontology_path / "skills_sample.json"
        if skills_file.exists():
            try:
                data = json.loads(skills_file.read_text(encoding="utf-8"))
                for item in data:
                    concept = OntologySkillConcept.model_validate(item)
                    self._register_concept(concept)
                return
            except Exception as exc:
                logger.warning(f"Failed to load ESCO ontology: {exc}")

        # Seed essential tech concepts
        defaults = [
            OntologySkillConcept(
                concept_uri="http://data.europa.eu/esco/skill/s1",
                preferred_label="Machine Learning",
                alt_labels=["ML", "Statistical Learning", "Deep Learning"],
                source="ESCO",
            ),
            OntologySkillConcept(
                concept_uri="http://data.europa.eu/esco/skill/s2",
                preferred_label="Python",
                alt_labels=["Python Programming", "Python 3"],
                source="ESCO",
            ),
            OntologySkillConcept(
                concept_uri="http://data.europa.eu/esco/skill/s3",
                preferred_label="Natural Language Processing",
                alt_labels=["NLP", "Text Analytics", "Language Models"],
                source="ESCO",
            ),
            OntologySkillConcept(
                concept_uri="http://data.europa.eu/esco/skill/s4",
                preferred_label="Docker",
                alt_labels=["Docker Containerization", "Container Technology"],
                source="ESCO",
            ),
        ]
        for c in defaults:
            self._register_concept(c)

    def _register_concept(self, concept: OntologySkillConcept) -> None:
        self._skills_map[concept.preferred_label.lower()] = concept
        for alt in concept.alt_labels:
            self._skills_map[alt.lower()] = concept

    def normalize_skill(self, raw_skill: str) -> OntologySkillConcept | None:
        """Look up canonical ESCO skill concept for an extracted skill string."""
        return self._skills_map.get(raw_skill.strip().lower())


class ONetNormalizer:
    """Enriches skills, tasks, and technology requirements against O*NET 31.0."""

    def __init__(self, ontology_path: Path | None = None) -> None:
        self.ontology_path = ontology_path or Path("./data/raw/ontologies/onet")
        self._tech_skills: dict[str, str] = {
            "pytorch": "Deep Learning Software",
            "transformers": "NLP AI Libraries",
            "fastapi": "Web Frameworks",
            "postgresql": "Relational Databases",
            "qdrant": "Vector Database Management Systems",
        }
        self._occupations_map: dict[str, OntologySkillConcept] = {}
        self._load_local_ontology()

    def _load_local_ontology(self) -> None:
        """Load local O*NET JSON ontology if present, else seed defaults."""
        skills_file = self.ontology_path / "skills_sample.json"
        if skills_file.exists():
            try:
                data = json.loads(skills_file.read_text(encoding="utf-8"))
                for item in data:
                    concept = OntologySkillConcept.model_validate(item)
                    self._register_occupation(concept)
                return
            except Exception as exc:
                logger.warning(f"Failed to load O*NET ontology: {exc}")

        defaults = [
            OntologySkillConcept(
                concept_uri="15-1252.00",
                preferred_label="Software Developers",
                alt_labels=[
                    "Full Stack Developer",
                    "Software Engineer",
                    "Backend Developer",
                ],
                skill_type="occupation",
                source="ONET",
            ),
            OntologySkillConcept(
                concept_uri="15-2051.00",
                preferred_label="Data Scientists",
                alt_labels=[
                    "Machine Learning Engineer",
                    "AI Scientist",
                    "Data Modeler",
                ],
                skill_type="occupation",
                source="ONET",
            ),
        ]
        for c in defaults:
            self._register_occupation(c)

    def _register_occupation(self, concept: OntologySkillConcept) -> None:
        self._occupations_map[concept.preferred_label.lower()] = concept
        for alt in concept.alt_labels:
            self._occupations_map[alt.lower()] = concept

    def normalize_occupation(self, title: str) -> OntologySkillConcept | None:
        """Look up canonical O*NET occupation concept for a title."""
        return self._occupations_map.get(title.strip().lower())

    def get_technology_category(self, tech_name: str) -> str:
        """Retrieve O*NET technology categorization."""
        return self._tech_skills.get(
            tech_name.strip().lower(), "General Technology Skill"
        )


class SkillOntologyManager:
    """Unified ontology manager combining ESCO and O*NET for semantic normalization."""

    def __init__(
        self, esco_dir: Path | None = None, onet_dir: Path | None = None
    ) -> None:
        self.esco = ESCONormalizer(ontology_path=esco_dir)
        self.onet = ONetNormalizer(ontology_path=onet_dir)

    def normalize_skill(self, skill_name: str) -> OntologySkillConcept | None:
        """Lookup ESCO concept for a skill."""
        return self.esco.normalize_skill(skill_name)

    def enrich_skill_list(self, skills: list[str]) -> list[str]:
        """Convert a list of raw skills into canonical labels where recognized."""
        enriched: list[str] = []
        for s in skills:
            concept = self.esco.normalize_skill(s)
            enriched.append(concept.preferred_label if concept else s)
        return enriched

    def normalize(self, skill_name: str) -> dict[str, Any]:
        """Normalize skill string to canonical label, source ontology, and category."""
        clean = skill_name.strip()
        esco_match = self.esco.normalize_skill(clean)
        canonical = esco_match.preferred_label if esco_match else clean
        category = self.onet.get_technology_category(clean)

        return {
            "raw_skill": skill_name,
            "canonical_label": canonical,
            "ontology_uri": esco_match.concept_uri if esco_match else "custom:skill",
            "category": category,
            "normalized": esco_match is not None,
        }
