"""Unit tests for the data layer: loader, ontologies, synthetic generator, and knowledge base."""

from pathlib import Path

from recruitment_ai.data.knowledge import KnowledgeBaseLoader
from recruitment_ai.data.loader import DatasetCatalog, DatasetLoader
from recruitment_ai.data.ontology import (
    ESCONormalizer,
    ONetNormalizer,
    SkillOntologyManager,
)
from recruitment_ai.data.synthetic_generator import SyntheticDataGenerator


class TestDatasetCatalogAndLoader:
    def test_catalog_contains_all_14_datasets(self) -> None:
        catalog = DatasetCatalog.get_catalog(Path("./data"))
        assert len(catalog) == 14
        expected_ids = {
            "resume_parser",
            "resume_ner",
            "resume_skill_extractor",
            "tech_jobs_2026",
            "esco",
            "onet",
            "job_matching",
            "doclaynet",
            "synthetic_cvs",
            "synthetic_jobs",
            "synthetic_pairs",
            "hr_knowledge_base",
            "agent_traces",
            "human_evaluation",
        }
        assert set(catalog.keys()) == expected_ids

    def test_dataset_metadata_integrity(self) -> None:
        tech_jobs = DatasetCatalog.get_by_id("tech_jobs_2026")
        assert tech_jobs.priority == "Core"
        assert (
            "huggingface.co/datasets/Sundaydream/tech-jobs-dataset-2026"
            in tech_jobs.source_url
        )

        esco = DatasetCatalog.get_by_id("esco")
        assert esco.priority == "Core"
        assert "esco.ec.europa.eu" in esco.source_url

    def test_loader_status_check(self) -> None:
        loader = DatasetLoader(base_dir=Path("./data"))
        statuses = loader.check_status()
        assert len(statuses) == 14
        for status in statuses:
            assert status.exists is True
            assert status.file_count > 0

    def test_loader_load_local_samples(self) -> None:
        loader = DatasetLoader(base_dir=Path("./data"))
        samples = loader.load_local_samples("synthetic_cvs", limit=3)
        assert len(samples) > 0
        assert "full_name" in samples[0]
        assert "skills" in samples[0]


class TestOntologyNormalization:
    def test_esco_normalizer_direct_and_alias(self) -> None:
        normalizer = ESCONormalizer(ontology_path=Path("./data/raw/ontologies/esco"))
        # Preferred label match
        match_ml = normalizer.normalize_skill("Machine Learning")
        assert match_ml is not None
        assert match_ml.preferred_label == "Machine Learning"

        # Alias match
        match_alias = normalizer.normalize_skill("Deep Learning")
        assert match_alias is not None
        assert match_alias.preferred_label == "Machine Learning"

    def test_onet_normalizer(self) -> None:
        onet = ONetNormalizer(ontology_path=Path("./data/raw/ontologies/onet"))
        occ = onet.normalize_occupation("Full Stack Developer")
        assert occ is not None
        assert occ.preferred_label == "Software Developers"

    def test_skill_ontology_manager_enrichment(self) -> None:
        manager = SkillOntologyManager()
        normalized = manager.normalize_skill("ML")
        assert normalized is not None
        assert normalized.preferred_label == "Machine Learning"

        enriched = manager.enrich_skill_list(["ML", "Python", "UnknownSkill123"])
        assert "Machine Learning" in enriched
        assert "Python" in enriched
        assert "UnknownSkill123" in enriched  # Preserved original


class TestSyntheticDataGenerator:
    def test_synthetic_generator_creates_complete_suite(self, tmp_path: Path) -> None:
        generator = SyntheticDataGenerator(output_root=tmp_path)
        counts = generator.generate_all()

        assert counts["cvs_generated"] >= 3
        assert counts["jobs_generated"] >= 2
        assert counts["matching_pairs_generated"] >= 3

        # Verify files created
        assert (tmp_path / "cvs").exists()
        assert (tmp_path / "job_descriptions").exists()
        assert (tmp_path / "matching_pairs").exists()

        # Check matching pair content
        pair_files = list((tmp_path / "matching_pairs").glob("*.json"))
        assert len(pair_files) >= 3


class TestKnowledgeBaseLoader:
    def test_load_all_knowledge_documents(self) -> None:
        loader = KnowledgeBaseLoader(base_path=Path("./data/knowledge_base"))
        docs = loader.load_all()
        assert len(docs) >= 5

        categories = {doc.category for doc in docs}
        assert "hr_policies" in categories
        assert "job_profiles" in categories
        assert "skill_definitions" in categories
        assert "recruitment_rules" in categories
