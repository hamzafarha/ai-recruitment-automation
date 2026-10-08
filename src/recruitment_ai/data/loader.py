"""Dataset catalog, data loader, and ingestion registry for recruitment datasets."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DatasetMetadata(BaseModel):
    """Metadata specification for each dataset in the data layer."""

    dataset_id: str
    name: str
    approx_size: str
    main_use: str
    priority: str  # Core, Core/Enrichment, Optional
    source_url: str
    local_dir: Path
    category: str  # raw, synthetic, knowledge_base, evaluation


@dataclass
class DatasetCatalog:
    """Registry of all 14 data sources defined in the project architecture."""

    DATASETS: dict[str, DatasetMetadata] = None  # type: ignore[assignment]

    @classmethod
    def get_catalog(cls, base_dir: Path | None = None) -> dict[str, DatasetMetadata]:
        root = base_dir or Path("./data")
        return {
            "resume_parser": DatasetMetadata(
                dataset_id="resume_parser",
                name="Resume Parser Dataset",
                approx_size="~14,635 CVs",
                main_use="CV -> structured JSON / information extraction",
                priority="Core",
                source_url="https://huggingface.co/datasets/md-zain/resume-parser-dataset",
                local_dir=root / "raw" / "resumes" / "resume_parser",
                category="raw",
            ),
            "resume_ner": DatasetMetadata(
                dataset_id="resume_ner",
                name="Resume NER Dataset",
                approx_size="~22,855 CVs",
                main_use="Named Entity Recognition: skills, education, experience, orgs",
                priority="Core",
                source_url="https://huggingface.co/datasets/yashpwr/resume-ner-training-data",
                local_dir=root / "raw" / "resumes" / "resume_ner",
                category="raw",
            ),
            "resume_skill_extractor": DatasetMetadata(
                dataset_id="resume_skill_extractor",
                name="Resume Skill Extractor / Job Description Dataset",
                approx_size="~3,050 JDs",
                main_use="Extract required skills from job descriptions",
                priority="Core",
                source_url="https://huggingface.co/datasets/keerthanshetty/resume-skill-extractor-dataset",
                local_dir=root / "raw" / "job_descriptions" / "resume_skill_extractor",
                category="raw",
            ),
            "tech_jobs_2026": DatasetMetadata(
                dataset_id="tech_jobs_2026",
                name="Tech Jobs 2026 Dataset",
                approx_size="~1,020 jobs",
                main_use="Realistic job descriptions for matching + RAG + testing",
                priority="Core",
                source_url="https://huggingface.co/datasets/Sundaydream/tech-jobs-dataset-2026",
                local_dir=root / "raw" / "job_descriptions" / "tech_jobs_2026",
                category="raw",
            ),
            "esco": DatasetMetadata(
                dataset_id="esco",
                name="ESCO v1.2.1",
                approx_size="Large ontology",
                main_use="Skill/occupation normalization, synonyms and relationships",
                priority="Core",
                source_url="https://esco.ec.europa.eu/en/use-esco/download",
                local_dir=root / "raw" / "ontologies" / "esco",
                category="raw",
            ),
            "onet": DatasetMetadata(
                dataset_id="onet",
                name="O*NET 31.0",
                approx_size="Large ontology",
                main_use="Skills, tasks, occupations and technology enrichment",
                priority="Core/Enrichment",
                source_url="https://www.onetcenter.org/database.html",
                local_dir=root / "raw" / "ontologies" / "onet",
                category="raw",
            ),
            "job_matching": DatasetMetadata(
                dataset_id="job_matching",
                name="Job Matching Dataset",
                approx_size="~9,544 pairs",
                main_use="Candidate <-> Job matching / ranking",
                priority="Core",
                source_url="https://huggingface.co/datasets/DjallelBr/job-matching-dataset",
                local_dir=root / "raw" / "matching" / "job_matching",
                category="raw",
            ),
            "doclaynet": DatasetMetadata(
                dataset_id="doclaynet",
                name="DocLayNet",
                approx_size="~80,863 pages",
                main_use="Document layout understanding / multimodal CV processing",
                priority="Optional",
                source_url="https://github.com/DS4SD/DocLayNet",
                local_dir=root / "raw" / "document_layout" / "doclaynet",
                category="raw",
            ),
            "synthetic_cvs": DatasetMetadata(
                dataset_id="synthetic_cvs",
                name="Synthetic CVs",
                approx_size="Generated (~100-500)",
                main_use="Controlled CV variations, edge cases and missing information",
                priority="Core",
                source_url="internal://synthetic_generator/cvs",
                local_dir=root / "synthetic" / "cvs",
                category="synthetic",
            ),
            "synthetic_jobs": DatasetMetadata(
                dataset_id="synthetic_jobs",
                name="Synthetic Job Descriptions",
                approx_size="Generated (~50-200)",
                main_use="Controlled job requirements and hard cases",
                priority="Core",
                source_url="internal://synthetic_generator/job_descriptions",
                local_dir=root / "synthetic" / "job_descriptions",
                category="synthetic",
            ),
            "synthetic_pairs": DatasetMetadata(
                dataset_id="synthetic_pairs",
                name="Synthetic CV-JD Matching Pairs",
                approx_size="Generated (~150-600)",
                main_use="Ground-truth matching + hard negatives",
                priority="Core",
                source_url="internal://synthetic_generator/matching_pairs",
                local_dir=root / "synthetic" / "matching_pairs",
                category="synthetic",
            ),
            "hr_knowledge_base": DatasetMetadata(
                dataset_id="hr_knowledge_base",
                name="HR Knowledge Base",
                approx_size="Generated/curated",
                main_use="RAG for HR policies, skills, job profiles and evaluation rules",
                priority="Core",
                source_url="internal://knowledge_base",
                local_dir=root / "knowledge_base",
                category="knowledge_base",
            ),
            "agent_traces": DatasetMetadata(
                dataset_id="agent_traces",
                name="Agent Traces / Interaction Logs",
                approx_size="Generated during project",
                main_use="Agent evaluation, observability and error analysis",
                priority="Core",
                source_url="internal://evaluation/agent_traces",
                local_dir=root / "evaluation" / "agent_traces",
                category="evaluation",
            ),
            "human_evaluation": DatasetMetadata(
                dataset_id="human_evaluation",
                name="Human Evaluation / Review Data",
                approx_size="Generated during evaluation",
                main_use="Validate LLM/agent outputs and matching quality",
                priority="Core",
                source_url="internal://evaluation/human_labels",
                local_dir=root / "evaluation" / "human_labels",
                category="evaluation",
            ),
        }

    @classmethod
    def get_by_id(
        cls, dataset_id: str, base_dir: Path | None = None
    ) -> DatasetMetadata:
        catalog = cls.get_catalog(base_dir)
        if dataset_id not in catalog:
            raise KeyError(
                f"Dataset '{dataset_id}' not found in catalog. Available: {list(catalog.keys())}"
            )
        return catalog[dataset_id]

    @classmethod
    def list_datasets(cls, base_dir: Path | None = None) -> list[DatasetMetadata]:
        return list(cls.get_catalog(base_dir).values())


class DatasetStatus(BaseModel):
    """File count and presence status for a dataset."""

    dataset_id: str
    name: str
    priority: str
    local_dir: str
    exists: bool
    file_count: int
    has_sample_data: bool


class DatasetLoader:
    """Manages dataset retrieval, sample caching, and local inspection."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path("./data")
        self.catalog = DatasetCatalog.get_catalog(self.base_dir)

    def check_status(self, dataset_id: str | None = None) -> list[DatasetStatus]:
        """Check local availability of dataset(s)."""
        targets = (
            [self.catalog[dataset_id]] if dataset_id else list(self.catalog.values())
        )
        results: list[DatasetStatus] = []

        for meta in targets:
            exists = meta.local_dir.exists()
            file_count = 0
            if exists:
                file_count = len(list(meta.local_dir.rglob("*.*")))
            results.append(
                DatasetStatus(
                    dataset_id=meta.dataset_id,
                    name=meta.name,
                    priority=meta.priority,
                    local_dir=str(meta.local_dir),
                    exists=exists,
                    file_count=file_count,
                    has_sample_data=file_count > 0,
                )
            )
        return results

    def load_local_samples(
        self, dataset_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Load locally available samples as dictionaries."""
        meta = self.catalog.get(dataset_id)
        if not meta or not meta.local_dir.exists():
            return []

        samples: list[dict[str, Any]] = []
        for file_path in meta.local_dir.rglob("*.json"):
            if len(samples) >= limit:
                break
            try:
                content = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(content, list):
                    samples.extend(content[: limit - len(samples)])
                elif isinstance(content, dict):
                    samples.append(content)
            except Exception as exc:
                logger.warning(f"Error parsing sample file {file_path}: {exc}")

        return samples

    def download_dataset(
        self,
        dataset_id: str,
        split: str = "train",
        sample_limit: int | None = None,
    ) -> bool:
        """Download or stream a dataset from Hugging Face or create local representation.

        Uses the `datasets` library if installed, otherwise saves a metadata descriptor.
        """
        meta = self.catalog.get(dataset_id)
        if not meta:
            raise ValueError(f"Unknown dataset '{dataset_id}'")

        meta.local_dir.mkdir(parents=True, exist_ok=True)

        if meta.category == "synthetic":
            from recruitment_ai.data.synthetic_generator import SyntheticDataGenerator

            generator = SyntheticDataGenerator(output_root=self.base_dir / "synthetic")
            generator.generate_all()
            logger.info(f"Generated synthetic dataset: {dataset_id}")
            return True

        if "huggingface.co/datasets/" in meta.source_url:
            repo_id = meta.source_url.replace(
                "https://huggingface.co/datasets/", ""
            ).split("?")[0]
            try:
                # Dynamic import to avoid strict dependency
                import datasets  # type: ignore[import-untyped]

                logger.info(
                    f"Downloading {repo_id} via Hugging Face datasets library..."
                )
                ds = datasets.load_dataset(repo_id, split=split)
                if sample_limit:
                    ds = ds.select(range(min(sample_limit, len(ds))))

                # Save sample to local directory
                target_file = meta.local_dir / "samples.json"
                records = [ds[i] for i in range(min(len(ds), sample_limit or 50))]
                target_file.write_text(
                    json.dumps(records, indent=2, default=str), encoding="utf-8"
                )
                logger.info(f"Saved {len(records)} samples to {target_file}")
                return True
            except ImportError:
                logger.warning(
                    f"The 'datasets' package is not installed. Saving manifest descriptor to {meta.local_dir}."
                )
                self._save_manifest(meta)
                return True
            except Exception as exc:
                logger.warning(
                    f"Could not download {repo_id} directly ({exc}). Saving manifest descriptor."
                )
                self._save_manifest(meta)
                return False

        # For external ontologies and layout benchmarks
        self._save_manifest(meta)
        return True

    def _save_manifest(self, meta: DatasetMetadata) -> None:
        """Saves a local manifest and README descriptor inside the dataset directory."""
        meta.local_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = meta.local_dir / "dataset_manifest.json"
        manifest_data = {
            "dataset_id": meta.dataset_id,
            "name": meta.name,
            "approx_size": meta.approx_size,
            "main_use": meta.main_use,
            "priority": meta.priority,
            "source_url": meta.source_url,
            "status": "ready_for_ingestion",
        }
        manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
