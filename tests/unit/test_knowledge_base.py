"""Quality gate for the HR knowledge base (data/knowledge_base/) used by RAG."""

import re
from pathlib import Path

import pytest

from recruitment_ai.data.knowledge import KnowledgeBaseLoader, split_front_matter
from recruitment_ai.data.tech_jobs import ROLE_PATTERNS

KB_PATH = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"
MAX_SECTION_WORDS = 300  # one "## section" must fit in one RAG chunk (chunk_size 500, configs/config.yaml)


@pytest.fixture(scope="module")
def docs():
    return KnowledgeBaseLoader(base_path=KB_PATH).load_all()


def test_every_document_has_metadata(docs):
    assert len(docs) >= 12
    for doc in docs:
        assert doc.doc_id and re.fullmatch(r"KB-[A-Z]{2}-\d{3}", doc.doc_id), doc.file_path
        assert doc.language in {"en", "fr"}, doc.file_path
        assert doc.version, doc.file_path
        assert not doc.content.startswith("---"), "metadata header must be stripped from content"


def test_doc_ids_are_unique(docs):
    ids = [d.doc_id for d in docs]
    assert len(ids) == len(set(ids))


def test_cross_references_point_to_existing_documents(docs):
    ids = {d.doc_id for d in docs}
    for doc in docs:
        for ref in re.findall(r"KB-[A-Z]{2}-\d{3}", doc.content):
            assert ref in ids, f"{doc.doc_id} cites unknown document {ref}"


def test_one_job_profile_per_target_role(docs):
    profiles = [d for d in docs if d.category == "job_profiles"]
    for role in ROLE_PATTERNS:
        matching = [d for d in profiles if role in d.roles]
        assert len(matching) == 1, f"role {role} needs exactly one job profile, found {len(matching)}"


def test_both_languages_are_present(docs):
    languages = [d.language for d in docs]
    assert languages.count("fr") >= 2 and languages.count("en") >= 8


def test_sections_fit_in_one_chunk(docs):
    for doc in docs:
        for section in re.split(r"\n## ", doc.content):
            words = len(section.split())
            assert words <= MAX_SECTION_WORDS, f"{doc.doc_id}: a section has {words} words"


def test_applies_to(docs):
    rubric = next(d for d in docs if d.doc_id == "KB-RR-002")
    ml_profile = next(d for d in docs if d.doc_id == "KB-JP-001")
    assert rubric.applies_to("devops_cloud")
    assert ml_profile.applies_to("ml_engineer") and not ml_profile.applies_to("devops_cloud")


def test_split_front_matter():
    meta, body = split_front_matter("---\ndoc_id: KB-XX-001\nroles: [a, b]\n---\n\n# Title\nText")
    assert meta == {"doc_id": "KB-XX-001", "roles": ["a", "b"]}
    assert body == "# Title\nText"
    assert split_front_matter("# No header") == ({}, "# No header")
