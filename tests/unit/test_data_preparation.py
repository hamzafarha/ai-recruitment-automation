"""Unit tests for Step 1b data preparation (no real dataset files needed)."""

import csv
from pathlib import Path

import pytest

from recruitment_ai.data.normalization import clean_value, normalize_date, text_hash
from recruitment_ai.data.resume_parser import convert_label, groundedness, tag_cv
from recruitment_ai.data.skill_lexicon import SkillNormalizer, build_skill_lexicon
from recruitment_ai.data.tech_jobs import assign_role, clean_jobs

CUSTOM_SKILLS = Path(__file__).resolve().parents[2] / "data" / "reference" / "custom_skills.json"


# --------------------------------------------------------------------------- normalization


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("07/2018", "2018-07"),
        ("2018-07", "2018-07"),
        ("August 2014", "2014-08"),
        ("Aug. 2014", "2014-08"),
        ("juin 2021", "2021-06"),
        ("07/15/2018", "2018-07"),
        ("2018", "2018"),
        ("Summer 2016", "2016"),
        ("Present", "present"),
        ("Current", "present"),
        ("aujourd'hui", "present"),
        ("", None),
        (None, None),
        ("unknown", None),
        ("13/2018", None),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


def test_clean_value_drops_junk_headings():
    assert clean_value("Certification") is None
    assert clean_value("  Copyright Notice ") is None
    assert clean_value("") is None
    assert clean_value(" AWS Certified ") == "AWS Certified"


def test_text_hash_ignores_case_and_punctuation():
    assert text_hash("Senior Python Developer!") == text_hash("senior  python developer")


# --------------------------------------------------------------------------- label conversion


@pytest.fixture
def target() -> dict:
    """A Resume Parser label with the awkward cases found during inspection."""
    return {
        "personal_information": {
            "full_name": None, "email": "a@b.com", "phone_numbers": ["555-1000", "555-2000"],
            "city": "Tunis", "state_or_region": None, "country": "Tunisia",
        },
        "professional_summary": None,
        "career_objective": "Build ML systems.",
        "work_experience": [
            {"job_title": "ML Engineer", "company": "Acme", "start_date": "07/2018", "end_date": "Present",
             "is_current": False, "description": "Built models.", "responsibilities": ["Trained CNNs"],
             "achievements": [], "skills_used": ["PyTorch", "python"]},
            {"job_title": "Support Technician", "company": None, "start_date": "2016", "end_date": "2017",
             "is_current": False, "description": None, "responsibilities": [], "achievements": [], "skills_used": []},
            {"job_title": None, "company": None, "start_date": None, "end_date": None},
        ],
        "education": [{"institution": None, "degree": "BSc Computer Science", "gpa": "3.8",
                       "honors": ["Dean's List"], "end_date": None, "graduation_date": "May 2016"}],
        "skills": {"technical": ["Python", "Machine Learning"], "tools_and_software": ["Docker"],
                   "domain": [], "soft": ["Teamwork"], "spoken_languages": ["English (Fluent)", "French"]},
        "certifications": [{"name": "Certification"}, {"name": "AWS Certified", "expiry_date": "2027"}],
        "projects": [{"name": "Chatbot", "technologies": ["RAG", ""]}, {"name": None}],
        "awards_and_honors": [{"name": "Best intern"}],
    }


def test_convert_label_identity_and_location(target):
    profile = convert_label(target, resume_id="r1")
    assert profile.id == "r1"
    assert profile.full_name is None  # missing stays missing, never invented
    assert profile.phone == "555-1000"
    assert profile.location == "Tunis, Tunisia"
    assert profile.summary == "Build ML systems."


def test_convert_label_experience(target):
    jobs = convert_label(target).experience
    assert len(jobs) == 2  # the entry without title and company is dropped
    assert jobs[0].start_date == "2018-07"
    assert jobs[0].end_date == "present" and jobs[0].is_current
    assert "- Trained CNNs" in jobs[0].description
    assert jobs[1].company is None and jobs[1].title == "Support Technician"


def test_convert_label_education_skills_languages(target):
    profile = convert_label(target)
    edu = profile.education[0]
    assert edu.institution is None and edu.end_date == "2016-05"
    assert edu.grade == "3.8; Dean's List"

    names = [(s.name, s.category) for s in profile.skills]
    assert names == [
        ("Python", "Technical"), ("Machine Learning", "Technical"), ("Docker", "Tool"),
        ("Teamwork", "Soft"), ("PyTorch", "Technical"),  # "python" from skills_used is a duplicate
    ]
    assert [(lang.language, lang.proficiency) for lang in profile.languages] == [
        ("English", "Fluent"), ("French", None),
    ]


def test_convert_label_certifications_projects(target):
    profile = convert_label(target)
    assert [c.name for c in profile.certifications] == ["AWS Certified"]  # junk heading removed
    assert profile.certifications[0].expiration_date == "2027"
    assert [(p.title, p.technologies) for p in profile.projects] == [("Chatbot", ["RAG"])]


# --------------------------------------------------------------------------- tagging


def test_tag_cv_flags():
    row = {
        "resume_id": "r1", "category": "React Developer",
        "resume_text": "JESSICA CLAIRE resumesample@example.com EClaireergetic aClaired diligent",
    }
    target = {"work_experience": [{"job_title": "Boiler Plant Operator/REACT"}]}
    tags = tag_cv(row, target, train_hashes={text_hash(row["resume_text"])})
    assert tags["is_tech_category"] and tags["leak_train"]
    assert tags["corrupted"] and tags["template_identity"]
    assert not tags["tech_title_match"]  # "REACT" in a boiler-plant title is not a tech profile


def test_tag_cv_tech_title():
    row = {"resume_id": "r2", "category": "Python Developer", "resume_text": "Senior Python Developer"}
    tags = tag_cv(row, {"work_experience": [{"job_title": "Senior Python Developer"}]}, train_hashes=set())
    assert tags["tech_title_match"] and not tags["leak_train"] and not tags["corrupted"]


def test_groundedness():
    assert groundedness("Python and SQL", {"skills": ["Python", "SQL"]}) == 1.0
    assert groundedness("Python only", {"skills": ["Python", "Kubernetes"]}) == 0.5


# --------------------------------------------------------------------------- tech jobs


def test_clean_jobs():
    long_text = "Build FastAPI services. " * 30
    jobs = [
        {"id": 1, "title": "Senior ML Engineer", "company": "A", "description": long_text},
        {"id": 2, "title": "Senior ML Engineer", "company": "A", "description": long_text + " Extra."},
        {"id": 3, "title": "Backend Developer", "company": "B", "description": long_text.upper()},
        {"id": 4, "title": "DevOps Engineer", "company": "C", "description": "Too short"},
        {"id": 5, "title": "SRE", "company": "D", "description": long_text + "� broken"},
        {"id": 6, "title": "Product Designer", "company": "E", "description": "Design. " * 100},
    ]
    clean, stats = clean_jobs(jobs)
    assert [j["id"] for j in clean] == [2, 6]  # longest duplicate kept; same text under B removed
    assert stats == {"input": 6, "too_short": 1, "broken_chars": 1, "duplicates": 2, "output": 2}
    assert clean[0]["role"] == "ml_engineer" and clean[1]["role"] is None


def test_assign_role():
    assert assign_role("Applied NLP Scientist") == "nlp_llm_engineer"
    assert assign_role("Site Reliability Engineer") == "devops_cloud"
    assert assign_role("Accountant") is None


# --------------------------------------------------------------------------- skill lexicon


def _write_esco(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["conceptUri", "preferredLabel", "altLabels", "hiddenLabels", "skillType"])
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def normalizer(tmp_path: Path) -> SkillNormalizer:
    """Lexicon from a tiny fake ESCO + the real custom extension."""
    en = [
        {"conceptUri": "esco:ml", "preferredLabel": "machine learning", "altLabels": "ML\nstatistical learning",
         "hiddenLabels": "", "skillType": "knowledge"},
        {"conceptUri": "esco:cv", "preferredLabel": "computer vision", "altLabels": "TensorFlow",
         "hiddenLabels": "", "skillType": "knowledge"},
        {"conceptUri": "esco:py", "preferredLabel": "Python (computer programming)", "altLabels": "Python",
         "hiddenLabels": "", "skillType": "knowledge"},
    ]
    fr = [
        {"conceptUri": "esco:ml", "preferredLabel": "apprentissage automatique", "altLabels": "",
         "hiddenLabels": "", "skillType": "knowledge"},
        {"conceptUri": "esco:cv", "preferredLabel": "vision par ordinateur", "altLabels": "",
         "hiddenLabels": "", "skillType": "knowledge"},
        {"conceptUri": "esco:py", "preferredLabel": "Python (programmation informatique)", "altLabels": "",
         "hiddenLabels": "", "skillType": "knowledge"},
    ]
    _write_esco(tmp_path / "esco" / "en" / "skills_en.csv", en)
    _write_esco(tmp_path / "esco" / "fr" / "skills_fr.csv", fr)
    out = tmp_path / "lexicon.json"
    stats = build_skill_lexicon(tmp_path / "esco", CUSTOM_SKILLS, out)
    assert stats["esco_concepts"] == 3
    return SkillNormalizer.from_file(out)


@pytest.mark.parametrize(
    ("raw", "expected_en"),
    [
        ("Machine Learning", "machine learning"),
        ("apprentissage automatique", "machine learning"),  # French label -> same concept
        ("ML", "machine learning"),
        ("Python 3", "Python (computer programming)"),       # version stripped
        ("ReactJS", "React"),                                 # custom extension
        ("react.js", "React"),
        ("AWS", "Amazon Web Services"),
        ("TensorFlow", "TensorFlow"),                         # custom beats ESCO's coarse mapping
        ("Intégration continue", "CI/CD"),
        ("integration continue", "CI/CD"),                    # accents optional
    ],
)
def test_skill_normalizer(normalizer, raw, expected_en):
    concept = normalizer.normalize(raw)
    assert concept is not None and concept.label_en == expected_en


def test_skill_normalizer_unknown_and_coverage(normalizer):
    assert normalizer.normalize("Underwater basket weaving") is None
    assert normalizer.coverage(["Python", "Docker", "Unknown skill", "Kubernetes"]) == 0.75
