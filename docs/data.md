# AI Recruitment Automation — Data Layer & Knowledge Architecture

This document specifies the complete **Data Layer** architecture, directory hierarchy, dataset catalog, normalization ontologies, synthetic data generators, and privacy governance guidelines for the AI Recruitment Automation System.

---

## 1. Complete Dataset Catalog (14 Data Sources)

| # | Dataset / Data Source | Approx. Size | Main Use in Project | Priority | Source / Repository Link |
|---|---|---:|---|---|---|
| **1** | **Resume Parser Dataset** | ~14,635 CVs | CV → structured JSON / information extraction | 🔴 Core | [`md-zain/resume-parser-dataset`](https://huggingface.co/datasets/md-zain/resume-parser-dataset) |
| **2** | **Resume NER Dataset** | ~22,855 CVs | Named Entity Recognition: skills, education, experience, orgs | 🔴 Core | [`yashpwr/resume-ner-training-data`](https://huggingface.co/datasets/yashpwr/resume-ner-training-data) |
| **3** | **Resume Skill Extractor / Job Description Dataset** | ~3,050 JDs | Extract required skills from job descriptions | 🔴 Core | [`keerthanshetty/resume-skill-extractor-dataset`](https://huggingface.co/datasets/keerthanshetty/resume-skill-extractor-dataset) |
| **4** | **Tech Jobs 2026 Dataset** | ~1,020 jobs | Realistic job descriptions for matching + RAG + testing | 🔴 Core | [`Sundaydream/tech-jobs-dataset-2026`](https://huggingface.co/datasets/Sundaydream/tech-jobs-dataset-2026) |
| **5** | **ESCO v1.2.1** | Large ontology | Skill/occupation normalization, synonyms and relationships | 🔴 Core | [`ESCO Download Portal`](https://esco.ec.europa.eu/en/use-esco/download) |
| **6** | **O*NET 31.0** | Large ontology | Skills, tasks, occupations and technology enrichment | 🟠 Core/Enrichment | [`O*NET Database Portal`](https://www.onetcenter.org/database.html) |
| **7** | **Job Matching Dataset** | ~9,544 pairs | Candidate ↔ Job matching / ranking | 🔴 Core | [`DjallelBr/job-matching-dataset`](https://huggingface.co/datasets/DjallelBr/job-matching-dataset) |
| **8** | **DocLayNet** | ~80,863 pages | Document layout understanding / multimodal CV processing | 🟡 Optional | [`DS4SD/DocLayNet`](https://github.com/DS4SD/DocLayNet) |
| **9** | **Synthetic CVs** | Generated | Controlled CV variations, edge cases and missing information | 🔴 Core | Internal Generator (`recruitment_ai.data.synthetic_generator`) |
| **10** | **Synthetic Job Descriptions** | Generated | Controlled job requirements and hard cases | 🔴 Core | Internal Generator (`recruitment_ai.data.synthetic_generator`) |
| **11** | **Synthetic CV–JD Matching Pairs** | Generated | Ground-truth matching + hard negatives | 🔴 Core | Internal Generator (`recruitment_ai.data.synthetic_generator`) |
| **12** | **HR Knowledge Base** | Curated Markdown | RAG for HR policies, skills, job profiles and evaluation rules | 🔴 Core | Local Directory (`data/knowledge_base/`) |
| **13** | **Agent Traces / Interaction Logs** | Generated during runs | Agent evaluation, observability and error analysis | 🔴 Core | Local Directory (`data/evaluation/agent_traces/`) |
| **14** | **Human Evaluation / Review Data** | Curated/Feedback | Validate LLM/agent outputs and matching quality | 🔴 Core | Local Directory (`data/evaluation/human_labels/`) |

---

## 2. Data Flow Architecture

```text
                         DATA LAYER
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   RESUME DATA             JOB DATA            KNOWLEDGE
        │                     │                     │
        ├─ Resume Parser      ├─ JD Dataset        ├─ ESCO v1.2.1
        ├─ Resume NER         ├─ Tech Jobs 2026    └─ O*NET 31.0
        └─ Synthetic CVs      └─ Synthetic JDs
        │                     │
        └──────────────┬──────┘
                       ▼
                CV / JD PIPELINE
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Information         Skill Extraction
        Extraction                │
             │                    │
             └─────────┬──────────┘
                       ▼
                NORMALIZATION
                  ESCO / O*NET
                       │
                       ▼
              JOB MATCHING DATA
                       │
                       ▼
                MATCHING MODEL
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        RAG KNOWLEDGE       AGENT SYSTEM
             │                   │
             └─────────┬─────────┘
                       ▼
                  EVALUATION
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
       Agent Traces          Human Reviews
```

---

## 3. Directory Layout

The filesystem follows the exact structure required by the system design:

```text
data/
│
├── raw/
│   ├── resumes/
│   │   ├── resume_parser/              # Raw CV text and extracted JSON (HF: md-zain/resume-parser-dataset)
│   │   └── resume_ner/                 # Token-level NER tags (HF: yashpwr/resume-ner-training-data)
│   │
│   ├── job_descriptions/
│   │   ├── resume_skill_extractor/     # JD text & skill annotations (HF: keerthanshetty/resume-skill-extractor)
│   │   └── tech_jobs_2026/             # 2026 tech job postings (HF: Sundaydream/tech-jobs-dataset-2026)
│   │
│   ├── matching/
│   │   └── job_matching/               # Paired CV-JD relevance pairs (HF: DjallelBr/job-matching-dataset)
│   │
│   ├── ontologies/
│   │   ├── esco/                       # ESCO v1.2.1 skills & occupations CSV/JSON
│   │   └── onet/                       # O*NET 31.0 database & taxonomy files
│   │
│   └── document_layout/
│       └── doclaynet/                  # Layout understanding benchmark (Optional)
│
├── synthetic/
│   ├── cvs/                            # Controlled candidate variations & edge cases (Pydantic validated)
│   ├── job_descriptions/               # Controlled job specs with varying requirement strictness
│   └── matching_pairs/                 # Ground-truth matches, partial matches, and hard negatives
│
├── knowledge_base/
│   ├── hr_policies/                    # Remote work, compensation bands, DE&I policies
│   ├── job_profiles/                   # Standardized role expectations (Senior ML, NLP, Full Stack)
│   ├── skill_definitions/              # Tech competency taxonomy & prerequisite hierarchies
│   └── recruitment_rules/              # Screening rules, threshold scoring rubrics, fit criteria
│
└── evaluation/
    ├── human_labels/                   # Recruiter annotations and inter-annotator agreements
    ├── agent_traces/                   # LangGraph session execution traces & agent intermediate logs
    └── benchmark/                      # Golden evaluation benchmarks (NDCG@K, Recall, Precision)
```

---

## 4. Ingestion, Downloading & Seeding CLI

The repository includes utilities to inspect the dataset catalog, stream/download data on demand, and seed local validation samples:

### A. Inspect Catalog & Local Status

```bash
# Display all 14 datasets and metadata
python scripts/download_data.py --list

# Verify local directories and file counts
python scripts/download_data.py --status
```

### B. Seed Deterministic Sample Data & Synthetic Sets

```bash
# Populate sample manifests, ontology subsets, synthetic profiles, and knowledge base documents
python scripts/seed_sample_data.py
```

### C. Download / Stream Specific or All Datasets

```bash
# Download or cache a specific dataset (e.g. tech_jobs_2026)
python scripts/download_data.py --dataset tech_jobs_2026 --limit 50

# Ingest/generate all datasets
python scripts/download_data.py --all
```

---

## 5. Ontologies & Normalization (`recruitment_ai.data.ontology`)

- **ESCO v1.2.1 (`ESCONormalizer`)**:
  - Resolves non-standard skill terms to canonical labels (e.g., `"ML"`, `"Statistical Learning"`, `"Deep Learning"` $\rightarrow$ `"Machine Learning"`).
  - Handles aliases, spelling variants, and multi-language European skill classifications.
- **O*NET 31.0 (`ONetNormalizer`)**:
  - Maps role titles to standardized SOC occupations (e.g., `"15-1252.00 Software Developers"`, `"15-2051.00 Data Scientists"`).
  - Categorizes tools and software into O*NET Hot Technology clusters (Deep Learning Frameworks, Vector Databases, Relational DBs).
- **Unified Manager (`SkillOntologyManager`)**:
  - Exposes `normalize(skill_name)` and `enrich_skill_list(skills)` to transform raw extracted tokens into structured ontology representations with URIs.

---

## 6. Synthetic Data Engine (`recruitment_ai.data.synthetic_generator`)

Produces clean, privacy-safe, reproducible data without external dependencies:
1. **Candidate Variations**:
   - High-match candidates (e.g. Senior ML Engineer with deep PyTorch, RAG, and Qdrant experience).
   - Near-miss / Junior candidates (meets foundational requirements but lacks years-of-experience threshold).
   - Cross-domain candidates (e.g. Data Analyst attempting to apply for ML Engineer).
2. **Hard Negatives**:
   - Candidates possessing deceptive buzzwords (e.g., superficial mentions of AI tools without practical engineering depth or necessary foundation).
3. **Pydantic Validation**:
   - All synthetic outputs conform to [`CandidateProfile`](file:///src/recruitment_ai/extraction/schemas.py) and [`JobDescription`](file:///src/recruitment_ai/extraction/schemas.py).

---

## 7. Privacy & GDPR Compliance

1. **Zero Real PII in Git**:
   - Raw candidate uploads, phone numbers, and identifying records are strictly omitted via `.gitignore`.
   - CI/CD pipelines and automated tests operate purely on synthetic and anonymized benchmarks.
2. **Blind Screening**:
   - The ingestion pipeline strips personal identifiers (name, age, gender, contact details) prior to feature extraction and ontology mapping to enforce fair evaluation.
