# Data Inspection Report — AI Recruitment Automation

**Project:** AI Recruitment Automation (Advanced Deep Learning)
**Step:** 1 — Building the evaluation set
**Date:** 15 October 2026
**Source:** `notebooks/00_data_inspection.ipynb` (all figures below come from its outputs)

---

## 1. Summary

We inspected every dataset file of the project before building anything, to decide which data can serve as our
*measuring stick* (evaluation set). Key conclusions:

- **Resume Parser is our main dataset.** Its labels are faithful to the CVs (96.1 % of short label values found in the
  text for tech CVs) and, checked visually, match the CV images. They are most likely LLM-generated, so we treat them as
  *silver* labels and hand-check the subset we keep.
- **All its original PDFs are scanned images** (40 / 40 sampled, no text layer). This gives us real multimodal test
  cases with labels: plain PDF text extraction returns nothing, so OCR or a vision-language model is mandatory.
- **Its `category` field is unreliable**: a "Data Science" CV belongs to a schoolteacher and a "React Developer" CV to a
  boiler-plant operator. CVs must be filtered by content, not by category.
- **Train/test leakage exists**: 126 test CVs have the exact same text as a train CV (18 of them in tech categories).
  They must be removed from the test set if we fine-tune on train.
- **Job Matching is rejected**: its 0–100 scores are not credible (95/100 for a data scientist on a full-stack job).
- **Tech Jobs 2026 is usable** for our 5 job descriptions and for RAG, after removing duplicates.
- **ESCO knows only 56 % of our tech skills**: it covers Python, SQL, machine learning, but not Docker, Kubernetes,
  AWS, PyTorch, React. It needs a project-specific extension to be useful for matching.
- **The in-repo HR knowledge base is far too small** (9 files, 518 words) and several in-repo sample files are
  hand-written placeholders, not real data.

---

## 2. Scope — what was inspected

| # | Dataset / files | Origin | Size | Location (`data/…`) |
|---|---|---|---|---|
| 1 | Resume Parser — test | Hugging Face `md-zain/resume-parser-dataset` | 1,463 CVs | `raw/resumes/resume_parser/hf/curated/` |
| 2 | Resume Parser — validation / train | same | 732 / 12,440 CVs | same |
| 3 | Resume Parser — original PDFs | same (`raw/` folder) | 40 sampled | `raw/resumes/resume_parser/hf/raw/` |
| 4 | Job Matching | HF `DjallelBr/job-matching-dataset` | 1,909 pairs | `raw/matching/job_matching/hf/` |
| 5 | Tech Jobs 2026 | HF `Sundaydream/tech-jobs-dataset-2026` | 1,020 postings | `raw/job_descriptions/tech_jobs_2026/hf/` |
| 6 | ESCO v1.2.1 (EN + FR) | esco.ec.europa.eu | 13,960 skills | `raw/ontologies/esco/hf/` |
| 7 | Skill Extractor | HF `keerthanshetty/resume-skill-extractor-dataset` | 3,050 jobs | `raw/job_descriptions/resume_skill_extractor/hf/` |
| 8 | Synthetic CVs / jobs / pairs | written in the repo | 4 / 2 / 3 | `synthetic/` |
| 9 | HR knowledge base | written in the repo | 9 files | `knowledge_base/` |
| 10 | Sample and evaluation files | written in the repo | 8 files | `raw/**/sample_*.json`, `evaluation/` |

All downloaded files sit in `hf/` folders, which are git-ignored because they contain real CVs (personal data).

---

## 3. Findings per dataset

### 3.1 Resume Parser — test split (text + JSON labels)

Each row holds the CV text (`resume_text`) and the expected structured output (`target_json`).

- **1,463 CVs in 68 categories**, of which **169 in our 8 tech categories**.
- CV text length: median 3,373 characters (min 90, max 35,163).

| Tech category | CVs (test) |
|---|---|
| Data Science | 30 |
| DevOps Engineer | 28 |
| Database | 25 |
| Python Developer | 24 |
| React Developer | 22 |
| Information Technology | 18 |
| Java Developer | 14 |
| Sql Developer | 8 |

**Field coverage** (share of CVs where the field is filled):

| Field | Coverage | Field | Coverage |
|---|---|---|---|
| skills | 100 % | professional summary | 73 % |
| work experience | 94 % | certifications | 33 % |
| education | 89 % | projects | 12 % |
| full name | 81 % | publications | 2 % |
| phone / email | 79 % / 74 % | references | 4 % |

The fields we need most (skills, experience, education) are almost always filled.

**Groundedness** — share of short label values (skills, titles, companies, dates) found word-for-word in the CV text:

| Split | All CVs | Tech CVs |
|---|---|---|
| test | 94.5 % | **96.1 %** |
| validation | 94.3 % | **96.9 %** |

The values *not* found in the text fall into two groups:

- **Inferred soft skills** ("Teamwork", "Communication", "Leadership"…): plausible, but written by the labeler rather
  than copied from the CV.
- **Section headings stored as values** ("Certification" ×93, "Copyright Notice" ×32, "Declaration" ×29,
  "Personal Details" ×19): small label noise.

**Evidence that labels are LLM-generated.** The worst tech CV (54 % grounded) has corrupted text: the anonymiser
replaced the letter *n* with "Claire" ("EClaireergetic aClaired diligeClairet…"). Its label is nevertheless clean
("Energetic and diligent…"). A human copying the text would not silently repair it; an LLM would. Only **2 of the 169**
tech CVs have this corruption.

**Template CVs.** 93 of the 169 tech CVs (55 %) come from a resume-template site and share the same placeholder
identity ("Jessica Claire", `resumesample@example.com`, `(555) 432-1000`). Personal-information fields are therefore
not meaningful for these CVs; work experience, education and skills are.

### 3.2 Resume Parser — original PDFs (multimodal)

- **40 PDFs sampled** (5 per tech category): **all 40 are image-only** — one embedded picture per page, no text layer.
  PyMuPDF extracts 0 characters.
- Image resolution ranges from **470×665 to 962×3,343 pixels**; the low-resolution ones are natural hard cases for OCR.
- **Visual check (image vs label)** on 3 CVs (Python developer, IT generalist, boiler-plant operator): names, job
  titles, companies, dates, degrees and skills in the label all appear in the image. **The labels describe the images
  correctly.**
- **Category noise**: of the first 5 sampled "tech" CVs, **2 are not tech at all** — a schoolteacher filed under
  *Data Science*, a boiler-plant operator filed under *React Developer* (his job title contains "REACT"). Categories
  were probably assigned by keyword. A simple keyword filter is too permissive to catch these (it accepted 164/169),
  so a manual or model-based check is needed.

### 3.3 Resume Parser — validation and train splits

| Split | CVs | Tech CVs | Duplicate texts inside split | Very short texts (< 300 chars) |
|---|---|---|---|---|
| train | 12,440 | 1,438 | 658 | 18 |
| validation | 732 | 84 | 1 | 0 |
| test | 1,463 | 169 | 19 | 2 |

**Leakage between splits** (no shared ids, but identical texts under different ids):

| Overlap | Same texts | of which tech CVs |
|---|---|---|
| test ∩ train | **126** | **18** |
| test ∩ validation | 6 | 0 |

If we fine-tune on `train` (Step 9), these 18 tech CVs would inflate the test score — the model would already have
seen them. They must be excluded from our test set.

### 3.4 Job Matching — rejected

- 1,909 pairs, each a prompt (job + short candidate summary) and a 0–100 score; **no documentation**; the prompt is
  written as an instruction to an *LLM evaluator*.
- Scores are squeezed: mean 66, half of all pairs between 57 and 78.
- Only **28 distinct jobs**, mostly outside tech (marketing, audit, civil and mechanical engineering).
- **The top scores are not credible**: 95/100 for a data-science/ML candidate on a *Full Stack (Python, React)* job;
  94/100 for a candidate with "ASP.NET, curriculum development, consulting" on the same job.

**Decision:** not used. Our matching labels will be produced by hand.

### 3.5 Tech Jobs 2026 — real job postings

| Role (title keywords) | Postings | Median description |
|---|---|---|
| Backend Developer | 188 | 3,549 chars |
| ML Engineer | 35 | 5,071 chars |
| DevOps / Cloud | 12 | 4,598 chars |
| NLP / LLM Engineer | 9 | 3,571 chars |
| Data Analyst / Scientist | 6 | 5,901 chars |

Quality:

- **144 duplicate postings** (same title + company) and **81 duplicate descriptions** → deduplicate first.
- 9 very short descriptions (< 500 chars), 1 with broken characters.
- Mostly full-time (807); also 106 internships, 75 contracts.
- Postings read as realistic, with explicit requirement lists (e.g. "Build and maintain FastAPI services in Python that
  serve ML models"), but they also contain long company boilerplate that RAG chunking must handle.

**Decision:** source of our 5 job descriptions (one per role) and of RAG content.

### 3.6 ESCO v1.2.1 — skills ontology (English + French)

- **13,960 skills** (10,734 skills/competences, 3,221 knowledge items) and **3,043 occupations**.
- **100 % of skills exist in French**, and every skill has at least one synonym.
- 18 tech occupations, e.g. *data scientist* (63 essential skills), *cloud DevOps engineer*, *software developer*.
- 21 duplicated concept ids in the CSV (to be dropped when loading).

**Coverage of our tech vocabulary — 18 / 32 terms found (56 %):**

| Found in ESCO | Not found in ESCO |
|---|---|
| Python, Java, JavaScript, SQL, Jenkins, Git, machine learning, deep learning, TensorFlow, NLP, computer vision, data analysis, statistics, Spark, Hadoop, DevOps, NoSQL, PostgreSQL | React, Docker, Kubernetes, Terraform, AWS, Microsoft Azure, Linux, PyTorch, CI/CD, FastAPI, REST API, MongoDB, Power BI, Tableau |

Examples of the French labels: *machine learning → apprentissage automatique*, *deep learning → apprentissage
profond*, *natural language processing → traitement automatique du langage naturel*.

Some synonym mappings are too coarse for matching: *TensorFlow* maps to the concept *computer vision*, *Git* to
*tools for software configuration management*, *data analysis* to *statistics*.

**Decision:** useful for EN↔FR skill normalisation and soft skills, but must be **extended with a small
project-specific list** of modern tools (cloud, containers, frameworks).

### 3.7 Skill Extractor — job description → required skills (optional)

- 3,050 job descriptions from 2 sources, **no duplicates**, 11 required skills per job on average (1 to 30).
- Labels were generated by **Gemini 2.5 Flash Lite** (stated in its README) → silver quality.
- Top skills are realistic (Python 2,256, SQL 1,908, R 1,178, Machine Learning 639, Tableau 409, Spark 312…), but
  spelling is not normalised ("Machine Learning" 639 vs "machine learning" 466).

**Decision:** optional — possible reference for job-requirement extraction or fine-tuning data; not needed for the
core pipeline.

### 3.8 In-repo data

- **Synthetic CVs (4) and job descriptions (2)**: all valid against our Pydantic schemas (`CandidateProfile`,
  `JobDescription`). The **3 matching pairs** cover a strong match (94), a hard negative (32) and a partial match (68),
  with a written rationale. Good seeds for the hand-made matching set, but far too few.
- **HR knowledge base**: **9 files, 518 words in total** (career ladder, remote-work policy, screening rules, two job
  profiles, skill definitions). Two folders overlap (`skills/` vs `skill_definitions/`, `policies/` vs `hr_policies/`).
  Too small for a meaningful RAG evaluation.
- **Sample files** (`raw/**/sample_*.json`) and **`evaluation/`** files are **hand-written placeholders**, not real
  data. Example: `esco/skills_sample.json` lists "Deep Learning" as a synonym of "Machine Learning", whereas real ESCO
  has two separate concepts; `matching_benchmark_v1.json` announces 50 pairs but contains none.

---

## 4. Decision table

| Dataset / files | Trust labels? | Main problems | Use |
|---|---|---|---|
| Resume Parser — test | Silver (96 % grounded, likely LLM-made) | Noisy categories, template identities, 2 corrupted texts | **Extraction reference** after filtering + hand-check |
| Resume Parser — validation / train | Silver | Duplicates; 126 test texts also in train | Prompt tuning (validation); fine-tuning only (train) |
| Resume Parser — PDFs | Labels match images | All scanned; some low resolution | **Multimodal extraction test** (OCR vs Qwen-VL) |
| Job Matching | **No** | Implausible scores, off-domain jobs | **Not used** |
| Tech Jobs 2026 | n/a | Duplicates, boilerplate | **5 job descriptions + RAG** |
| ESCO (EN + FR) | n/a | Misses 44 % of our tech terms | Skill normalisation + **custom extension** |
| Skill Extractor | Silver (Gemini) | Unnormalised spelling | Optional |
| Synthetic CVs / jobs / pairs | Valid | Only 4 / 2 / 3 items | **Seeds** for the hand-made matching set |
| HR knowledge base | n/a | 518 words only | **Must be extended** for RAG |
| Sample / evaluation files | Placeholders | Not real data | Replace or delete |

---

## 5. Consequences for the evaluation set (next steps)

1. **Extraction set**: start from the 169 tech CVs of the test split, then remove
   - the 18 CVs whose text also appears in train (leakage),
   - the 2 corrupted texts,
   - CVs that are not actually tech profiles (manual check of job titles).
   Download all remaining PDFs and **hand-check about 20 labels** against the images.
2. **Personal-information fields** are scored only on CVs that do not use the template identity.
3. **Job descriptions**: deduplicate Tech Jobs 2026 and pick 5 postings, one per role (ML Engineer, NLP/LLM Engineer,
   Data Analyst/Scientist, Backend Developer, DevOps/Cloud).
4. **Matching set**: about 30 CV–job pairs labelled by hand (grade 0–3 + reason), mixing real tech CVs with about 10
   synthetic ones, including French CVs and hard cases (keyword stuffing, synonyms, career change).
5. **Text PDFs and DOCX**: generated from the synthetic CVs, so that both scanned and digital documents are tested.
6. **ESCO**: add a small custom list for missing tools (Docker, Kubernetes, AWS, Azure, PyTorch, React, FastAPI…).
7. **Knowledge base**: write more HR documents (screening criteria per role, skill definitions, interview guidelines).
8. **Freeze** the result into a dev / test split under `data/processed/eval_v1/`.

### Open items

- The image-vs-label check covered 3 CVs; it should be extended to at least 20 during step 1 above.
- The share of non-tech CVs inside tech categories is unknown (2 of 5 in our sample); it will be measured during the
  manual filtering.
