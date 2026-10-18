# AI Recruitment Automation

AI-assisted CV screening (Advanced Deep Learning project). Every application goes through **one workflow, from
the CV to the final decision** (n8n): the system reads the CV (digital PDF, scanned PDF, Word), extracts a
structured profile with our fine-tuned Qwen, then a LangGraph **screening agent** retrieves the CV evidence and
the company's HR rules (RAG) for every job requirement and writes an evidence-backed report; the recruiters are
notified, the workflow waits for their decision, then answers the candidate. **A human recruiter always makes
the decision.**

Stack: Python 3.12 · Qwen2.5 / Qwen2.5-VL via Hugging Face Transformers · PaddleOCR models (RapidOCR) · BGE-M3 embeddings ·
BGE-reranker-v2-m3 · Qdrant · PostgreSQL · FastAPI · Streamlit · LangGraph · n8n · Kaggle (free GPUs) for every model run.

**Project status:** complete — every part of the architecture is built and usable from the interface, chained
in the application pipeline: intake, document processing, CV extraction with our QLoRA fine-tuned Qwen,
storage and search index, the screening agent (RAG over the CV and the HR knowledge base, reranking,
iterations, evidence-checked analysis), reports, notifications, human validation and the answer to the
candidate. Plus a LangGraph AI assistant with tools. What was done, the results and a guided tour:
**[final report](docs/reports/final_report.md)**. To try it: [section 8](#8-try-everything).

---

## 1. Prerequisites

Install these once:

| Tool | Why | Link |
|---|---|---|
| **Python 3.12** (exactly 3.12) | Project language | https://www.python.org/downloads/ |
| **Git** | Clone the repo | https://git-scm.com/downloads |
| **Docker Desktop** | Runs PostgreSQL, Qdrant and n8n | https://www.docker.com/products/docker-desktop/ |
| **Kaggle account** | Runs the AI models on free GPUs | https://www.kaggle.com |

> Commands below are for **Windows PowerShell**, run from the project root. On macOS/Linux, activate the
> environment with `source .venv/bin/activate` and use `cp` instead of `Copy-Item`.

---

## 2. Install

```powershell
git clone https://github.com/hamzafarha/ai-recruitment-automation.git
cd ai-recruitment-automation

python -m venv .venv
.\.venv\Scripts\Activate.ps1          # your prompt now starts with (.venv)
python -m pip install --upgrade pip
```

Install PyTorch **before** the project. The CPU version is enough (the models run on Kaggle), and
`torchvision` must come from the same place as `torch`:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then the project and all its dependencies:

```powershell
pip install -e ".[all]"
```

> **Always run tools as `python -m <tool>`** (`python -m pytest`, `python -m uvicorn`, …). On Windows the
> `.venv\Scripts\*.exe` shortcuts are sometimes missing and PowerShell then silently uses another Python.

---

## 3. Configure

```powershell
Copy-Item .env.example .env
```

The defaults work as-is: PostgreSQL on port **5433** (not 5432, to avoid clashing with a PostgreSQL installed
on Windows), Qdrant on 6333, and `LLM_PROVIDER=mock` (fixed fake LLM answers).

**The AI models do not run on our computers.** Qwen2.5-7B, Qwen2.5-VL-7B, BGE-M3 and the reranker run on
Kaggle's free GPUs (section 9): nothing to install for them, and no RAM or GPU needed on your PC.

---

## 4. Start the services and the database

Open **Docker Desktop** first and wait until it says *Engine running*. Then, from the project root:

```powershell
docker compose up -d postgres qdrant n8n     # first time: downloads the images (~1 GB), then starts them
python -m alembic upgrade head               # creates/updates the database tables (the API also does it when it starts)
```

| Service | Container | URL / port |
|---|---|---|
| PostgreSQL | `recruitment_postgres` | `localhost:5433` (browse it: `docker compose --profile tools up -d adminer` → http://localhost:8080) |
| Qdrant | `recruitment_qdrant` | http://localhost:6333/dashboard |
| n8n | `recruitment_n8n` | http://localhost:5678 (create a local account on first visit) |

Everyday Docker commands (from the project root):

```powershell
docker compose ps                             # what is running (STATUS should be "Up" / "healthy")
docker compose logs -f qdrant                 # follow a service's log (Ctrl+C to quit); also postgres, n8n
docker compose stop                           # stop everything (data is kept)
docker compose up -d postgres qdrant n8n      # start again
docker compose restart postgres               # restart one service
docker compose down                           # stop and remove the containers (data is kept in volumes)
```

> `docker compose down -v` also **deletes all data** (database, vector index, n8n workflows). Only use it to
> start from zero, then run `python -m alembic upgrade head` again.

The API and the interface run with Python (section 8). They can also run in Docker
(`docker compose --profile app up -d --build`: API on :8000, interface on :8501, settings from `.env`).

---

## 5. Download the datasets (once)

The datasets contain real CVs: they stay on your machine in `hf/` folders, which **git ignores. Never commit
them.**

```powershell
mkdir -Force data\raw\resumes\resume_parser\hf\curated, data\raw\job_descriptions\tech_jobs_2026\hf, data\raw\ontologies\esco\hf

$HF = "https://huggingface.co/datasets"
curl.exe -L -o data\raw\resumes\resume_parser\hf\curated\test.jsonl       "$HF/md-zain/resume-parser-dataset/resolve/main/curated/test.jsonl"
curl.exe -L -o data\raw\resumes\resume_parser\hf\curated\validation.jsonl "$HF/md-zain/resume-parser-dataset/resolve/main/curated/validation.jsonl"
curl.exe -L -o data\raw\resumes\resume_parser\hf\curated\train.jsonl      "$HF/md-zain/resume-parser-dataset/resolve/main/curated/train.jsonl"
curl.exe -L -o data\raw\job_descriptions\tech_jobs_2026\hf\jobs.json       "$HF/Sundaydream/tech-jobs-dataset-2026/resolve/main/jobs.json"
```

Use `curl.exe`, not `curl` (in Windows PowerShell `curl` is a different command). `train.jsonl` (137 MB) is
needed to detect CVs duplicated between splits.

**ESCO** (skill dictionary, manual download): https://esco.ec.europa.eu/en/use-esco/download → version
**v1.2.1**, **Classification**, **CSV**, once in **English** and once in **French** (download links arrive by
e-mail). Unzip into `data\raw\ontologies\esco\hf\en\` and `data\raw\ontologies\esco\hf\fr\`.

---

## 6. Prepare the data and the evaluation set

```powershell
python scripts/prepare_data.py               # converts labels, cleans job postings, builds the EN/FR skill lexicon (~1-2 min)
python scripts/build_eval_set.py pool        # downloads the 192 evaluation CV PDFs from Hugging Face (~2-5 min)
python scripts/build_eval_set.py verify      # must print: OK: every file matches the frozen manifest
```

The evaluation set itself (`data/processed/eval_v1/`: CV ids, job descriptions, synthetic CVs, labels,
checksums) is **already in the repo and frozen**. Its datasheet: [data/processed/eval_v1/README.md](data/processed/eval_v1/README.md).

> Do **not** run `build_eval_set.py pairs --force` or `freeze`: they rebuild the frozen evaluation set, and
> results would no longer be comparable with the team's.

---

## 7. Check that everything works

```powershell
python -m pytest -q                  # unit tests (no GPU, no Docker needed), all must pass
python -m ruff check src scripts tests
python scripts/health_check.py       # real services: PostgreSQL, Qdrant, API (models are checked on Kaggle)
```

---

## 8. Try everything

**What runs where.** On your PC: Docker (PostgreSQL, Qdrant, n8n), the API and the interface — all light. On
Kaggle's free GPUs: every model (Qwen2.5-7B + our fine-tuned adapter, BGE-M3, BGE reranker), in one notebook
called the **AI server**. The app sends its model requests to it through a private link.

### Step 1 — Start the AI server on Kaggle (each time, ~5 min)

1. Once, and again after you change the code: `python scripts/kaggle_run.py push server`. It sends the code to
   your private Kaggle dataset and creates the notebook **recruitment-ai-model-server** with the code and the
   fine-tuned adapter attached. (The automatic run that a push starts stops at once: that is expected.)
2. kaggle.com → *Your Work* → *Code* → **recruitment-ai-model-server** → **Edit**. Right panel: **GPU T4 x2**,
   **Internet on**. Then **Run All**. After ~5 minutes the 4th cell prints four lines:
   `LLM_PROVIDER=remote`, `EMBEDDING_PROVIDER=remote`, `REMOTE_LLM_URL=…`, `REMOTE_LLM_TOKEN=…`.
3. Paste them into `.env` (replace the previous values). The link and token change at every session.
4. **When you are done, stop the Kaggle session** (power icon): the GPU quota counts while it is open.

### Step 2 — Start the services, the API and the interface

```powershell
docker compose up -d postgres qdrant n8n
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000     # terminal 1 — API docs: http://localhost:8000/docs
python -m streamlit run frontend/streamlit_app.py              # terminal 2 — interface: http://localhost:8501
```

`--host 0.0.0.0` lets n8n (in Docker) call the API; if Windows asks, allow Python on **private** networks.
The API creates or updates the database tables when it starts. Restart the API after every `.env` change.

### Step 3 — Load the data (once)

Interface → **Dashboard** → *Set up: index the HR knowledge + load the demo data* (~1 min). It indexes the 15 HR
documents in Qdrant (RAG) and loads the 5 real job postings and the 12 invented candidates of the evaluation
set. Running it again adds nothing twice.

### Step 4 — n8n workflows (once)

```powershell
python scripts/n8n_setup.py        # imports and activates the 5 workflows of n8n/workflows/, sends a test event
```

The main one is **Recruitment — Application pipeline**: one execution per application, from the CV to the
final decision (it replaces the former "CV intake" workflow; re-running the script replaces it). Then set
`N8N_EVENTS_ENABLED=true` in `.env` and restart the API. The n8n editor is at http://localhost:5678
(the first visit asks you to create the local owner account).

### Guided tour

| # | Page | Do | Architecture part |
|---|---|---|---|
| 1 | **Candidate portal** | Apply with a CV *through n8n*, for a job or spontaneously: the answer comes after the extraction | 1, 2, 3, 4, 8, 7 |
| 2 | **Application pipeline** | Watch the application go through every step: the screening agent's path (tools, searches, rounds, checks), then it waits | 5, 6, 7 |
| 3 | **Human validation** | Decide (shortlist, interview with a date, hold, reject with a reason): the workflow resumes, answers the candidate and closes | 9, 7 |
| 4 | **Reports** | Open the analysis: requirement-by-requirement evidence, the agent's trace; download the PDF | 6 |
| 5 | **Candidates** → Add | The same workflow from the recruiter's side (any CV: PDF, Word or image) | 1, 3, 4 |
| 6 | **Jobs** → New job from an advertisement | Paste a job ad, check the requirements, save | 1, 6 |
| 7 | **Matching** | Choose a job: all candidates ranked in seconds; analyse the best ones (fixed analysis) | 5 |
| 8 | **Notifications & n8n** | The messages prepared for the candidates; run the daily digest | 2, 7 |
| 9 | **AI assistant** | Click an example question; open "How it answered" to see the tools | 5 (LangGraph assistant) |
| 10 | **Search** | Ask the HR knowledge base (EN/FR), search CVs and jobs | 5, 6 |
| 11 | **Evaluation** | Results of every evaluation (extraction before/after fine-tuning, RAG, matching, screening agent, assistant) | — |

Only use CVs you are allowed to process (your own, or invented ones): their text goes to your Kaggle session.
Without the AI server (`LLM_PROVIDER=mock`) the pages work but the AI answers are fake.

**Observe every part while it works** — the Application pipeline page (every step of every application, the
screening agent's path), the app's *Under the hood* pages (architecture live, document processing and fine-tuned
vs base model, vectors and reranking, matching internals, the assistant's path, the database), plus
Adminer/psql, the Qdrant dashboard and the n8n runs: [step-by-step guide](docs/reports/observe_the_data.md).

Other app: `python -m streamlit run frontend/labeling_app.py` (optional human labeling of the evaluation pairs).

---

## 9. Run the evaluations

Every evaluation runs on **Kaggle's free GPUs** ("GPU T4 x2": two 16 GB cards). Free quota: **30 GPU hours
per week** per account, renewed weekly.

One-time setup: Kaggle → *Settings* → *API Tokens* → **Generate New Token**, then save it (never share or
commit it):

```powershell
mkdir -Force $env:USERPROFILE\.kaggle
Set-Content -NoNewline -Path $env:USERPROFILE\.kaggle\access_token -Value "<your token>"
python -m kaggle config view                 # shows your username if the token works
```

Your Kaggle account needs a **verified phone number** (GPU and internet access in notebooks).

```powershell
python -m kaggle quota                                        # GPU hours left this week
python scripts/kaggle_run.py all rag                          # Step 3: retrieval evaluation (~15 min)
python scripts/kaggle_run.py all extraction extraction-text   # Step 2: CV extraction, 2 jobs in parallel (~1-3 h)
python scripts/kaggle_run.py all extraction-vlm               # Step 2: Qwen2.5-VL reads the page images (~2-3 h)
python scripts/kaggle_run.py all matching                     # DSO3: matching vs the 36 graded pairs (~1 h)
python scripts/kaggle_run.py all screening                    # DSO4: the screening agent on the same pairs (~1 h)
python scripts/kaggle_run.py all agent                        # the AI assistant on 20 questions (~30 min)
python scripts/kaggle_run.py status extraction extraction-text
python scripts/kaggle_run.py fetch extraction extraction-text # download the reports if you stopped waiting
```

`all` packages the code and the evaluation set into a **private** Kaggle dataset, starts the jobs, waits
and downloads the reports into `data/evaluation/<folder>/`. CV PDFs are not uploaded: the jobs download them
from the public Hugging Face dataset and check them against the frozen checksums. Follow a running job
on kaggle.com → *View Active Events*.

The extraction jobs (one per engine: `extraction` = OCR + Qwen, the real pipeline; `extraction-text` = Qwen on
the dataset's clean text; `extraction-vlm` = Qwen2.5-VL) run the 7B models in float16, split across the two
GPUs. As a safety net for the quota, a job stops starting new CVs after 4 hours; the CVs are ordered so that
a partial run keeps the mix of the whole split, and the report says how many were processed.

The `matching` job calibrates the fast matcher's two thresholds on the dev pairs and writes them to
`data/evaluation/matching/thresholds.json`, which the app then uses. The `screening` job runs the application
pipeline's screening agent on the same pairs and adds it to the matching report, next to the fixed analysis
(whose saved answers it reuses). The `agent` job builds a fresh workspace
(SQLite, embedded Qdrant, demo data) and asks the questions of `data/processed/agent_eval/scenarios.json`.

Reports: `data/evaluation/<extraction/run | rag | matching | agent>/report.md`, also shown on the
**Evaluation** page of the interface.

### Fine-tuning (Step 9, QLoRA on Kaggle)

The notebook [notebooks/kaggle/finetune_qwen_qlora.ipynb](notebooks/kaggle/finetune_qwen_qlora.ipynb) teaches
Qwen2.5-7B our CV format with QLoRA, on ~1,000 CVs of the Resume Parser **train** split (evaluation CVs are
excluded). It produces a LoRA adapter (~160 MB). Run the evaluation first: its score is the "before" number.

```powershell
python scripts/kaggle_run.py all finetune         # training (~2-3 GPU hours on 2x T4 for 1,000 CVs)
python scripts/kaggle_run.py all extraction-ft    # the same evaluation as `extraction`, with the adapter
```

Built for Kaggle's limits: every cell ends with a "saved and exported" line; a checkpoint every 15 minutes;
training stops by itself 40 minutes before a 7-hour session budget, so outputs are always written; running it
again resumes from the last checkpoint. To change a setting (number of CVs, session budget…), open the
notebook on kaggle.com, edit its first cell (*PARAMETERS*), then *Save Version → Save & Run All*. Its
header explains the settings, the optional Hugging Face backup and how to resume.

---

## 10. Project structure

```text
src/recruitment_ai/
  ingestion/         CV readers: PDF (PyMuPDF), DOCX, OCR for scans (PaddleOCR PP-OCRv5 via RapidOCR)
  extraction/        CV extractor (Qwen + fine-tuned adapter), job-advertisement extractor, Pydantic schemas
  llm/               Hugging Face Transformers adapter (Qwen2.5, Qwen2.5-VL), remote adapter + the Kaggle AI server
  embeddings/ reranking/  BGE-M3 and the BGE reranker (local, or remote on the Kaggle AI server)
  rag/               chunking, HR knowledge index, search index of candidates and jobs (Qdrant)
  matching/          rubric KB-RR-002 as code, experience rules, fast semantic matcher, LLM evidence analyzer,
                     the CV as searchable evidence (cv_index.py)
  agents/ tools/     the LangGraph screening agent of the pipeline; the AI assistant and its 8 tools
  reports/           evidence-backed report as Markdown and PDF
  services/          use cases: the application pipeline (pipeline.py), candidates, jobs, matching, reviews,
                     events/notifications, dashboard, demo, assistant
  database/          PostgreSQL tables (applications, candidates, jobs, analyses, reviews, notifications,
                     agent_runs) + repositories
  data/              dataset preparation, EN/FR skill lexicon (ESCO + custom), knowledge-base loader
  evaluation/        evaluation set, extraction / retrieval / matching / assistant metrics, before-after comparison
  training/          fine-tuning: training examples (data.py) and QLoRA training (qlora.py)
api/                 FastAPI app: routers for every feature (docs at /docs)
frontend/            Streamlit interface (streamlit_app.py + app_pages/), labeling app
n8n/workflows/       the 5 n8n workflows (application pipeline, new job, events, daily digest, error alert)
scripts/             prepare_data, build_eval_set, eval_*, finetune_qlora, kaggle_run, n8n_setup, health_check
scripts/kaggle/      the jobs that run on Kaggle
notebooks/kaggle/    the fine-tuning notebook and the AI server notebook (run on Kaggle)
data/knowledge_base/ 15 HR documents used by RAG (rubric, job profiles, policies…; 3 in French)
data/processed/eval_v1/  frozen evaluation set (committed; no CV content)
data/reference/      custom skills added to ESCO
data/raw/, data/interim/, data/evaluation/**/predictions/   git-ignored (real CVs, personal data)
api/  frontend/  tests/  docs/  notebooks/  alembic/  configs/
```

Reports: **[final report](docs/reports/final_report.md)** · [end-to-end test plan](docs/reports/test_plan.md) ·
[observe the data](docs/reports/observe_the_data.md) · [validation day guide](docs/reports/validation_guide.md) ·
[API](docs/api.md) ·
[data inspection report](docs/reports/data_inspection_report.md) · [data layer](docs/data.md).

---

## 11. Rules for the team

- **Never commit personal data**: anything in `data/raw/**/hf/`, `data/interim/`, `build/`,
  `data/evaluation/**/predictions/`, or your `.env`. Check `git status` before every commit.
- **Clear notebook outputs before committing** (VS Code: `⋯` → *Clear All Outputs*): notebooks can show real CVs.
- **Do not modify `data/processed/eval_v1/`**: it is the frozen exam; `build_eval_set.py verify` detects changes.
- Tune prompts and settings on the **dev** split; report final numbers on **test** only.

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'recruitment_ai'`, or `pytest`/`uvicorn`/`streamlit` not found | Activate the venv and use `python -m pytest`, `python -m uvicorn`, … |
| PostgreSQL password error mentioning port 5432 | A Windows PostgreSQL service owns 5432; the project uses **5433**: keep `POSTGRES_PORT=5433` in `.env` |
| Qdrant `SSL: WRONG_VERSION_NUMBER` | Leave `QDRANT_API_KEY=` empty (no value) in `.env` |
| `failed to connect to the docker API` | Start Docker Desktop, then `docker compose up -d postgres qdrant n8n` |
| Tests very slow (minutes) | Docker stopped is fine for unit tests; make sure you run the latest code (`git pull`) |
| `PermissionError` writing a PDF report | Close the PDF in Adobe Acrobat (the whole window), then retry |
| `verify` reports changed files | Re-download the datasets (section 5) and run `prepare_data.py` and `build_eval_set.py pool` again |
| Kaggle push fails with a GPU / quota message | `python -m kaggle quota`: wait for the weekly renewal, or run fewer jobs at once |
| `pip install -e ".[all]"` replaces your torch | Install `torch torchvision` first from the PyTorch index (section 2), then the project |
| Interface: *The Kaggle AI server does not answer* | The Kaggle session stopped or its link changed: run the notebook again, paste the new lines into `.env`, restart the API |
| *The AI server refused the token* | Copy `REMOTE_LLM_TOKEN` again from the notebook (no spaces), restart the API |
| Portal: *n8n answered 404* | The workflows are not imported or not active: `python scripts/n8n_setup.py` |
| n8n cannot reach the API (workflow error) | Start the API with `--host 0.0.0.0` and allow Python through the Windows firewall (private networks) |
| Search returns nothing | Dashboard → *Set up* (indexes the knowledge base, jobs and candidates); after a Qdrant reset → *Re-index* |
