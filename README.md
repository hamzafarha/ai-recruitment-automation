# AI Recruitment Automation

AI-assisted CV screening (Advanced Deep Learning project). The system reads CVs (digital PDF, scanned PDF,
Word), extracts structured candidate profiles, retrieves the company's HR rules (RAG), compares candidates
with job requirements and produces evidence-backed reports. **A human recruiter always makes the decision.**

Stack: Python 3.12 · Qwen2.5 / Qwen2.5-VL via Hugging Face Transformers · PaddleOCR models (RapidOCR) · BGE-M3 embeddings ·
BGE-reranker-v2-m3 · Qdrant · PostgreSQL · FastAPI · Streamlit · LangGraph · n8n · Kaggle (free GPUs) for every model run.

**Project status:** Steps 0–3 built (setup, data and frozen evaluation set, CV extraction, RAG over the HR
knowledge base). Next: matching (Step 4).

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
python -m alembic upgrade head               # creates the database tables (once, and after a git pull that adds tables)
```

| Service | Container | URL / port |
|---|---|---|
| PostgreSQL | `recruitment_postgres` | `localhost:5433` |
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

The other services in `docker-compose.yml` (`api`, `frontend`) are not needed for development: run the API
and the UI with Python instead (section 8).

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

## 8. Run the applications

```powershell
python -m uvicorn api.main:app --reload                 # API: http://localhost:8000/docs
python -m streamlit run frontend/streamlit_app.py       # recruiter UI: http://localhost:8501 (second terminal)
python -m streamlit run frontend/labeling_app.py        # optional: human labeling of the evaluation pairs
```

### Try the CV extraction in the app (Qwen on Kaggle)

With `LLM_PROVIDER=mock` the app gives fake answers. To use the real model, Qwen runs in a Kaggle GPU
session and the app on your PC talks to it through a temporary private link:

1. Once: `python scripts/kaggle_run.py upload` (sends the latest code to your Kaggle dataset), then on
   kaggle.com *Create → New Notebook → File → Import Notebook* →
   [notebooks/kaggle/model_server.ipynb](notebooks/kaggle/model_server.ipynb). Settings: **GPU T4 x2**,
   **Internet On**, *Add Input → Your Work → Datasets → recruitment-ai-eval-v1*.
2. Each time: *Run All* in that notebook (~4 min). It prints `REMOTE_LLM_URL=...` and `REMOTE_LLM_TOKEN=...`.
3. Put them in `.env` with `LLM_PROVIDER=remote`, then **restart** the API and the app (commands above).
4. App → *Candidates* → upload a CV → *Extract profile* (about a minute).
5. **Stop the Kaggle session when done**: the GPU quota counts while it is open, even idle.

The document is read on your PC (OCR for scans); only its text goes to your Kaggle session. Use your own
or invented CVs. The link and token change at every session.

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

Reports: `data/evaluation/extraction/<run>/report.md`, `data/evaluation/rag/report.md`.

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
  extraction/        Pydantic schemas (CandidateProfile, JobDescription…) and the LLM CV extractor
  llm/               LLM adapters: Hugging Face Transformers (Qwen2.5, Qwen2.5-VL), OpenAI-compatible, mock
  data/              dataset preparation, EN/FR skill lexicon (ESCO + custom), knowledge-base loader
  rag/               section chunking, knowledge index (BGE-M3 + Qdrant + reranker)
  evaluation/        evaluation set builder, freeze/verify, labeling, extraction and retrieval metrics
  training/          fine-tuning: training examples (data.py) and QLoRA training (qlora.py)
  embeddings/ vectorstore/ reranking/ matching/ agents/ tools/ database/ services/ config/ utils/
scripts/             prepare_data, build_eval_set, eval_extraction, eval_rag, finetune_qlora, kaggle_run, health_check
scripts/kaggle/      the jobs that run on Kaggle
notebooks/kaggle/    the fine-tuning notebook that runs on Kaggle
data/knowledge_base/ 15 HR documents used by RAG (rubric, job profiles, policies…; 3 in French)
data/processed/eval_v1/  frozen evaluation set (committed; no CV content)
data/reference/      custom skills added to ESCO
data/raw/, data/interim/, data/evaluation/**/predictions/   git-ignored (real CVs, personal data)
api/  frontend/  tests/  docs/  notebooks/  alembic/  configs/
```

Reports: [data inspection report](docs/reports/data_inspection_report.md) ·
[data layer](docs/data.md) · [architecture](docs/architecture.md).

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
