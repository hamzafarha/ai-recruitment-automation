# Architecture

The complete description, with results and a guided tour, is the [final report](reports/final_report.md).
This page is the short technical map.

## One workflow per application: from the CV to the final decision

Every CV (candidate portal, recruiter upload) goes through the same workflow, `services/pipeline.py`. n8n runs
it (workflow *Recruitment — Application pipeline*, one node per step, one execution per application); when n8n is
stopped, the API runs the same steps itself. Each step is logged on the application (Application pipeline page).

```text
 1 Intake checks ─► 3 Document processing ─► 3-4 Candidate JSON ─► 8 Store & index ─► 7 Acknowledge the candidate
   consent,          PyMuPDF · python-docx     Qwen2.5-7B + our       PostgreSQL +        (the portal gets its
   type, size        · OCR (PaddleOCR)         QLoRA adapter          Qdrant (BGE-M3)      answer here)
 ─► 5 Screening agent (LangGraph) ─► 6 Evidence-backed report ─► 7 Notify the recruiters
 ─► 9 Human validation: the workflow WAITS (n8n Wait node) until a recruiter decides
 ─► 9 Final decision ─► 7 Notify the candidate (interview invitation / answer) ─► 8 Close
```

Fine-tuning is not a step: it ran once on Kaggle (QLoRA on 873 CVs); the *Candidate JSON* step uses the adapter
it produced. A failed step is recorded and notified; the workflow continues from it once fixed.

### The screening agent (`agents/screening.py`)

The workflow gives it its task (the job applied to, or — spontaneous application — the open jobs that fit) and
it returns one evidence-backed report per job. Its LangGraph graph follows the architecture's seven agent steps:

```text
START ─► understand ─► plan ─► act ─► evaluate ──(weak must-haves, rounds left)──► plan
                                          └──(evidence enough)──► analyse ─► verify ─► next_job ─► (plan | END)
```

| Node | Diagram step | What it does |
|---|---|---|
| understand | 1. Understand the task | the job applied to, or the open jobs graded by the fast matching (grade ≥ 2, at most 2; else the best) |
| plan | 2. Decide which tools to use | round 0 by code: every requirement searched in the CV + the job's HR rules; then Qwen chooses further searches for the weak requirements (CV with other words, HR equivalences), at most 4 per round, 2 rounds |
| act | 3-4. Call the tools, retrieve | CV search (`matching/cv_index.py`: the CV text as passages tagged with their section, BGE-M3 + BGE reranker), HR knowledge search (Qdrant) |
| evaluate | 5-6. Evaluate & rerank, iterate | the best 3 passages per requirement; must-haves still weak → plan again |
| analyse | 7. Final analysis | Qwen2.5-7B (base model) judges every requirement from its passages and the CV |
| verify | 7. Checks by code | quote in the CV; quote supports the requirement (cross-encoder); only in a skills section → "listed"; strong evidence the model left out; years from the positions; rubric KB-RR-002 → grade, score, action |

Guards by code: the systematic searches always run, no repeated search, no search on protected attributes, a
failed model call degrades (default searches, or the fast semantic estimate).

## Other flows

```text
Job ad ──► extraction/job_extractor.py (Qwen, base model) ──► requirements M1.. / N1.. ──► PostgreSQL + Qdrant (jobs)
Ranking ──► matching/semantic.py: BGE-M3 retrieval of CV snippets, BGE reranker, lexicon, rubric (no LLM)
Matching page ──► matching/analyzer.py: the fixed analysis (HR rules retrieved + one Qwen judgement + checks)
Questions ──► agents/agent.py: the AI assistant (LangGraph decide → act → respond) with tools/registry.py (8 tools)
```

## Layers

| Layer | Package | Notes |
|---|---|---|
| Interface | `frontend/` | Streamlit pages; talks to the API only |
| API | `api/` | FastAPI routers → services; errors mapped to 400/404/409/422/503 |
| Services | `src/recruitment_ai/services/` | use cases (`pipeline.py`: the application workflow); `container.py` wires the configured models and stores |
| AI | `extraction/`, `matching/`, `agents/`, `tools/`, `rag/` | prompts, parsing, retrieval and scoring (same code in the app and the evaluations) |
| Models | `llm/`, `embeddings/`, `reranking/` | local (Transformers, sentence-transformers) or remote (Kaggle AI server, `llm/server.py`) |
| Storage | `database/` (PostgreSQL, Alembic), `vectorstore/` (Qdrant) | 7 tables (`applications` holds each application's step log); SQLite and embedded Qdrant in tests and Kaggle jobs |
| Orchestration | `n8n/workflows/` | the application pipeline, new job, events, daily digest, error alert |
| Evaluation | `evaluation/`, `scripts/eval_*.py`, `scripts/kaggle/` | run on Kaggle by `scripts/kaggle_run.py` |

## Settings that choose where models run

| Setting | Values |
|---|---|
| `LLM_PROVIDER` | `remote` (Kaggle AI server), `huggingface` (Transformers on this machine), `openai`, `mock` |
| `EMBEDDING_PROVIDER` | `remote` (BGE-M3 and the reranker on the Kaggle AI server) or `local` |
| `REMOTE_LLM_URL`, `REMOTE_LLM_TOKEN` | printed by `notebooks/kaggle/model_server.ipynb` at every session |
| `N8N_URL`, `N8N_EVENTS_ENABLED` | n8n address and whether the other app events are forwarded to it |
