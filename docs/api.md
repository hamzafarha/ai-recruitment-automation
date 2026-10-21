# REST API

FastAPI backend of the interface, of the n8n workflows (the application pipeline) and of the AI assistant.
Start it with `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`; interactive documentation (try every
endpoint): **http://localhost:8000/docs**. Base path: `/api/v1`. Errors are JSON `{"detail": "..."}`: 400 invalid
input, 404 unknown id, 409 an action out of turn (a pipeline step before its time), 422 validation or unreadable
document/model answer, 503 a service is unreachable (Kaggle AI server stopped, Qdrant down).

## Health and administration

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | Liveness: `{"status": "ok"}` |
| GET | `/api/v1/health` | PostgreSQL, Qdrant, the Kaggle AI server (when used) and n8n (when enabled) |
| POST | `/admin/setup` | Index the HR knowledge base (RAG) and load the demo jobs and candidates (`{"demo_data": true}`); idempotent |
| POST | `/admin/reindex` | Rebuild the search entries of every job and candidate |
| GET | `/admin/status` | Providers, models, Qdrant collection sizes, AI server info, n8n settings |
| GET | `/dashboard` | Figures of the dashboard (counts, grades, statuses, recent analyses and notifications) |

## Application pipeline (every part: one workflow from the CV to the final decision)

Steps, in order: `intake` → `document` → `extract` → `store` → `acknowledge` → `screen` → `report` →
`notify_recruiters` → `validation` (waits for the recruiter) → `decision` → `notify_candidate` → `close`.

| Method | Path | What it does |
|---|---|---|
| POST | `/applications` | Intake checks and registration: `{"filename", "content_base64", "source": "upload" \| "portal" \| "email" \| "n8n", "consent", "job_id"?}` (no `job_id` = spontaneous application; a portal application needs `consent: true`). The caller (the n8n workflow) then runs the steps |
| POST | `/applications/{id}/steps/{step}` | Run one step. Body: `{"content_base64"}` for `document` (the file is never stored), `{"resume_url"}` for `notify_recruiters` (the n8n Wait node's URL, called back when the recruiter decides). 409 if it is not the next step; a step already done answers its stored result (`already_done`); `decision` answers `{"decided": false}` while the recruiter has not decided |
| POST | `/applications/submit` | Intake, then the API runs every step itself in the background (202): when n8n is stopped, or from the interface |
| POST | `/applications/{id}/continue` | Continue with the API from the next (or failed) step |
| GET | `/applications?status=` · `/applications/{id}` | Applications (running, waiting_review, closed, failed) and the log of every step: status, timings, result (the screening agent's timeline and tool calls), error |
| GET | `/applications/steps` | The steps, with the part of the architecture each one implements |
| GET | `/applications/{id}/outputs` | What every step produced, in full: CV text, structured profile, Qdrant points, e-mails, the screening agent's tool calls with their results and the HR documents retrieved, reports, reviews |
| POST | `/applications/via-n8n` | Same body as `/applications`: the CV is sent to the n8n workflow's webhook (`cv-intake`), which runs every step (the interface cannot call n8n itself) |

A recruiter's decision (`POST /analyses/{id}/review`) on an analysis of the pipeline resumes its workflow: n8n
(its Wait node) or, if n8n does not answer, the API runs the final decision, the candidate's message and closes.

## Candidates (DSO1)

| Method | Path | What it does |
|---|---|---|
| GET | `/candidates?q=` | List (filter on name or location) |
| POST | `/candidates` | Import a CV without screening it (multipart `file`: PDF, DOCX, PNG, JPG): extraction by Qwen + adapter, storage, search index. The interface sends CVs into the application pipeline instead |
| POST | `/candidates/extract` | Extraction only, nothing stored (to test DSO1) |
| GET | `/candidates/{id}` | Profile, extraction details and the candidate's analyses |
| GET | `/candidates/{id}/cv-text` | The CV text used as evidence |
| POST · GET | `/candidates/{id}/compare-models` | Base Qwen2.5 (no adapter) on the stored CV text next to the fine-tuned profile (POST runs it, about a minute; GET returns the stored comparison) |
| DELETE | `/candidates/{id}` | Delete the candidate, their analyses, applications, e-mails and search entries (right to erasure) |

## Jobs

| Method | Path | What it does |
|---|---|---|
| GET | `/jobs` | List |
| POST | `/jobs` | Create: title, company, location, employment type, role, description, requirements (`requirement_text`, `requirement_type` required/preferred, `importance` high/medium/low, `min_years_experience`) |
| POST | `/jobs/extract` | Job advertisement text → structured requirements (Qwen), not stored |
| GET / PATCH / DELETE | `/jobs/{id}` | Read, open/close (`{"status": "closed"}`), delete |
| GET | `/jobs/{id}/ranking` | Every candidate graded for the job in seconds (fast semantic matching + rubric) |

## Analyses, reports and human validation (DSO3, parts 6 and 9)

| Method | Path | What it does |
|---|---|---|
| POST | `/analyses` | `{"candidate_id", "job_id", "method": "llm" \| "semantic"}`: evidence-backed analysis (the Matching page's fixed analysis or the fast estimate), stored as *pending review*. The pipeline's analyses come from its screening agent (`method: "agent"`) |
| GET | `/analyses?job_id=&candidate_id=&status=` | List |
| GET / DELETE | `/analyses/{id}` | The full report (requirements, evidence, gaps, questions, knowledge sources) and its reviews |
| GET | `/analyses/{id}/report.md` · `/report.pdf` | Download the report |
| POST | `/analyses/{id}/review` | Recruiter decision: `{"decision": "shortlist" \| "interview" \| "hold" \| "reject", "reviewer", "adjusted_grade", "interview_at", "notes"}` (a rejection needs notes). For a pipeline analysis it resumes the application's workflow (the answer shows the application's status) |

## AI assistant and search

| Method | Path | What it does |
|---|---|---|
| POST | `/agent/ask` | `{"question": "..."}` → answer, tool steps (with their data), sources, timings |
| GET | `/agent/runs` · `/agent/runs/{id}` | Stored questions and traces |
| GET | `/agent/tools` | The assistant's tools |
| GET | `/search/knowledge?q=&role=&top_k=` | HR knowledge base (BGE-M3 + Qdrant + reranker) |
| GET | `/search/candidates?q=` · `/search/jobs?q=` | Candidates' CV evidence, job descriptions |

## Notifications and n8n

| Method | Path | What it does |
|---|---|---|
| GET | `/notifications?unread_only=` | Notifications (app events and n8n messages) |
| POST | `/notifications` | Used by n8n: `{"event", "title", "message", "payload", "source": "n8n"}` |
| POST | `/notifications/{id}/read` · `/notifications/read-all` | Mark as read |
| POST | `/webhooks/n8n/job` | A job advertisement: `{"text", "title"?, "company"?}` → structured and stored |
| GET | `/webhooks/n8n/digest` | The daily recruiting digest (pending reviews, service levels) |

A CV received by n8n goes into the application pipeline (above). With `N8N_EVENTS_ENABLED=true`, the other
notifications created by the app are also posted to n8n (`{N8N_URL}/webhook/recruitment-events`); the pipeline's
own messages are written by its workflow.

## E-mails and interview meetings (part 7)

| Method | Path | What it does |
|---|---|---|
| GET | `/emails/settings` | How e-mails are sent (`EMAIL_MODE` smtp or outbox, redirection) and how meetings are created (Google Meet or Jitsi) |
| GET | `/emails?application_id=&status=&kind=` | E-mails to the candidates (acknowledgement, decision), with counts per status |
| GET | `/emails/{id}` | One e-mail: text, HTML, meeting, calendar invite (`ics`) |
| POST | `/emails/{id}/send` | Send again with the current settings (an e-mail of the outbox, or a failed one); 409 if already sent |

The pipeline's steps `acknowledge` and `notify_candidate` write them; an interview decision creates the meeting.

## HR knowledge base (part 6)

| Method | Path | What it does |
|---|---|---|
| GET | `/knowledge/documents` | The 15 documents the RAG retrieves from, with their sections and chunks |
| GET | `/knowledge/documents/{doc_id}` | One document (e.g. `KB-RR-002`): markdown and the chunks indexed in Qdrant |

## Evaluation

| Method | Path | What it does |
|---|---|---|
| GET | `/evaluation/overview` | What each evaluation measures, with which technologies, and its headline number |
| GET | `/evaluation/extraction?before=&after=` | CV extraction, base vs fine-tuned (or any two runs): metrics, groups, per CV |
| GET | `/evaluation/finetune` · `/evaluation/rag` · `/evaluation/matching` · `/evaluation/assistant` | The other evaluations of data/evaluation/ |
| GET | `/evaluation/pipeline` | The live pipeline: timings and outcomes of every step |
| GET | `/evaluation/mlflow` | MLflow installed / interface running / experiments (scripts/mlflow_sync.py logs them) |

The React interface is served at `/` (any path that is not `/api/...`, `/health` or `/docs`).
