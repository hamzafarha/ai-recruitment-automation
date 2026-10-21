# Run guide — AI Recruitment Automation (TalentLens)

How to install, start and use everything: the commands to type, the links to open, and where to go for each
feature. Commands are for **Windows PowerShell**, run from the project folder with the virtual environment
active (`.\.venv\Scripts\Activate.ps1`; your prompt then starts with `(.venv)`).

> **The AI models never run on your PC.** Qwen2.5-7B (+ our QLoRA adapter), BGE-M3 and the BGE reranker run on
> Kaggle's free GPUs (section 6). Without them (`LLM_PROVIDER=mock`) every page works, but AI answers are fixed
> fakes.

---

## 1. What runs where

| What | How it starts | Address |
|---|---|---|
| **Interface** (React) | built once, served by the API | **http://localhost:8000** |
| Interface, development mode (hot reload) | `cd web; npm run dev` | http://localhost:5173 |
| API (FastAPI) | `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` | http://localhost:8000/docs |
| PostgreSQL | Docker | `localhost:5433` (browser: Adminer http://localhost:8080) |
| Qdrant (vectors) | Docker | http://localhost:6333/dashboard |
| n8n (orchestrator) | Docker | http://localhost:5678 |
| MLflow (experiment tracking) | `python -m mlflow ui ...` | http://localhost:5000 |
| AI models (Qwen + adapter, BGE-M3, reranker) | Kaggle notebook *recruitment-ai-model-server* | private link in `.env` |

---

## 2. Install (once)

Prerequisites: **Python 3.12**, **Git**, **Docker Desktop**, **Node.js 20 LTS** (https://nodejs.org — the
interface), a **Kaggle account** with a verified phone number (the models).

```powershell
git clone https://github.com/hamzafarha/ai-recruitment-automation.git
cd ai-recruitment-automation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

`[all]` now also installs **MLflow** (`[tracking]`) and the **Google Calendar** client (`[google]`). If you
installed the project before, run `pip install -e ".[all]"` again.

The interface's packages (once, and after a `git pull` that changes `web/package.json`):

```powershell
cd web
npm install
npm run build        # writes web/dist, which the API serves at http://localhost:8000
cd ..
```

---

## 3. Configure (once)

```powershell
Copy-Item .env.example .env
```

The defaults work as they are. The settings you may change:

| Setting | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, `REMOTE_LLM_URL`, `REMOTE_LLM_TOKEN` | mock / local | paste the 4 lines printed by the Kaggle AI server (section 6) |
| `EMAIL_MODE` | `outbox` | `outbox` = e-mails are written and shown in the app, **not sent**; `smtp` = really sent (section 9) |
| `EMAIL_REDIRECT_TO` | empty | demos: every e-mail goes to this address instead of the candidate's |
| `MEETING_PROVIDER` | `auto` | interview video meetings: Google Meet if authorised (section 10), else Jitsi |
| `INTERVIEW_TIMEZONE`, `INTERVIEW_MINUTES` | Africa/Tunis, 45 | time zone and length of the interviews |
| `N8N_EVENTS_ENABLED` | false | `true` after the n8n setup (section 7) |
| `MLFLOW_TRACKING_URI`, `MLFLOW_UI_URL` | sqlite:///mlruns/mlflow.db, http://localhost:5000 | MLflow store and interface |

**Restart the API after every change of `.env`.**

---

## 4. Start everything (every day)

Open Docker Desktop and wait for *Engine running*. Then, from the project folder:

```powershell
docker compose up -d postgres qdrant n8n
```

Terminal 1 — the API (it also creates/updates the database tables and serves the interface):

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**. The bar at the top shows the services: green = reachable. If Windows asks,
allow Python on **private** networks (n8n, in Docker, calls the API).

Optional — Terminal 2, the interface in development mode (when you change `web/src`; it calls the same API):

```powershell
cd web
npm run dev          # http://localhost:5173
```

Optional — Terminal 3, MLflow (section 11):

```powershell
python scripts/mlflow_sync.py
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

Stop: `Ctrl+C` in each terminal, then `docker compose stop`. Data is kept.

---

## 5. Load the data (once)

Interface → **Dashboard** → *Index the HR knowledge + load the demo* (about a minute). It chunks and embeds the
15 HR documents into Qdrant (RAG), and loads the 5 real job postings and the 12 synthetic candidates of the
evaluation set. Running it again adds nothing twice. *Rebuild search entries* re-indexes after a Qdrant reset.

---

## 6. Connect the real AI models (Kaggle, ~5 minutes per session)

1. Once (and after you change the model code): `python scripts/kaggle_run.py push server`.
2. https://www.kaggle.com → *Your Work* → *Code* → **recruitment-ai-model-server** → **Edit** → right panel:
   **GPU T4 x2**, **Internet on** → **Run All**.
3. After ~5 minutes, the 4th cell prints four lines (`LLM_PROVIDER=remote`, `EMBEDDING_PROVIDER=remote`,
   `REMOTE_LLM_URL=…`, `REMOTE_LLM_TOKEN=…`). Paste them into `.env`, restart the API.
4. Check: the top bar shows **Kaggle AI** in green and **LLM: remote**; Dashboard → *System* shows the model
   and *QLoRA adapter: loaded*.
5. **Stop the Kaggle session when you are done** (power icon): the 30 h/week GPU quota counts while it is open.

---

## 7. n8n workflows (once)

```powershell
python scripts/n8n_setup.py
```

It imports and activates the 5 workflows of `n8n/workflows/`. Then set `N8N_EVENTS_ENABLED=true` in `.env` and
restart the API. n8n: http://localhost:5678 (the first visit asks you to create the local owner account).
Optional: n8n → *Settings* → *n8n API* → create a key, put it in `N8N_API_KEY`: the app then lists the
executions (page *Notifications & n8n*).

The main workflow, **Recruitment — Application pipeline**, runs one execution per application: one node per
step, a *Wait* node for the recruiter's decision. The e-mails are sent by the API (section 9): no SMTP
credentials are needed in n8n. **After this update, run `python scripts/n8n_setup.py` again** (the two former
“Send … (needs SMTP)” nodes were removed from the workflow).

---

## 8. Where to go for each feature

Every page is in the left menu of http://localhost:8000.

| Page (menu) | Address | What you see / do |
|---|---|---|
| **Dashboard** | `/` | Key figures, latest applications, grades given, mean time per step, models in use; the setup button |
| **Application pipeline** | `/pipeline` | Submit a CV (run by the API or by n8n), follow every application through the 12 steps |
| ↳ one application | `/pipeline/<id>` | The live stepper; **click any step to see its full output** (below) |
| **Candidates** | `/candidates` | Every structured CV; open one: profile, CV text, analyses, *Base vs fine-tuned* |
| **Jobs** | `/jobs` | Jobs and requirements; *New job from an advertisement* (Qwen structures a pasted ad) |
| **Matching** | `/matching` | Pick a job: every candidate ranked in seconds; *AI analysis* for the best ones |
| **Reports** | `/reports` | Every evidence-backed report (PDF / Markdown), and the screening agent's trace |
| **Human validation** | `/validation` | Decide: **Schedule interview** (date + time), Shortlist, Hold, Reject (reason) |
| **AI assistant** | `/assistant` | Ask questions; under each answer, *How the agent answered*: tools, arguments, results, sources |
| **Semantic search** | `/search` | Search the HR knowledge base (EN/FR), CVs and jobs |
| **HR knowledge base** | `/knowledge` | The 15 documents the RAG uses, their sections and the chunks indexed |
| **Candidate portal** | `/portal` | What a candidate sees: apply with consent, for a job or spontaneously |
| **E-mails & meetings** | `/outbox` | Every e-mail to a candidate, its status, the interview meeting and calendar invite; *Send now* |
| **Notifications & n8n** | `/notifications` | App events, n8n workflows live or not, n8n executions |
| **Base vs fine-tuned** | `/models` | Qwen2.5 vs Qwen2.5 + QLoRA: evaluation charts, live on a CV, on a stored candidate, training run |
| **Evaluation & MLflow** | `/evaluation` | Every metric of every part (tabs) + MLflow status and link |
| **Architecture (live)** | `/architecture` | The 9 parts of the diagram with live counts; click a part to open its page |
| **Vectors & RAG** | `/observe/vectors` | Qdrant collections, points and vectors, 2D map, one search step by step (embedding → Qdrant → reranker) |
| **Matching internals** | `/observe/matching` | For a candidate and a job: evidence per requirement, HR rules retrieved, the exact prompt for Qwen |
| **Database** | `/observe/database` | Every PostgreSQL table, rows and JSON columns |
| **Run guide** | `/guide` | This document |

### The output of every step (application page)

| Step | What the panel shows |
|---|---|
| Intake checks | file, size, source, consent, job applied to |
| Document processing | reader (PyMuPDF, python-docx or OCR), pages and OCR confidence, **the full CV text read** |
| Candidate JSON | **the structured CV**: identity, summary, experience timeline, education, projects (with technologies), skills by category, certifications, languages; model, adapter, tokens; button **Compare with the base model** |
| Store and index | the candidate row, and **its Qdrant points** (one BGE-M3 vector per position, project, degree, skills group) |
| Acknowledge | the acknowledgement e-mail and its delivery status |
| Screening agent | **the LangGraph graph with the path taken** (click a node to filter), the timeline, **every tool call with its inputs and what it returned** (CV Search, HR Knowledge Search, Semantic Matching), **the HR documents retrieved — click one to open the exact document on the section used**, planning rounds (Qwen's thoughts, calls refused by code), CV evidence per requirement, checks by code, **the exact prompt and Qwen's raw answer** |
| Evidence-backed report | grade, score, requirements with CV quotes, strengths, gaps, interview questions, HR rules used (RAG), PDF/Markdown |
| Notify the recruiters | the notification |
| Human validation | waiting: **decide right here**; then the decision, reviewer, interview date, notes |
| Final decision | interview, shortlist or rejection |
| Notify the candidate | **the answer e-mail**; for an interview: **the video meeting** (Join link, date, Google Calendar event) and the calendar invite |
| Close | final decision, total time |

---

## 9. E-mails to the candidates

The pipeline writes two e-mails per application: the **acknowledgement** (step *Acknowledge the candidate*)
and the **answer after the recruiter's decision** (step *Notify the candidate*). For **Schedule interview**,
the answer includes the date, a video meeting link and a calendar invitation (`invite.ics`, or Google's own
invitation, section 10). Nothing is sent before a human decides (KB-RR-001). The rejection reason stays
internal.

- **`EMAIL_MODE=outbox` (default)**: e-mails are stored and shown on **E-mails & meetings**, not sent.
- **`EMAIL_MODE=smtp`**: really sent. With Gmail:
  1. Google account → *Security* → turn on *2-Step Verification*;
  2. *Security* → *App passwords* (https://myaccount.google.com/apppasswords) → create one (16 characters);
  3. in `.env`:

     ```text
     EMAIL_MODE=smtp
     SMTP_HOST=smtp.gmail.com
     SMTP_PORT=587
     SMTP_USER=your.address@gmail.com
     SMTP_PASSWORD=the 16-character app password
     EMAIL_REDIRECT_TO=your.address@gmail.com
     ```

  4. restart the API. *E-mails & meetings* shows “SMTP — really sent”; e-mails kept in the outbox can be sent
     with **Send now**.

**Keep `EMAIL_REDIRECT_TO` set during demos**: the evaluation CVs belong to real people; with it, every
e-mail goes to your own address (the app shows “redirected from …”). Synthetic candidates use reserved
`@example.com` addresses, which are never e-mailed. If sending the answer fails (wrong password…), the step
*Notify the candidate* fails visibly: fix `.env`, restart the API, then *Continue the workflow* on the
application page.

---

## 10. Google Meet invitations (optional)

Without this setup, interviews get a **Jitsi Meet** link (works at once, no account) and a calendar invite.
With it, every scheduled interview creates a **Google Calendar event with a Google Meet link**, and Google
e-mails the invitation to the candidate (or to `EMAIL_REDIRECT_TO`).

1. https://console.cloud.google.com → create a project → *APIs & Services* → *Library* → enable
   **Google Calendar API**.
2. *APIs & Services* → *OAuth consent screen* → *External* → fill the app name and your e-mail → *Test users*:
   add your Gmail address.
3. *Credentials* → *Create credentials* → *OAuth client ID* → **Desktop app** → *Download JSON* → save it as
   `secrets\google_client_secret.json` in the project folder.
4. Authorise once (a browser opens, sign in, accept):

   ```powershell
   python scripts/google_calendar_setup.py
   ```

   It saves `secrets\google_token.json` (git ignores `secrets/`: never commit it).
5. Keep `MEETING_PROVIDER=auto` (or `google`), restart the API. *E-mails & meetings* shows
   “Google Calendar: authorised” and “Interview meetings: Google Meet”.

With `auto`, a Google error falls back to Jitsi (with a note on the meeting); with `google`, it stops the
step so that you can fix it.

---

## 11. Evaluation and MLflow

**In the app**: **Evaluation & MLflow** (`/evaluation`) — tabs *Extraction* (base vs fine-tuned, per field,
per document type, per CV), *Fine-tuning* (loss curve, hyper-parameters, training data, smoke test), *RAG*
(Hit@k, MRR, recall per language and cross-lingual; dense vs reranked), *Matching* (semantic matcher, Qwen
analysis, screening agent per split), *Screening agent* (tool and model calls, checks, fallbacks), *Assistant*
(success, tool choice, refusals, per category, every scenario), *Live pipeline* (timings and outcomes of every
step of the real applications). **Base vs fine-tuned** (`/models`) focuses on the fine-tuning.

**In MLflow** — the same results as experiments, runs, params, metrics (with the loss curve per step) and the
report files as artifacts:

```powershell
python scripts/mlflow_sync.py            # logs what changed in data/evaluation/ (add --force to log everything again)
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000
```

Open http://localhost:5000. Experiments: *DSO1 - CV extraction (Qwen2.5 base vs QLoRA)*, *DSO1 - QLoRA
fine-tuning*, *DSO2 - RAG retrieval*, *DSO3 - Matching*, *DSO4 - Screening agent*, *DSO4 - AI assistant*.
To compare base vs fine-tuned: open the extraction experiment, tick both runs, **Compare**. The evaluation page
of the app shows whether MLflow is running and links to each experiment.

New results come from Kaggle (`python scripts/kaggle_run.py all <job>`, README section 9); run
`python scripts/mlflow_sync.py` again afterwards.

---

## 12. A 10-minute demonstration

1. **Dashboard**: all services green, *LLM: remote*, adapter loaded.
2. **Candidate portal**: accept the policy, choose a job, drop a CV, *Run by: n8n workflow* → *Send my
   application*. The answer arrives after the extraction.
3. **Application pipeline** → the application: watch the stepper. Click **Candidate JSON**: the structured CV;
   **Compare with the base model** (about a minute).
4. Click **Screening agent**: the graph path, the tool calls (open a few to see their results), click an HR
   document (e.g. KB-RR-002) to open it on the section used; the prompt and Qwen's answer.
5. **Evidence-backed report**: requirements with CV quotes; download the PDF.
6. **Human validation** (or the *Human validation* step): **Schedule interview**, pick a date → *Save*.
7. Back on the application: *Notify the candidate* shows the e-mail and the **Join** link; **E-mails &
   meetings** lists it (and your inbox has it with `EMAIL_MODE=smtp` + `EMAIL_REDIRECT_TO`).
8. **Base vs fine-tuned** and **Evaluation & MLflow**: the numbers; open MLflow.
9. **AI assistant**: “Who are the best candidates for the ML Engineer job?” → *How the agent answered*.

---

## 13. Run the tests and checks

```powershell
python -m pytest -q                        # backend unit tests (no GPU, no Docker), all must pass
python -m ruff check src scripts tests api # code style
cd web; npm run typecheck; cd ..           # interface: TypeScript types
python scripts/health_check.py             # real services: PostgreSQL, Qdrant, API
```

New tests of this update: `tests/unit/test_emails.py` (e-mails, outbox/SMTP, redirection, meetings, calendar
invite), `tests/unit/test_knowledge_evaluation.py` (knowledge documents, evaluation endpoints, base vs
fine-tuned on a stored candidate, the interface served by the API), `tests/unit/test_mlflow_sync.py`.

---

## 14. Everything in Docker (optional)

```powershell
docker compose --profile app up -d --build     # builds the interface and the API image; http://localhost:8000
```

The API container reads `.env` (models, e-mails); `secrets/` is mounted for Google Calendar.

---

## 15. Troubleshooting

| Problem | Fix |
|---|---|
| http://localhost:8000 says *The interface is not built yet* | `cd web; npm install; npm run build`, then reload (no API restart needed) |
| Blank page or old interface after a `git pull` | `cd web; npm install; npm run build`, then `Ctrl+F5` |
| `npm` not found | Install Node.js 20 LTS (https://nodejs.org), open a new terminal |
| Top bar: *API* red | Start the API (section 4); check that port 8000 is free |
| *LLM: mock* in the top bar | Kaggle AI server not connected (section 6); AI answers are fakes |
| *The Kaggle AI server does not answer* | The Kaggle session stopped or its link changed: run the notebook again, paste the new lines, restart the API |
| *n8n answered 404* when submitting through n8n | `python scripts/n8n_setup.py` (imports and activates the workflows) |
| E-mail status *Failed — SMTPAuthenticationError* | Use a Gmail **App password**, not your normal password; restart the API; *Send now* |
| E-mail status *Not sent — reserved example address* | Synthetic candidates (`@example.com`) are never e-mailed: set `EMAIL_REDIRECT_TO` |
| Meeting provider stays *Jitsi* | `python scripts/google_calendar_setup.py` was not run, or `pip install -e ".[google]"` is missing |
| *Google authorisation is no longer valid* | Run `python scripts/google_calendar_setup.py` again |
| Interview time shown in UTC with a note | `pip install tzdata` (Windows needs it for time zones), restart the API |
| MLflow page: *not installed* | `pip install -e ".[tracking]"` |
| MLflow page: *interface not running* | `python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000` |
| MLflow shows no experiment | `python scripts/mlflow_sync.py` from the project folder |
| Search or agent finds no HR document | Dashboard → *Index the HR knowledge + load the demo* |
| Database error after the update | Restart the API: it applies the new migration (table `emails`) |
| A change in the Python code has no effect | The API reads the code once, when it starts (no `--reload`): stop it (`Ctrl+C`) and start it again |
| A candidate stored with a wrong e-mail or phone (extracted before the contact check, or by an API started before it) | `python scripts/check_contacts.py` shows the corrections (columns, profile, e-mails not sent yet), `--apply` saves them; new CVs are checked automatically |

---

## 16. Command cheat sheet

```powershell
.\.venv\Scripts\Activate.ps1                                   # every new terminal
docker compose up -d postgres qdrant n8n                       # services
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000      # API + interface -> http://localhost:8000
cd web; npm run dev                                            # interface in development -> http://localhost:5173
cd web; npm run build                                          # rebuild the interface served by the API
python scripts/n8n_setup.py                                    # (re)import the n8n workflows
python scripts/kaggle_run.py push server                       # update the Kaggle AI server's code
python scripts/google_calendar_setup.py                        # authorise Google Meet invitations
python scripts/mlflow_sync.py                                  # log the evaluations into MLflow
python scripts/check_contacts.py --apply                       # fix stored e-mails/phones against the CVs
python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000   # MLflow -> http://localhost:5000
python -m pytest -q                                            # tests
docker compose stop                                            # stop the services (data kept)
```
