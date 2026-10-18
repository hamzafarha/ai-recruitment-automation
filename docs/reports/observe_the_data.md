# Run the project and observe every part of it — step-by-step guide

**Yes, you can watch every part work and see everything the system stores**, live, while you use the app:

| What | Where it lives | See it in the browser | See it in the terminal | See it in the app |
|---|---|---|---|---|
| **Every part of the architecture working**: reading a CV, fine-tuned vs base model, retrieval and reranking, contextualization, Qwen's raw answers, the agent's path through its graph | the API and the Kaggle AI server | — | — | **Under the hood** pages (part 2) |
| **Database rows** (candidates, jobs, analyses, decisions, notifications, assistant runs) | PostgreSQL, container `recruitment_postgres`, port 5433 | **Adminer**: http://localhost:8080 | `psql` inside the container | Dashboard, Candidates, Reports, Notifications |
| **Vector database** (HR knowledge chunks, CV snippets, job chunks, with their BGE-M3 vectors) | Qdrant, container `recruitment_qdrant`, port 6333 | **Qdrant dashboard**: http://localhost:6333/dashboard | REST calls (`Invoke-RestMethod`) | Search page, Dashboard → System status |
| **n8n workflows** (and every run of them, with the data that went through each step) | n8n, container `recruitment_n8n`, port 5678 | **n8n editor**: http://localhost:5678 | `docker exec recruitment_n8n n8n list:workflow` | Notifications & n8n page |

Part 1 starts everything; **part 2 shows each part of the architecture working, in the app**; parts 3–5 show each store with its own tools; part 6 follows one CV through all three; part 7 gives the numbers you should see. All commands are for **Windows PowerShell**, run from the project folder.

---

## 1. Start everything

### 1.1 Once, after getting the latest code

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[all]"
python -m pytest -q
python -m ruff check src scripts tests api frontend
```

`pytest` checks the whole application with fake models (no GPU, no Docker); `ruff` checks the code style. Both
must pass before you go on (send any failure to fix it).

### 1.2 Start the AI server on Kaggle (each session)

1. Once, and after each code change: `python scripts/kaggle_run.py push server`. It sends the code to Kaggle and
   creates the notebook **recruitment-ai-model-server** with your fine-tuned adapter attached. (The automatic run
   it starts stops at once: that is normal.)
2. kaggle.com → *Your Work* → *Code* → **recruitment-ai-model-server** → **Edit** → right panel: *GPU T4 x2*,
   *Internet on* → **Run All**.
3. After about 5 minutes, the 4th cell prints four lines. Paste them into `.env` (replace the old values):

```text
LLM_PROVIDER=remote
EMBEDDING_PROVIDER=remote
REMOTE_LLM_URL=https://<something>.trycloudflare.com
REMOTE_LLM_TOKEN=<secret>
```

### 1.3 Start the Docker services

Open **Docker Desktop** and wait for *Engine running*, then:

```powershell
docker compose up -d postgres qdrant n8n            # database, vector database, n8n
docker compose --profile tools up -d adminer        # optional: the database viewer (part 3.1)
docker compose ps                                   # all must be "Up" (postgres and qdrant "healthy")
```

### 1.4 Start the API and the interface (two terminals)

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000     # terminal 1: creates/updates the tables at start
python -m streamlit run frontend/streamlit_app.py              # terminal 2: http://localhost:8501
```

If Windows asks, allow Python on **private** networks (n8n, in Docker, calls the API).

### 1.5 First time only: data and workflows

1. Interface → **Dashboard** → **Set up: index the HR knowledge + load the demo data** (about 1 minute).
2. Import and activate the n8n workflows: `python scripts/n8n_setup.py`. Then set `N8N_EVENTS_ENABLED=true` in
   `.env` and restart the API (Ctrl+C in terminal 1, then the same command).
3. Open http://localhost:5678 once and create the local n8n owner account (any e-mail and password: they stay on
   your PC).

You now have data in all three stores. Use the app (part 6 suggests a path) and watch it appear.

---

## 2. Observe each part of the architecture in the app ("Under the hood")

The menu's **Under the hood** section has one page per part of the diagram. Each shows the part working on your
own data, with the same code the system uses (nothing is simulated).

| Part of the diagram | Page | Do | You see |
|---|---|---|---|
| All 9 | 🗺️ **Architecture (live)** | open it | the architecture drawn with each part colored live (green = working with data, grey = not used yet, red = a problem), its figures, and a link to observe it |
| 1. Users & inputs, 2. n8n | **Candidate portal**, **Notifications & n8n** | apply *through n8n*; open Notifications & n8n | which workflows are published (🟢), their latest runs, the messages n8n wrote |
| 3. Document processing | 📄 **Document processing** | upload a CV → *Process this CV step by step* | each page read from its text layer or by OCR (with the OCR confidence), the extracted text, the exact prompt, Qwen's raw JSON answer, the checked profile |
| 4. Fine-tuning | 📄 **Document processing** (box *also run the base model* ticked) | same | the base Qwen2.5-7B and our fine-tuned adapter on the same CV: sections found, identity fields, the skills only one of them found, both raw answers and timings |
| 5. Embeddings, Qdrant, reranker | 🧭 **Vectors & RAG** → *One search, step by step* | type a question | the question's vector (1,024 numbers), the closest points by cosine similarity, then the order after the BGE reranker, with how far each point moved (↑ ↓) and which are kept |
| 5. Contextualization, LLM analysis | 🧩 **Matching internals** | choose a job and a candidate → *Show what the analysis uses*, then *Run the full AI analysis now* | for each requirement, the CV snippets compared and their scores; the 3 searches in the HR knowledge base; the full prompt; Qwen's raw answer; the quotes the code rejected |
| 5. Agent (LangGraph) | 🤖 **Agent internals** | ask a question → *Ask and trace* | the graph, the path of this question through it (decide → act → … → respond) with times and tokens, each tool's result, the system prompts |
| 6. Knowledge sources, outputs | 🧭 **Vectors & RAG** → *Collections*, *Browse points*, *2D map*; **Reports** | browse | the three knowledge sources as points (text + metadata + vector), a 2D map where similar texts form groups; each report's own *Under the hood* section |
| 7. External services | 🗺️ **Architecture**, **Notifications & n8n** | open them | the Kaggle AI server's models and requests served; every notification |
| 8. Data storage | 🗄️ **Database** | pick a table, open a row | every PostgreSQL table, its rows and its JSON columns |
| 9. Human validation | **Human validation**, then 🗄️ **Database** → `reviews` | decide on an analysis | the decision stored, the analysis status changed, n8n's message |

Optional: to list the n8n runs inside the app too, create an API key in n8n (*Settings → n8n API → Create an API
key*), add `N8N_API_KEY=<the key>` to `.env` and restart the API. Without it, the runs are in n8n (part 5.3).

### 2.1 How the AI assistant (agent) works — what *Agent internals* shows

The assistant is a **LangGraph** graph of three nodes, run in a loop:

1. **decide** — Qwen2.5-7B receives the list of 8 tools (its system prompt), the question, the jobs, and the
   results of the tools already called. It answers in JSON: a *thought*, an *action* (a tool name, or
   `final_answer`) and the tool's *arguments*.
2. **act** — the chosen tool runs: a Qdrant search reranked by the BGE cross-encoder, a PostgreSQL lookup, the
   ranking or matching code. Its result becomes an *observation*, added to what Qwen sees.
3. Back to **decide**, until Qwen chooses `final_answer`, or 4 tool calls are done.
4. **respond** — Qwen writes the answer from the observations only, naming the candidates, jobs and HR documents.

Code guards every step: the arguments are validated (common variants such as `q` for `query` are accepted), the
same call is never repeated, a failed step degrades gracefully (the tool results are shown as they are), and
every run is stored with its timeline in the `agent_runs` table. Example: *"Who are the best candidates for the
GenAI Engineer job?"* → decide → act `rank_candidates(job="GenAI Engineer")` → decide → respond.

### 2.2 How the agentic RAG works — what *Vectors & RAG* and *Matching internals* show

1. **Indexing**: every text (an HR document section, a CV snippet, a job chunk) becomes a vector of 1,024 numbers
   with BGE-M3, stored in Qdrant with its metadata (the *payload*).
2. **Retrieval**: a question becomes a vector too; Qdrant returns the closest points (cosine similarity).
3. **Reranking**: the BGE cross-encoder reads each (question, text) pair and scores it more precisely; the best
   ones are kept.
4. **Contextualization**: the AI analysis builds one prompt from the job's requirements, the company rules
   retrieved, the experience timeline computed by code (KB-RR-003) and the CV text.
5. **LLM analysis**: Qwen judges every requirement and quotes the CV.
6. **Checks by code**: every quote must be found in the CV, and the grade comes from the rubric (KB-RR-002).

It is *agentic* in the assistant: Qwen itself decides which searches to run (CV search, job search, HR knowledge
search, ranking, matching, reranking), reads the results, and decides whether to search more before answering.

---

## 3. PostgreSQL — the database rows

Connection details (from `.env`): host `localhost`, port **5433**, database `recruitment_db`, user `postgres`,
password `postgres` (unless you changed them).

### 3.1 In the browser: Adminer

1. `docker compose --profile tools up -d adminer` (once per session; it is not started by default).
2. Open **http://localhost:8080** and log in:
   - **System**: PostgreSQL (not MySQL!)
   - **Server**: `postgres` (already filled in)
   - **Username**: `postgres` · **Password**: `postgres` · **Database**: `recruitment_db`
3. Click a table (left), then **Select data** to see its rows. The JSON columns (`profile`, `requirements`,
   `report`, `steps`…) show as text: click a row's **edit** link to read a whole value.
4. **SQL command** (top left) runs any query from the cookbook below.

Port 8080 already used on your PC? Add `ADMINER_PORT=8081` to `.env`, run the command again and open
http://localhost:8081.

### 3.2 In the terminal: psql (nothing to install)

```powershell
docker exec -it recruitment_postgres psql -U postgres -d recruitment_db
```

Useful psql commands: `\dt` (list tables), `\d candidates` (columns of a table), `\x auto` (readable wide rows),
`\pset pager off` (no paging), `\q` (quit). For a single query without entering psql:

```powershell
docker exec recruitment_postgres psql -U postgres -d recruitment_db -c "SELECT full_name, source FROM candidates;"
```

### 3.3 Query cookbook

Copy these into psql or Adminer's *SQL command*. JSON columns are cast with `::jsonb` to look inside them.

**How many rows in each table**

```sql
SELECT 'applications' AS table_name, count(*) FROM applications
UNION ALL SELECT 'candidates', count(*) FROM candidates
UNION ALL SELECT 'jobs', count(*) FROM jobs
UNION ALL SELECT 'analyses', count(*) FROM analyses
UNION ALL SELECT 'reviews', count(*) FROM reviews
UNION ALL SELECT 'notifications', count(*) FROM notifications
UNION ALL SELECT 'agent_runs', count(*) FROM agent_runs;
```

**Candidates, newest first**

```sql
SELECT left(id, 8) AS id, full_name, location, source, cv_filename, created_at::date
FROM candidates ORDER BY created_at DESC;
```

**One candidate's full extracted profile (pretty JSON), skills, and experience**

```sql
SELECT jsonb_pretty(profile::jsonb) FROM candidates WHERE full_name ILIKE '%priya%';

SELECT s->>'name' AS skill
FROM candidates, jsonb_array_elements(profile::jsonb->'skills') s WHERE full_name ILIKE '%priya%';

SELECT e->>'title' AS title, e->>'company' AS company, e->>'start_date' AS start, e->>'end_date' AS "end"
FROM candidates, jsonb_array_elements(profile::jsonb->'experience') e WHERE full_name ILIKE '%priya%';
```

**How the extraction of your uploaded CVs went** (model, adapter, time, document type)

```sql
SELECT full_name, extraction->>'model' AS model, extraction->>'adapter' AS adapter,
       extraction->>'seconds' AS seconds, extraction->>'input_type' AS input, extraction->>'indexed' AS indexed
FROM candidates WHERE source <> 'demo';
```

**Jobs and their requirements**

```sql
SELECT left(id, 8) AS id, title, company, status, source, json_array_length(requirements) AS requirements
FROM jobs ORDER BY created_at;

SELECT r->>'id' AS id, r->>'requirement_type' AS type, r->>'importance' AS importance,
       r->>'min_years_experience' AS years, r->>'requirement_text' AS requirement
FROM jobs, jsonb_array_elements(requirements::jsonb) r WHERE title ILIKE '%GenAI%';
```

**Analyses: who was graded for which job**

```sql
SELECT c.full_name, j.title, a.method, a.grade, a.final_grade, a.score, a.status, a.created_at::timestamp(0)
FROM analyses a
JOIN candidates c ON c.id = a.candidate_id
JOIN jobs j ON j.id = a.job_id
ORDER BY a.created_at DESC;
```

**The evidence of the latest analysis, requirement by requirement**

```sql
SELECT q->>'id' AS req, q->>'status' AS status, q->>'level' AS level, left(q->>'evidence', 80) AS evidence
FROM (SELECT report FROM analyses ORDER BY created_at DESC LIMIT 1) a,
     jsonb_array_elements(a.report::jsonb->'requirements') q;
```

**Recruiter decisions (human validation)**

```sql
SELECT c.full_name, j.title, r.decision, r.reviewer, r.adjusted_grade, r.interview_at, r.notes,
       r.created_at::timestamp(0)
FROM reviews r
JOIN analyses a ON a.id = r.analysis_id
JOIN candidates c ON c.id = a.candidate_id
JOIN jobs j ON j.id = a.job_id
ORDER BY r.created_at DESC;
```

**Notifications** (`source = n8n` = messages prepared by the n8n workflows)

```sql
SELECT created_at::timestamp(0), source, event, title, read
FROM notifications ORDER BY created_at DESC LIMIT 20;
```

**AI assistant: questions, and the tools it used for each**

```sql
SELECT created_at::timestamp(0), left(question, 60) AS question, seconds
FROM agent_runs ORDER BY created_at DESC;

SELECT left(r.question, 50) AS question, s->>'tool' AS tool, s->>'seconds' AS seconds
FROM agent_runs r, jsonb_array_elements(r.steps::jsonb) s
ORDER BY r.created_at DESC;
```

**Every application and the step it is at** (the application pipeline)

```sql
SELECT a.created_at, c.full_name, a.status, a.stage, a.orchestrator, a.decision
FROM applications a LEFT JOIN candidates c ON c.id = a.candidate_id
ORDER BY a.created_at DESC;
```

**The steps of the newest application, one row each**

```sql
SELECT s ->> 'step' AS step, s ->> 'status' AS status, s ->> 'seconds' AS seconds, s ->> 'error' AS error
FROM (SELECT steps FROM applications ORDER BY created_at DESC LIMIT 1) a, jsonb_array_elements(a.steps::jsonb) s;
```

**Database version** (should be `b7c1d2e3f4a5`, the migration that added the application pipeline)

```sql
SELECT version_num FROM alembic_version;
```

### 3.4 What each table holds

| Table | One row = | Main columns |
|---|---|---|
| `applications` | one application through the pipeline (CV → final decision) | `status` (running, waiting_review, closed, failed), `stage` (the step it is at), `steps` (each step's status, times, result, error, JSON), `orchestrator` (n8n / api), `decision`, `resume_url` (the n8n Wait node to resume) |
| `candidates` | a candidate | `full_name`, `email`, `location`, `profile` (the whole extracted profile, JSON), `cv_text` (the CV text quoted as evidence), `source` (upload / demo / portal), `extraction` (model, adapter, timings) |
| `jobs` | a job | `title`, `company`, `role` (HR job profile), `requirements` (M1.. must-haves, N1.. nice-to-haves, JSON), `status` |
| `analyses` | one candidate analysed for one job | `method` (agent / llm / semantic), `grade` (0–3), `score`, `status` (pending_review, shortlisted, interview, on_hold, rejected), `final_grade`, `report` (the full evidence-backed report, JSON), `application_id` (the pipeline application it belongs to) |
| `reviews` | a recruiter decision | `decision`, `reviewer`, `adjusted_grade`, `interview_at`, `notes` |
| `notifications` | an event or an n8n message | `event`, `title`, `message`, `payload`, `source` (app / n8n), `read` |
| `agent_runs` | a question to the AI assistant | `question`, `answer`, `steps` (tool calls with their results), `sources`, `seconds` |

Other ways: any database tool (DBeaver, pgAdmin, the VS Code "PostgreSQL" extension) with the connection details
above; or the API pages at http://localhost:8000/docs (`GET /api/v1/candidates`, `/jobs`, `/analyses`…).

---

## 4. Qdrant — the vector database

### 4.1 In the browser: the Qdrant dashboard

Open **http://localhost:6333/dashboard** (built into Qdrant, nothing to install).

1. **Collections**: you should see three — `hr_knowledge`, `candidates`, `jobs` — with their number of points.
2. Click a collection → **Points**: every point is one piece of text, with
   - its **payload** (the text and where it comes from: see the table in 3.4);
   - its **vector**: 1,024 numbers computed by BGE-M3 (expand it to see them).
   Buttons on a point let you find the most similar points: try it on a CV snippet.
3. **Info** tab: vector size 1024, distance *Cosine*, points count, status.
4. **Visualize** tab (in recent versions): draws all the vectors of the collection on a 2D map. Points that are close
   mean similar texts. Color them by a payload field, for example in the request editor:
   `{"limit": 500, "color_by": {"payload": "section"}}` for `hr_knowledge`, or `"document_id"`; for `candidates`
   use `"name"` — each candidate's snippets form a group.
5. **Console** (left menu): type REST requests directly, e.g.

```text
GET collections

POST collections/candidates/points/scroll
{ "limit": 5, "with_payload": true }
```

### 4.2 In the terminal: REST calls

```powershell
# the collections and their sizes
Invoke-RestMethod http://localhost:6333/collections | ConvertTo-Json -Depth 5
(Invoke-RestMethod http://localhost:6333/collections/candidates).result.points_count

# five HR knowledge chunks with their text and metadata (no vectors)
$url = "http://localhost:6333/collections/hr_knowledge/points/scroll"
$body = '{"limit": 5, "with_payload": true, "with_vector": false}'
(Invoke-RestMethod -Method Post -Uri $url -ContentType "application/json" -Body $body).result.points | ConvertTo-Json -Depth 6

# every snippet of one candidate (what the search and the matching compare with the job requirements)
$url = "http://localhost:6333/collections/candidates/points/scroll"
$body = '{"limit": 50, "with_payload": true, "filter": {"must": [{"key": "name", "match": {"value": "Priya Raman"}}]}}'
(Invoke-RestMethod -Method Post -Uri $url -ContentType "application/json" -Body $body).result.points | ForEach-Object { $_.payload.label }

# one vector: 1,024 numbers
$url = "http://localhost:6333/collections/jobs/points/scroll"
$body = '{"limit": 1, "with_payload": false, "with_vector": true}'
$v = (Invoke-RestMethod -Method Post -Uri $url -ContentType "application/json" -Body $body).result.points[0].vector
$v.Count
$v[0..4]
```

### 4.3 In the app

- **Search** page: type a question; the results are the closest Qdrant points after the BGE reranker, with their
  relevance score (HR knowledge, candidates, jobs).
- **Dashboard → System status**: the number of points per collection.
- `GET http://localhost:8000/api/v1/admin/status` (from http://localhost:8000/docs): the same counts as JSON.

### 4.4 What each collection holds

| Collection | One point = | Payload fields |
|---|---|---|
| `hr_knowledge` | one section of an HR document (99 chunks from 15 documents) | `text`, `document_id` (e.g. KB-RR-002), `title`, `section`, `category`, `language` (en/fr), `roles` (job profiles it applies to), `document_type`, `source`, `chunk_id` |
| `candidates` | one CV snippet: a position, a project, a degree, a certification, a skills group (30 skills), the languages, the summary | `text`, `candidate_id`, `name`, `section` (experience, project, education, skills…), `label` (e.g. "GenAI Engineer at Northwind Analytics (2023-02 → present, 3.8 years)") |
| `jobs` | the requirements of a job, or a paragraph of its description | `text`, `job_id`, `title`, `section` (requirements, description 1, 2…) |

Point ids are fixed from the source (for example, from the candidate id and the snippet), so indexing the same
candidate again replaces its points instead of duplicating them. Deleting a candidate in the app also deletes their
points.

---

## 5. n8n — the workflows

### 5.1 Open the editor

http://localhost:5678 — log in with the owner account created in step 1.5.

### 5.2 See the workflows

**Overview → Workflows** lists the 5 workflows. If they are missing: `python scripts/n8n_setup.py`, or by hand:
create a new workflow, open its **⋯** menu (top right) → **Import from File…** → one file of `n8n/workflows/`,
save, and repeat for the 5 files.

| Workflow | Starts when | Steps you will see on the canvas |
|---|---|---|
| Recruitment — Application pipeline (CV → final decision) | a CV arrives: **Candidate portal**, or **Candidates → Add** run by n8n | CV received → Intake checks → Document processing → Candidate JSON (fine-tuned Qwen) → Store and index → Acknowledge the candidate → Answer the candidate → Screening agent (LangGraph) → Evidence-backed report → Notify the recruiters → **Wait for the recruiter's decision** → Final decision → Decided? → Notify the candidate → Close the application (each node calls the API; an error branch answers the candidate when a step fails) |
| Recruitment — New job (advertisement intake) | a job ad is posted to `/webhook/new-job` | Job advertisement received → Structure and store the job → Answer |
| Recruitment — Events (messages and notifications) | the other app events (Matching page analyses and their decisions), when `N8N_EVENTS_ENABLED=true` | App event → Prepare the message → Show it in the app (+ a disabled "Send e-mail" node) |
| Recruitment — Daily digest | every day at 9:00, or the app's *Run the daily digest now* | Get the digest → Prepare the notification → Show it in the app |
| Recruitment — Error alert | another workflow fails | A workflow failed → Prepare the alert → Show it in the app |

Open one: each box is a step; the yellow note explains the workflow; click a box to see its settings (for example
the API address it calls). A workflow must be **Active** (switch at the top right; **Publish** in recent n8n
versions) to answer the app.

### 5.3 See what happened: executions

Every run is recorded with the data that went through each step:

1. Open a workflow → **Executions** tab (top of the page), or **Overview → Executions** for all workflows.
2. Click a run (green = succeeded, red = error).
3. Click a step: the right panel shows its **input** and **output** JSON. For example, in *Application
   pipeline* the step "Screening agent (LangGraph)" shows the API's answer (the jobs analysed, every tool call, the
   agent's timeline); an execution that waits for a recruiter shows as *waiting* until the decision resumes it; in
   *Events*, "Prepare the message" shows the message written for the event.

A run of *Events* with no output after "Prepare the message" is normal: that event needs no message (for example
a CV uploaded by the recruiter, or "setup completed").

### 5.4 Make each workflow run

From the app:

| Do this in the app | Runs |
|---|---|
| **Candidate portal** → apply with a CV, route *Through n8n* (with or without a job) | Application pipeline, up to its Wait node |
| **Human validation** → decide on that application's analysis | the same Application pipeline execution resumes and closes |
| **Human validation** → decide on an analysis made from the Matching page | Events (interview invitation, or the answer) |
| **Notifications & n8n** → *Send a test event to n8n* | Events ("n8n is connected") |
| **Notifications & n8n** → *Run the daily digest now* | Daily digest |

From PowerShell:

```powershell
# a test event -> a notification "n8n is connected" in the app
$body = '{"event": "test", "title": "Test", "message": "", "payload": {}, "source": "app"}'
Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook/recruitment-events -ContentType "application/json" -Body $body

# the daily digest, now
Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook/daily-digest -ContentType "application/json" -Body '{}'

# a new job from an advertisement (Qwen reads it: the Kaggle AI server must be running)
$ad = @{ text = "Data Engineer - Tunis. We need 3+ years of Python and SQL, experience with Airflow pipelines and data modelling. Spark and AWS are a plus." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook/new-job -ContentType "application/json" -Body $ad -TimeoutSec 900

# list the workflows from the terminal
docker exec recruitment_n8n n8n list:workflow
```

The workflow files themselves are in the project: `n8n/workflows/*.json`.

---

## 6. Follow one CV through the three stores

A good way to understand the system — and to show it in a demo.

| # | Do | Then look at |
|---|---|---|
| 1 | **Candidates → Add**: upload your own CV | PostgreSQL: the new row (`SELECT left(id,8), full_name, source FROM candidates ORDER BY created_at DESC LIMIT 1;`); `profile` holds what Qwen extracted |
| 2 | — | Qdrant → `candidates` → filter on your name (part 4.2): one point per position, project, degree, skills group |
| 3 | **Matching**: choose a job, analyse yourself | PostgreSQL: a row in `analyses` (status `pending_review`) and its evidence (cookbook "latest analysis"); a row in `notifications` (`analysis.completed`) |
| 4 | — | n8n → *Events* → Executions: the `analysis.completed` event; if your grade is 2 or 3, a "Priority review" message, also in `notifications` with `source = n8n` |
| 5 | **Human validation**: invite yourself to an interview | PostgreSQL: a row in `reviews`; `analyses.status` = `interview`; n8n → *Events*: the invitation text, shown in **Notifications** |
| 6 | **AI assistant**: "Who are the best candidates for the GenAI Engineer job?" | PostgreSQL: a row in `agent_runs`; its `steps` list the tools used (cookbook "AI assistant") |
| 7 | **Candidates** → your profile → *Delete* | PostgreSQL: your rows in `candidates`, `analyses`, `reviews` are gone; Qdrant: your points are gone (the filter returns nothing) |

Notifications are kept as history after a deletion (they contain names in their text): delete them in Adminer if
needed.

---

## 7. Numbers you should see after the first "Set up"

| Store | Expected |
|---|---|
| PostgreSQL | `candidates` 12 (source `demo`), `jobs` 5, `notifications` 1 (`setup.completed`), `analyses`, `reviews`, `agent_runs` 0; `alembic_version` = `a1b2c3d4e5f6` |
| Qdrant | `hr_knowledge` **99** points, `candidates` **77**, `jobs` **27**; every vector has 1,024 dimensions |
| n8n | 5 workflows; with `N8N_EVENTS_ENABLED=true`, one *Events* run for `setup.completed` (no message: normal) |

Each uploaded CV adds 1 row to `candidates` and one point per position, project, degree, certification, skills
group and languages line (often 5 to 15) to the `candidates` collection; each new job adds 1 row to `jobs` and one
point for its requirements plus one per part of its description. Running *Set up* again adds nothing but one
`setup.completed` notification.

---

## 8. Stop, reset, clean up

```powershell
docker compose stop                                # stop everything; all data is kept
docker compose --profile tools stop adminer        # stop only Adminer
```

- **Remove one candidate or job**: delete it in the app (rows and vector points are removed together).
- **Rebuild the vector database only**: in the Qdrant dashboard delete a collection (or
  `Invoke-RestMethod -Method Delete http://localhost:6333/collections/candidates`), then in the app
  **Dashboard → Set up** (knowledge base) or **Re-index** (jobs and candidates).
- **Start from zero** (deletes all rows, all vectors, the n8n account, workflows and executions):
  `docker compose down -v`, then redo step 1.5. Use it only when you really want an empty system.
- **Stop the Kaggle session** when you finish (power icon): the GPU quota counts while it is open.

---

## 9. Privacy reminders

- The `candidates` table and the `candidates` collection contain CV data. Use only your own or invented CVs; do
  not share screenshots of real candidates' rows.
- n8n keeps the data of every run. A *CV intake* run contains the whole CV file (base64): after a demo, delete
  those runs (**Executions** → select → **Delete**).
- `.env` holds `REMOTE_LLM_TOKEN`: never commit it or show it in a screenshot.

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| Adminer: *Unable to connect* | Choose **PostgreSQL** as System, Server `postgres` (the Docker name, not localhost) |
| psql: *No such container* | `docker compose up -d postgres`, check with `docker compose ps` |
| Tables missing (`relation "candidates" does not exist`) | Start the API once (it creates the tables), or `python -m alembic upgrade head` |
| Qdrant dashboard shows no collection | Interface → **Dashboard → Set up** (it needs the Kaggle AI server for the embeddings) |
| A collection has fewer points than expected | **Dashboard → Re-index**; check the API terminal for "Indexing ... failed" |
| n8n shows no workflow | `python scripts/n8n_setup.py`, or import the files of `n8n/workflows/` in the editor |
| n8n run in red at "Extract and store the candidate" | The API is not reachable from Docker: start it with `--host 0.0.0.0` and allow Python in the firewall |
| No n8n message after a decision | `N8N_EVENTS_ENABLED=true` in `.env`, API restarted, *Events* workflow active |
