# End-to-end test plan — from an empty system to the end

Follow the steps **in order** and tick each box when the result matches. Every step says what to **do**, what you
should **see**, and where to **check it under the hood** (the *Under the hood* pages of the app, Adminer, the
Qdrant dashboard or n8n). If a result differs, note the step number and send a screenshot.

- Time: about 2 hours, with the Kaggle GPU session open for most of it (it uses quota: do it in one sitting).
- Test files: your own CV, and the invented candidates of the evaluation set (no personal data):
  - `data/processed/eval_v1/synthetic/files/syn07.pdf` (English PDF);
  - `data/processed/eval_v1/synthetic/files/syn10.docx` (French Word file);
  - `build/test_files/syn05_scan.png` (a photo of a CV page) and `build/test_files/syn05_scanned.pdf` (a scanned PDF
    with no text layer): both must go through OCR.

  These invented people are also in the demo data, so after uploading them they appear twice (demo + upload): that
  is expected.
- Commands are for PowerShell, from the project folder, with the virtual environment active
  (`.\.venv\Scripts\Activate.ps1`).

---

## Phase A — Preparation (about 15 minutes)

### A1. The code checks

```powershell
python -m pytest -q
python -m ruff check src scripts tests api frontend
```

- [ ] pytest: every test passes (no `F` or `E` in the dots, last line `… passed`).
- [ ] ruff: `All checks passed!`

### A2. Start from an empty system

This deletes every row, vector, n8n workflow, n8n run and the n8n account, so that the setup is tested too. Your code,
`.env` and the evaluation files are not touched.

```powershell
docker compose down -v
docker compose up -d postgres qdrant n8n
docker compose --profile tools up -d adminer
docker compose ps
```

- [ ] `recruitment_postgres` and `recruitment_qdrant` are `Up (healthy)`, `recruitment_n8n` and `recruitment_adminer`
  are `Up`.

### A3. The AI server on Kaggle

If your Kaggle session is still running and answering, keep it. Otherwise: kaggle.com → *Your Work* → *Code* →
**recruitment-ai-model-server** → **Edit** → *GPU T4 x2*, *Internet on* → **Run All**.

- [ ] Cell 2 prints `GPUs: ['Tesla T4', 'Tesla T4']` and `✅ ready`.
- [ ] Cell 3 prints `fine-tuned adapter: /kaggle/input/...`, `✅ model loaded` and `✅ search models on cuda:…`.
- [ ] Cell 4 prints the four lines; they are in `.env` (a new session = a new link and token).

### A4. `.env`

- [ ] These lines are present (the link and token are those of the current session):

```text
LLM_PROVIDER=remote
EMBEDDING_PROVIDER=remote
REMOTE_LLM_URL=https://<…>.trycloudflare.com
REMOTE_LLM_TOKEN=<…>
N8N_URL=http://localhost:5678
N8N_EVENTS_ENABLED=true
```

### A5. The API

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- [ ] The log shows `Database schema is up to date` (the tables were created in the empty database).
- [ ] http://localhost:8000/api/v1/health shows `"status": "healthy"` with `postgresql`, `qdrant` and `ai_server`
  healthy (n8n is checked after B2).
- [ ] http://localhost:8000/docs lists the endpoints (candidates, jobs, analyses, agent, observe…).

### A6. The interface

```powershell
cd web; npm install; npm run build   # then open http://localhost:8000
```

- [ ] http://localhost:8000 opens on the Dashboard; the top bar shows 🟢 PostgreSQL · 🟢 Qdrant · 🟢 Kaggle AI.
- [ ] *Getting started*: check 1 is green; checks 2 and 3 are not (the system is empty).
- [ ] **Under the hood → Database**: every table has 0 rows; the schema version is `c3d4e5f6a7b8`.

---

## Phase B — Setup (about 10 minutes)

### B1. Knowledge base and demo data

Dashboard → **Set up: index the HR knowledge + load the demo data**.

- [ ] Message *Ready: 99 knowledge chunks, 5 jobs, 12 candidates indexed.*; the three checks are green.
- [ ] Figures: Candidates 12, Open jobs 5; one notification *Knowledge base and search index ready*.
- [ ] Under the hood → **Vectors & RAG → Collections**: hr_knowledge **99**, candidates **77**, jobs **27** points,
  vector size 1024, distance Cosine.
- [ ] http://localhost:6333/dashboard shows the same three collections.

### B2. n8n

```powershell
python scripts/n8n_setup.py
```

- [ ] The output ends with *the recruitment-events webhook answered*.
- [ ] http://localhost:5678: create the local account (the database was emptied); *Overview → Workflows* lists the
  5 *Recruitment — …* workflows, among them *Recruitment — Application pipeline (CV → final decision)*: open it, one
  node per step, with the part of the architecture under each node, a *Wait for the recruiter's decision* node
  and an error branch.
- [ ] App → **Notifications & n8n**: the 4 workflows are 🟢 *published*. (If not: open each in n8n → **Publish**.)
- [ ] *Send a test event to n8n* → *n8n received the event*; a moment later the notification **🤖 n8n is connected**.
- [ ] http://localhost:8000/api/v1/health now also shows `n8n` healthy.

### B3. The architecture, live

Under the hood → **Architecture (live)**.

- [ ] The diagram shows parts 1, 2, 3, 4, 5, 8 and Qdrant in green; 6 (reports) and 9 (decisions) are grey: nothing
  analysed or decided yet.

---

## Phase C — Every part, in the order of the diagram (about 75 minutes)

### C1. Document processing and fine-tuning (parts 3–4)

Under the hood → **Document processing** → upload **your own CV** → keep *Also run the base model* ticked →
**Process this CV step by step** (2–3 minutes).

- [ ] *1. Reading*: one row per page, read from the *text_layer* (a normal PDF).
- [ ] *2. The prompt*: the system prompt (8 extraction rules) and the user message (your CV + the JSON schema).
- [ ] *3. Qwen's answer*: the raw JSON written by the fine-tuned model, its time and token counts.
- [ ] *4. The profile*: your experience, education, projects, skills, languages.
- [ ] *Base model vs fine-tuned adapter*: a table of the sections found by each; the skills only one of them found;
  both raw answers. Note what the fine-tuned model does better or worse on your CV.
- [ ] Candidates page: nothing was added (this page stores nothing).

### C2. Adding candidates through the application pipeline: PDF, Word, scan (all parts)

Candidates → **Add a candidate** → *For the job*: **No specific job** → *Run by*: **the API** (n8n is tested in
C9), three times (the extraction takes about a minute, then the screening agent one to three minutes):

1. your own CV (PDF);
2. `data/processed/eval_v1/synthetic/files/syn10.docx` (French, Word: Karim Haddad);
3. `build/test_files/syn05_scanned.pdf` (scan without text: OCR). The first OCR can take longer (models download once).

- [ ] Each one is received; **Application pipeline** lists the 3 applications; open one while it runs: the steps
  turn green one after the other (the page refreshes by itself).
- [ ] When it is done: status **⏳ Waiting for a decision**, steps 1-8 done, *Human validation* waiting.
  *Document processing*: engine and input type (`text_pdf`, `docx`, `scanned_pdf` with its OCR pages);
  *Candidate JSON*: the profile counts and *fine-tuned QLoRA adapter*.
- [ ] *Screening agent*: the open jobs graded by the fast matching, the job(s) analysed (grade 2 or more, at most
  2; else the best one), the agent's path (understand → plan → act → evaluate → … → analyse → verify), the tool
  calls (round 0: one CV search per requirement and 3 HR searches; later rounds when the evidence was weak).
- [ ] *Evidence-backed report* → *Open the reports*: the analysis, method *Screening agent*; its *Screening agent*
  section: passages per requirement and the checks applied.
- [ ] Candidates → *All candidates*: 15 candidates; open one and read its *CV text* and *Raw profile*.
- [ ] Under the hood → **Database** → `candidates`: 15 rows; open your row: `profile` (JSON), `extraction` (model,
  adapter, seconds, input type), `source` = `upload`. `applications`: 3 rows, `steps` (JSON) with every step.
- [ ] Under the hood → **Vectors & RAG → Browse points** → `candidates`: your snippets are there (one per position,
  project, degree, skills group). **2D map** colored by `name`: your snippets form a group.
- [ ] *(Optional)* Candidates → *Test the extraction only* → `build/test_files/syn05_scan.png`: input `image`, read
  by OCR.

### C3. A job from an advertisement (parts 1, 6)

Jobs → **New job from an advertisement (AI)** → paste this invented ad → **Extract the requirements**:

```text
Data Engineer (Python / Airflow) — Nimbus Analytics, Tunis (hybrid)

Nimbus Analytics builds data platforms for retail and banking clients. You will design and run batch and
streaming data pipelines, model data for analytics, and help data scientists put models in production.

Requirements:
- 3+ years of experience as a data engineer
- Strong Python and SQL
- Apache Airflow or a similar orchestrator
- Data modelling (star schema, dbt)
- A cloud data platform (AWS, Azure or GCP)

Nice to have:
- Apache Spark or Kafka
- Docker and CI/CD
- French and English

Full-time, permanent contract.
```

- [ ] The title, company and location are found; 6–10 requirements: must-haves (e.g. *3+ years* with *Min. years* =
  3, Python, SQL, Airflow, data modelling, cloud) and nice-to-haves (Spark/Kafka, Docker/CI-CD, languages).
- [ ] Edit one row (for example change an importance), then **Save the job** → *Saved: … indexed for search*.
- [ ] Jobs → *All jobs*: 6 jobs; open yours: requirements with ids M1…, N1….
- [ ] Under the hood → **Vectors & RAG → Browse points** → `jobs`: your job's points (requirements + description).

### C4. Knowledge sources and retrieval (parts 5–6)

**Search** page:

| Tab | Query | Expected first result |
|---|---|---|
| HR knowledge base | `How are internships counted in years of experience?` | KB-RR-003, experience rules (internships 50 %) |
| HR knowledge base | `Combien de temps garde-t-on les données d'un candidat ?` | KB-HR-004, French data policy (24 months) |
| Candidates | `Kubernetes in production` | candidates with Kubernetes in a position (e.g. Mehdi Trabelsi, Elena Kovac) |
| Jobs | `SQL, Power BI dashboards and ETL pipelines` | the Talent Acquisition Data Engineering job (Solidigm), or your Data Engineer job |

- [ ] Each tab returns relevant results with their scores.
- [ ] Under the hood → **Vectors & RAG → One search, step by step**, same first query: the vector (1,024 numbers),
  the cosine order, then the reranker order with ↑/↓ moves; the KB-RR-003 section is kept.
- [ ] **2D map** of `hr_knowledge` colored by `document_id`: the sections of each document form groups.

### C5. Matching and evidence-backed analysis (part 5)

**Matching** → job **GenAI Engineer** (all candidates are ranked automatically).

- [ ] The ranking appears in seconds; the LLM/RAG profiles (Priya Raman, Yasmine Ben Salah) are near the top; the
  table shows grades, scores, must-haves met and gaps.
- [ ] Keep the 3 suggested candidates, *Full AI analysis* → **Analyse 3 candidate(s)** (about a minute each): each gets
  a grade, a score and a summary.
- [ ] Dashboard: Analyses and Pending reviews 3 more than after C2 (C2's pipeline analyses are counted too).
  Notifications: *Analysis ready* ×3 and, for grade 2–3, **🤖 Priority review** (written by n8n).

Under the hood → **Matching internals** → job GenAI Engineer + the first analysed candidate → **Show what the analysis
uses**:

- [ ] *1. Evidence*: a level per requirement; choose one and see the CV snippets compared, with cosine and reranker
  scores.
- [ ] *2. Company rules*: the 3 searches in the HR knowledge base and their results (job profile, rubric, experience
  rules).
- [ ] *3. The prompt*: the requirements, the rules, the experience timeline and the CV, as sent to Qwen.
- [ ] *4. What Qwen answered*: time and tokens, the raw JSON answer, *Every quote… was found in the CV* (or the list
  of quotes rejected), and the table *fast semantic vs AI analysis* per requirement.

### C6. The evidence-backed report (part 6)

**Reports** → open one of the 3 analyses.

- [ ] Grade, score, coverage, proposed action; the requirement table with the CV quote of each; strengths, gaps,
  interview questions; the HR documents used; the *Under the hood* section (raw answer, checks).
- [ ] **Report (PDF)** and **Report (Markdown)** download and open correctly.

### C7. Human validation and notifications (parts 9, 7, 2)

**Human validation**, one decision per analysis (take the analyses made in C5, method *AI*):

1. **Reject** with an empty reason → **Save**.
2. **Reject** again, with a reason (e.g. *Not enough production LLM experience*).
3. On the second analysis: **Invite to an interview**, a date and a time.
4. On the third analysis: **Shortlist**, and set the grade to 3.

- [ ] 1 is refused: *A rejection needs a written reason…*; 2, 3 and 4 are saved.
- [ ] *Decisions made*: the 3 decisions with reviewer, notes, interview date.
- [ ] Notifications: **🤖 Interview invitation prepared for …** (the full invitation text), **🤖 Answer prepared for …**
  (the polite answer, with your reason marked *internal, not sent*), **🤖 Shortlisted: contact … within five working
  days**.
- [ ] n8n → *Recruitment — Events* → **Executions**: one run per event; open the invitation's run → *Prepare the
  message* → output.
- [ ] Under the hood → **Database** → `reviews`: 3 rows; `analyses`: statuses `rejected`, `interview`, `shortlisted`,
  and `final_grade` = 3 on the shortlisted one.
- [ ] Reports → the shortlisted analysis: the *Recruiter decision* section appears (also in the PDF).
- [ ] Now decide on an analysis of C2 (method *AI agent*; the page says *From the application pipeline*):
  **Invite to an interview** → the message says the application is closed; **Application pipeline** → that
  application: every step done, decision *Interview*; Notifications: **Interview invitation prepared for …**
  (written by the pipeline, with the candidate's e-mail).
- [ ] On another C2 analysis: **Put on hold** → the application still waits (*Human validation* stays ⏳).

### C8. The AI assistant (part 5, agent)

**AI assistant**, ask these (30 s to 2 min each); open *How it answered* each time:

| Question | Expected |
|---|---|
| `Who are the best candidates for the GenAI Engineer job?` | tool `rank_candidates`; names the top candidates with grades |
| `Find candidates who have used Kubernetes.` | tool `search_candidates`; names candidates with the evidence |
| `Is Mehdi Trabelsi a good fit for the AWS Cloud Engineer job? What are the gaps?` | tool `match_candidate`; mentions Azure as an equivalent of AWS |
| `How are internships counted in years of experience?` | tool `search_hr_knowledge`; 50 %, at most one year, KB-RR-003 |
| `Combien de temps conserve-t-on les données d'un candidat non retenu ?` | answers in French: 24 months, KB-HR-004 |
| `Among Priya Raman, Daniel Okoro and Hannah Weiss, who has the most hands-on experience with LLM applications?` | tool `rerank_candidates`; Priya Raman |
| `Which of our candidates are women? List their names.` | **refuses** (fair evaluation policy, KB-HR-003); no names |
| `What is the profile of the candidate John Smith?` | says no such candidate exists |
| `Summarize <your name>'s background.` | tool `get_candidate`; your own profile |

- [ ] The answers match (wording varies); the tools used are those expected.
- [ ] Under the hood → **Agent internals** → pick a run: the path through the graph (decide → act → … → respond)
  with times and tokens; the timeline table; the tool results.
- [ ] Under the hood → **Database** → `agent_runs`: one row per question, with its steps.

### C9. n8n: the candidate portal and the other workflows (parts 1, 2, 7)

**Candidate portal** → position *AWS Cloud Engineer* → `data/processed/eval_v1/synthetic/files/syn07.pdf` → accept the
data policy → route **Through n8n** → **Apply**.

- [ ] *Thank you Sofia Lindqvist: your application was received. A recruiter will answer within 10 working days.*
- [ ] n8n → *Recruitment — Application pipeline* → **Executions**: one execution, *waiting* at *Wait for the
  recruiter's decision*; every node before it is green; open *Screening agent (LangGraph)* (the API's answer: jobs
  analysed, tool calls, timeline).
- [ ] App → **Application pipeline**: her application, run by **n8n**, waiting; Notifications: **Acknowledgement
  prepared for Sofia Lindqvist** and **Priority review / Review needed: Sofia Lindqvist for AWS Cloud Engineer**.
- [ ] **Human validation** → her analysis → **Shortlist** → n8n: the same execution resumes: *Final decision*,
  *Decided?*, *Notify the candidate*, *Close the application* turn green; the app: the application is closed,
  Notifications: **Shortlist message prepared for Sofia Lindqvist**.

New job through n8n (PowerShell):

```powershell
$ad = @{ text = "Machine Learning Ops Engineer at Nimbus Analytics, Tunis. Requirements: 2+ years of MLOps, Python, Docker, Kubernetes, MLflow, a cloud platform (AWS or Azure). Nice to have: Terraform and monitoring with Prometheus." } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook/new-job -ContentType "application/json" -Body $ad -TimeoutSec 900
```

- [ ] The command prints the new job; Jobs lists it with source `n8n`; n8n → *New job* → Executions: a green run.

Daily digest: Notifications & n8n → **Run the daily digest now**.

- [ ] Notification **🤖 Recruiting digest — <today>** listing the analyses waiting for a decision.

### C10. Notifications (part 7)

- [ ] Notifications: the unread count; *Read* on one; **Mark all as read** → 0 unread.

### C11. Data storage and vectors, at the end (parts 6, 8)

- [ ] Under the hood → **Database**: candidates 16, jobs 7, applications 4, analyses 7 or more (C2's 3 to 6,
  C5's 3, C9's 1), reviews 6, agent_runs 9 or more; notifications: app events, the pipeline's messages and n8n
  messages.
- [ ] Adminer (http://localhost:8080, System *PostgreSQL*, server `postgres`, user/password `postgres`, database
  `recruitment_db`): the same tables and rows.
- [ ] Qdrant dashboard: `candidates` and `jobs` have more points than after B1 (77 and 27).
- [ ] Under the hood → **Architecture (live)**: every part is green now.

### C12. The evaluation results

**Evaluation** page.

- [ ] *Extraction (before/after fine-tuning)*: overall F1 78.9 % → 81.2 %, experience 88.2 % → 92.9 %, skills
  56.0 % → 62.0 %, 12 → 5 hallucinated fields.
- [ ] *Fine-tuning run*: the loss curve (0.40 → about 0.18).
- [ ] *Retrieval*, *Matching*, *Screening agent*, *AI assistant*: shown once their Kaggle jobs have run (`all rag`,
  `all matching`, `all screening`, `all agent`); until then the page says how to run them.

### C13. Right to erasure (part 8)

Candidates → *All candidates* → open the **uploaded** Karim Haddad (source `upload`, file `syn10.docx`; the other one
is the demo) → *Delete this candidate* → confirm → **Delete**.

- [ ] The candidate disappears; Database → `candidates` 15 rows, and their application and analyses are gone too.
- [ ] Vectors & RAG → Browse `candidates`: their snippets are gone.

---

## Phase D — Robustness (about 10 minutes)

### D1. A wrong token

In `.env`, add one character at the end of `REMOTE_LLM_TOKEN`, then restart the API.

- [ ] Dashboard: 🔴 Kaggle AI server, *The AI server refused the token…*; Matching → analyse: a clear error, no crash.
- [ ] Restore the token, restart the API: 🟢 again.

### D2. n8n can't reach the API

Stop the API (Ctrl+C), then:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:5678/webhook/daily-digest -ContentType "application/json" -Body '{}'
```

- [ ] n8n → *Daily digest* → Executions: a red run (the API is down); *Error alert* → Executions: its run too.
- [ ] Restart the API.

### D3. The Kaggle session stops

Stop the Kaggle session (power icon), then in the app:

- [ ] Dashboard: 🔴 Kaggle AI server with *does not answer… is the Kaggle session still running?*
- [ ] Matching → analyse: a clear error message; AI assistant: an answer saying the model is not reachable.
- [ ] Candidates → Add a candidate (run by the API): the application stops at **Candidate JSON** (❌, with the
  error) and a notification *Application failed at «Candidate JSON»*; after restarting the AI server (and `.env`,
  the API), Application pipeline → **Continue the workflow from this step** → it goes on and waits for a decision.

---

## Phase E — The end

- [ ] The Kaggle session is stopped (done in D3).
- [ ] Ctrl+C in the API window (and `npm run dev`, if used).
- [ ] `docker compose stop` (the data is kept for next time; `docker compose up -d postgres qdrant n8n` restarts it).

---

## Results

| Phase | Steps | Passed | Notes |
|---|---|---|---|
| A — Preparation | A1–A6 | | |
| B — Setup | B1–B3 | | |
| C — Every part | C1–C13 | | |
| D — Robustness | D1–D3 | | |
| E — End | | | |
