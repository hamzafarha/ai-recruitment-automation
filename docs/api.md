# REST API Documentation

The backend service is powered by **FastAPI** with modular routers and strict Pydantic contract validation.

Base URL: `http://localhost:8000/api/v1`

---

## 1. System Health

### `GET /health`
Liveness probe.
- **Response:** `{"status": "ok", "app": "AI Recruitment Automation"}`

### `GET /api/v1/health`
Detailed component readiness probe verifying PostgreSQL, Qdrant, and LLM configuration.
- **Response:**
  ```json
  {
    "status": "healthy",
    "app_name": "AI-Recruitment-Automation",
    "environment": "development",
    "components": {
      "postgresql": {"healthy": true, "details": "PostgreSQL connected successfully."},
      "qdrant": {"healthy": true, "details": "Qdrant connected successfully at http://localhost:6333."},
      "llm_provider": {"configured": "mock"}
    }
  }
  ```

---

## 2. Ingestion & Documents

### `POST /api/v1/documents/upload`
Uploads a candidate document (`.pdf`, `.docx`, `.png`, `.jpg`).
- **Enforces:** File size limit (<15MB), safe filename sanitization, path traversal prevention.
- **Response:**
  ```json
  {
    "status": "success",
    "message": "Document 'resume.pdf' successfully uploaded and validated.",
    "filename": "resume.pdf",
    "size_bytes": 1048576,
    "path": "./data/raw/cvs/resume.pdf"
  }
  ```

---

## 3. Candidates & Jobs

### `GET /api/v1/candidates`
Lists ingested candidates.

### `GET /api/v1/candidates/{candidate_id}`
Retrieves candidate profile and history.

### `POST /api/v1/candidates/extract`
Triggers structured extraction pipeline (`CandidateProfile`).

### `GET /api/v1/jobs`
Lists configured job vacancies and requirement criteria.

---

## 4. Matching, Search & Analysis

### `POST /api/v1/matching`
Executes multi-factor match evaluation (`CandidateMatch`).

### `POST /api/v1/search`
Queries Qdrant dense vector index for candidate or job snippets.

### `POST /api/v1/analysis`
Executes multi-agent LangGraph workflow returning `AnalysisReport`.

---

## 5. Recruiter Review & Feedback

### `POST /api/v1/feedback`
Submits human validation decision (`Approved`, `Rejected`, or `Modified`).
- **Request Body:**
  ```json
  {
    "match_id": "match_uuid",
    "recruiter_id": "recruiter_01",
    "decision": "Approved",
    "rating": 5,
    "comments": "Strong alignment with ML requirements"
  }
  ```
