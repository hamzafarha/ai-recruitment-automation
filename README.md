# AI Recruitment Automation System

> An end-to-end, multi-agent AI system for candidate CV ingestion, structured information extraction, semantic RAG matching, and recruiter decision support.

---

## Architecture Overview

```text
Candidate CV (PDF / DOCX / Image)
            │
            ▼
┌─────────────────────────────────┐
│ 1. DOCUMENT INGESTION           │
│    PyMuPDF, python-docx, OCR    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 2. DOCUMENT PROCESSING          │
│    Cleaning & Normalization     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 3. STRUCTURED EXTRACTION        │
│    Transformers / LLM           │
│    Pydantic CandidateProfile    │
└──────────────┬──────────────────┘
               │
         ┌─────┴──────────────┐
         ▼                    ▼
 Structured Profile       Raw Text
         │                    │
         └─────────┬──────────┘
                   ▼
┌─────────────────────────────────┐
│ 4. KNOWLEDGE & RAG PIPELINE     │
│    Sentence Transformers        │
│    Qdrant Vector Database       │
│    Optional Cohere Reranking    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 5. MULTI-AGENT ORCHESTRATION   │
│    LangGraph State Machine      │
│    Candidate, Job, Match Agents │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 6. HUMAN-IN-THE-LOOP & UI       │
│    Recruiter Review             │
│    Streamlit Web Interface      │
└─────────────────────────────────┘
```

---

## Technology Stack

- **Core:** Python 3.12, Pydantic v2, pytest
- **Document Processing:** PyMuPDF, python-docx, PaddleOCR
- **Machine Learning / NLP:** PyTorch, Hugging Face Transformers, Sentence Transformers
- **Vector Database:** Qdrant (via Docker Compose)
- **Relational Database:** PostgreSQL 16, SQLAlchemy 2.0, Alembic
- **Agents & RAG:** LangChain, LangGraph
- **Backend API:** FastAPI, Uvicorn
- **Frontend:** Streamlit

---

## Windows Quickstart (PowerShell)

### 1. Initialize Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install Project in Editable Mode
```powershell
pip install -e ".[dev]"
```

### 3. Environment Configuration
```powershell
Copy-Item .env.example .env
```
*(Default settings run completely offline with mock LLM and CPU embeddings — no paid API key required!)*

### 4. Seed Synthetic Test Data
```powershell
python scripts/seed_data.py
```

### 5. Launch Background Services (Docker)
```powershell
docker compose up -d postgres qdrant
```

### 6. Run Tests & Health Checks
```powershell
# Run automated tests
pytest

# Verify system health
python scripts/health_check.py
```

### 7. Run Applications

**FastAPI Backend:**
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

**Streamlit Recruiter UI:**
```powershell
streamlit run frontend/streamlit_app.py
```
- Web UI: [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```text
ai-recruitment-automation/
├── configs/             # Configuration defaults
├── data/                # Data directory (raw, interim, processed)
├── models/              # Model artifacts & weights
├── notebooks/           # Exploratory Jupyter notebooks
├── src/recruitment_ai/  # Core package
│   ├── config/          # Pydantic Settings
│   ├── ingestion/       # Document parsers (PDF, DOCX, OCR)
│   ├── document_processing/ # Text cleaner & layout normalizer
│   ├── extraction/      # Pydantic data contract schemas
│   ├── embeddings/      # SentenceTransformer service
│   ├── vectorstore/     # Qdrant vector store adapter
│   ├── rag/             # Chunking, metadata, retrieval pipeline
│   ├── reranking/       # Optional neural reranking
│   ├── matching/        # Multidimensional candidate matching
│   ├── agents/          # LangGraph state machine workflow
│   ├── tools/           # Typed agent tools
│   ├── llm/             # Provider-agnostic LLM adapters
│   ├── evaluation/      # Quantitative benchmark metrics
│   ├── database/        # SQLAlchemy models & repositories
│   ├── services/        # Business logic services
│   └── utils/           # Structured logging & security
├── api/                 # FastAPI routers & application
├── frontend/            # Streamlit recruiter web application
├── tests/               # Pytest unit & integration test suite
├── scripts/             # Health check & setup automation
└── docs/                # Architecture, setup, API & data docs
```

---

## Documentation

- [Architecture Design](docs/architecture.md)
- [Setup & Troubleshooting](docs/setup.md)
- [API Specification](docs/api.md)
- [Data & Privacy Policy](docs/data.md)
- [Evaluation Metrics](docs/evaluation.md)
