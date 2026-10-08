# AI Recruitment Automation — System Architecture

This document details the architectural design and data flow for the end-to-end AI Recruitment Automation System.

---

## 1. High-Level Pipeline Architecture

```text
Candidate CV (PDF / DOCX / Image)
            │
            ▼
┌─────────────────────────────────┐
│ 1. DOCUMENT INGESTION           │
│    PyMuPDF (PDF), python-docx   │
│    PaddleOCR (scanned CVs)      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 2. DOCUMENT PROCESSING          │
│    Unicode normalization        │
│    Whitespace & control clean   │
│    Layout & metadata extraction │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 3. STRUCTURED EXTRACTION        │
│    Transformer / LLM extraction │
│    Pydantic Schema validation   │
│    (CandidateProfile v1.0.0)    │
└──────────────┬──────────────────┘
               │
         ┌─────┴──────────────┐
         ▼                    ▼
 Structured Profile       Raw Text
         │                    │
         └─────────┬──────────┘
                   ▼
┌─────────────────────────────────┐
│ 4. KNOWLEDGE / RAG INGESTION    │
│    Recursive chunking           │
│    Metadata provenance tagging  │
│    Sentence Transformers        │
│    Qdrant Vector Store          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 5. RETRIEVAL & RERANKING        │
│    Dense semantic search        │
│    Optional Cohere neural rerank│
│    Context Assembly             │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 6. MULTI-AGENT ORCHESTRATION   │
│    LangGraph State Machine      │
│    - Candidate Analysis Agent   │
│    - Job Analysis Agent         │
│    - Matching Agent             │
│    - Recruiter Assistant Agent  │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 7. HUMAN-IN-THE-LOOP VALIDATION │
│    Recruiter review & decision  │
│    Feedback audit logging       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ 8. REPORTING & RECORDER UI      │
│    Candidate match score        │
│    Strengths & gaps evidence    │
│    Streamlit Recruiter UI       │
└─────────────────────────────────┘
```

---

## 2. Component Specifications

### 2.1 Ingestion & Processing (`src/recruitment_ai/ingestion` & `document_processing`)
- **Format Routing:** `UnifiedDocumentParser` dynamically delegates to `PdfParser` (PyMuPDF), `DocxParser` (python-docx), or `ImageOCRParser` (PaddleOCR).
- **Security:** Filename sanitization, directory traversal defense via strict canonical path checks, and MIME/size limits (default 15MB).
- **Normalization:** Strip non-printable bytes, normalize unicode accents (NFKD), preserve layout structures and extract metadata heuristics.

### 2.2 Extraction & Schemas (`src/recruitment_ai/extraction`)
- **Contract Driven:** Pydantic models guarantee typed, validated outputs without uncontrolled JSON dictionaries.
- **Key Schemas:** `CandidateProfile`, `Education`, `Experience`, `Skill`, `Certification`, `Project`, `Language`, `JobDescription`, `CandidateMatch`, `RecruiterFeedback`, `AnalysisReport`.

### 2.3 Embeddings & Vector Store (`src/recruitment_ai/embeddings` & `vectorstore`)
- **Decoupled Architecture:** `VectorStore` abstract base class decouples business logic from storage engines.
- **Qdrant Implementation:** `QdrantVectorStore` manages collection provisioning, payload upsert, cosine search, and health monitoring.
- **Embedding Model:** `SentenceTransformerEmbeddingService` provides lazy-loaded models with deterministic test fallbacks.

### 2.4 RAG Engine (`src/recruitment_ai/rag` & `reranking`)
- **Metadata Provenance:** `ChunkMetadata` attaches `document_id`, `document_type`, `candidate_id`, `job_id`, `source`, `page`, and `chunk_id`.
- **Reranker:** `OptionalReranker` enables optional neural reranking when `COHERE_API_KEY` is present, with zero-cost fallback passthrough.

### 2.5 Multi-Agent System (`src/recruitment_ai/agents` & `tools`)
- **State Machine:** Explicit `AgentWorkflowState` tracks trace identifiers, step transitions, and stops at `awaiting_human_validation`.
- **Tooling Layer:** `AgentTool` encapsulates tools with input schemas, error isolation, and structured logging.

### 2.6 Persistence & API Layer (`src/recruitment_ai/database` & `api`)
- **Relational Storage:** PostgreSQL handles structured records (Candidates, Jobs, Matches, Feedback) via SQLAlchemy and Alembic migrations.
- **Separation of Concerns:** Business logic lives strictly in `src/recruitment_ai/services`, while FastAPI routers handle request validation, status codes, and routing.
