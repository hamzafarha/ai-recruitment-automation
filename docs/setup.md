# Development Environment Setup Guide

This guide describes how to initialize, configure, and verify the AI Recruitment Automation development environment on Windows 11/10 (PowerShell), Linux, or macOS.

---

## 1. Prerequisites

- **Python:** 3.12 (Check via `python --version`)
- **Docker & Docker Desktop:** Installed and running
- **Git:** Version 2.30+
- **PowerShell:** Default shell on Windows

---

## 2. Windows Development Setup (PowerShell)

### Step 1: Create and Activate Virtual Environment
Open PowerShell inside the project directory:

```powershell
# Create Python 3.12 virtual environment (do not use uv)
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

> **Note on Execution Policy:** If PowerShell blocks script execution, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Step 2: Install Dependencies
Install in editable mode with development dependencies:

```powershell
pip install -e ".[dev]"
```

Or install all optional components:

```powershell
pip install -e ".[all]"
```

### Step 3: Configure Environment Variables
Copy the template `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Review `.env` settings. The default configuration uses the offline mock LLM and CPU embeddings, so no paid API keys are required.

### Step 4: Seed Synthetic Test Fixtures
Generate synthetic test CV and job fixtures:

```powershell
python scripts/seed_data.py
```

---

## 3. Launching Infrastructure Services (Docker)

To launch PostgreSQL and Qdrant locally:

```powershell
docker compose up -d postgres qdrant
```

Check running container status:

```powershell
docker compose ps
```

---

## 4. Running Verification & Health Checks

Run the automated test suite and the system health check:

```powershell
# Run pytest unit tests
pytest

# Run system health check
python scripts/health_check.py
```

---

## 5. Running the Application

### Start FastAPI Backend
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- OpenAPI JSON: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

### Start Streamlit Frontend
In a new terminal (with `.venv` activated):
```powershell
streamlit run frontend/streamlit_app.py
```
Access the recruiter web UI at: [http://localhost:8501](http://localhost:8501)
