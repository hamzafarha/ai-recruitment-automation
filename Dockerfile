FROM python:3.12-slim

# Prevent Python from writing pyc files and keep stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and basic package configuration
COPY pyproject.toml README.md ./

# Install python dependencies with minimal build
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[api,database,vectorstore,document-processing,agents,frontend]" reportlab

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY frontend/ ./frontend/
COPY configs/ ./configs/
COPY alembic/ ./alembic/
COPY alembic.ini ./

# Install package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
