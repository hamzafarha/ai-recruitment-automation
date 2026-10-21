# Stage 1: build the React interface (web/) -> web/dist
FROM node:20-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY web/ ./
COPY docs/USER_GUIDE.md /docs/USER_GUIDE.md
RUN npm run build

# Stage 2: the FastAPI backend, which also serves the built interface at /
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
    pip install --no-cache-dir ".[api,database,vectorstore,document-processing,agents,google]" reportlab

# Copy source code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY --from=web /web/dist ./web/dist

# Install package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
