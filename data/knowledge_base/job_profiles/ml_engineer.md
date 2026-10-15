---
doc_id: KB-JP-001
title: Job Profile — Machine Learning Engineer
category: job_profiles
language: en
roles: [ml_engineer]
version: 1.0
last_reviewed: 2026-10-15
---

# Job Profile — Machine Learning Engineer

## Mission

Design, train, deploy and maintain machine learning models that run reliably in production. The ML Engineer sits
between data science and software engineering: models must be accurate, but also tested, versioned, monitored and
efficient.

## Responsibilities

- Build training and evaluation pipelines for supervised and deep learning models.
- Turn research prototypes into production services (APIs, batch jobs).
- Version data, models and experiments; monitor model performance and drift.
- Optimise inference cost and latency (quantisation, batching, hardware choice).
- Work with data engineers on features and with product teams on success metrics.

## Must-have requirements

- **M1** Python in production, with testing and code review practices.
- **M2** Machine learning fundamentals: supervised learning, validation, overfitting, metrics.
- **M3** At least one deep learning framework: PyTorch or TensorFlow/Keras.
- **M4** Deploying a model behind an API or in a batch pipeline (e.g. FastAPI, Flask, Docker).
- **M5** SQL and data manipulation (pandas or Spark).

## Nice-to-have requirements

- **N1** MLOps tools: MLflow, experiment tracking, model registry.
- **N2** Containers and orchestration: Docker, Kubernetes.
- **N3** A cloud platform: AWS, Azure or GCP.
- **N4** LLM fine-tuning (LoRA, QLoRA) or retrieval-augmented generation.
- **N5** Monitoring: Prometheus, Grafana, drift detection.

## Equivalent and transferable skills

- PyTorch, TensorFlow and JAX are equivalent for M3.
- AWS, Azure and GCP are equivalent for N3.
- A data scientist who deployed their own models satisfies M4.
- A backend developer with strong Python and deployment skills but no ML practice meets M1 and M4 only: usually
  grade 1 or 2.

## Experience by level

- **Junior**: ML projects (academic or personal) plus an internship; M1–M3 at least listed.
- **Mid-level**: 2–5 years with at least one model in production.
- **Senior**: 5+ years, has designed ML systems end to end and mentored others.

## Red flags

- Only notebook or Kaggle experience, with no deployment for a mid-level or senior position.
- Long lists of ML libraries with no project that uses them.
- Metrics reported without baseline or validation method.

## Role-specific interview questions

- "How did you choose the evaluation metric for your last model, and what was the baseline?"
- "Your model performs well offline but badly in production. How do you investigate?"
- "How do you version a model together with the data it was trained on?"
