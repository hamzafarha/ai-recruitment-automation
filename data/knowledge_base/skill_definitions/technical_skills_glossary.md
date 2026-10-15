---
doc_id: KB-SD-001
title: Technical Skills Glossary and Evidence Guide
category: skill_definitions
language: en
roles: [all]
version: 1.0
last_reviewed: 2026-10-15
---

# Technical Skills Glossary and Evidence Guide

For each skill family: what it covers, what counts as **demonstrated** evidence in a CV (weight 1.0 in the scoring
rubric KB-RR-002), and which skills are equivalent or closely related.

## Python and software engineering

- **Covers**: Python 3, packaging, typing, asynchronous code (asyncio), testing (pytest), web frameworks (FastAPI,
  Django, Flask), clean code and code review.
- **Demonstrated when**: the CV describes a Python application, service or library the candidate built or
  maintained, ideally with tests or users.
- **Related**: other backend languages (Java, Go, C#) show general engineering skill but not Python itself.

## SQL and databases

- **Covers**: relational modelling, SQL queries (joins, aggregations, window functions), indexes, transactions;
  PostgreSQL, MySQL, SQL Server, Oracle (PL/SQL, T-SQL); NoSQL stores (MongoDB, Redis, Cassandra).
- **Demonstrated when**: the candidate designed a schema, optimised queries or built reports on a database.
- **Equivalent**: relational databases are interchangeable for most requirements.

## Machine learning

- **Covers**: supervised and unsupervised learning, feature engineering, cross-validation, metrics (accuracy, F1,
  AUC, RMSE), scikit-learn, gradient boosting (XGBoost, LightGBM).
- **Demonstrated when**: a model was trained, evaluated against a baseline, and used for a decision or a product.
- **Synonyms**: ML, statistical learning, apprentissage automatique.

## Deep learning

- **Covers**: neural networks, CNNs, RNNs, transformers; PyTorch (autograd, DataLoader, distributed training with
  DDP), TensorFlow/Keras; GPU training and mixed precision.
- **Demonstrated when**: the candidate trained or fine-tuned a neural network on real data.
- **Equivalent**: PyTorch, TensorFlow and JAX.

## NLP, LLMs and retrieval

- **Covers**: tokenisation, embeddings (e.g. BGE-M3, E5), named entity recognition, classification; large
  language models (Qwen, Llama, Mistral), prompt engineering, structured outputs.
- **Retrieval-augmented generation (RAG)**: chunking, dense retrieval with vector databases (Qdrant, Milvus, FAISS;
  HNSW indexes, cosine similarity), sparse retrieval (BM25), hybrid search, cross-encoder reranking (e.g.
  bge-reranker-v2-m3), grounded generation with citations.
- **Fine-tuning**: supervised fine-tuning (SFT), parameter-efficient methods (LoRA, QLoRA), quantisation
  (bitsandbytes, 4-bit), libraries PEFT and TRL.
- **Demonstrated when**: an NLP or LLM system was built **and evaluated** (retrieval recall, accuracy, human review).

## Data analysis and visualisation

- **Covers**: pandas, NumPy, statistics, A/B testing, BI tools (Power BI, Tableau, Looker), Excel for analysis.
- **Demonstrated when**: an analysis led to a decision, or a dashboard is used by a team.
- **Equivalent**: BI tools are interchangeable; Python and R are interchangeable for analysis.

## Cloud and infrastructure

- **Covers**: AWS (EC2, S3, RDS, Lambda, IAM, VPC), Azure, GCP; infrastructure as code (Terraform,
  CloudFormation); Linux administration.
- **Demonstrated when**: the candidate provisioned or operated production resources. A certification alone is
  "Listed" evidence.
- **Equivalent**: the three major clouds are equivalent for general requirements.

## Containers, CI/CD and observability

- **Covers**: Docker, Kubernetes, Helm; CI/CD (Jenkins, GitLab CI, GitHub Actions, Azure DevOps); monitoring and
  logging (Prometheus, Grafana, ELK, Splunk, Datadog); MLOps (MLflow, model registry).
- **Demonstrated when**: the candidate built or maintained a pipeline, a cluster or a monitoring setup used in
  production.
