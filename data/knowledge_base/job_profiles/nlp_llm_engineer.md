---
doc_id: KB-JP-002
title: Job Profile — NLP / LLM Engineer
category: job_profiles
language: en
roles: [nlp_llm_engineer]
version: 1.0
last_reviewed: 2026-10-15
---

# Job Profile — NLP / LLM Engineer

## Mission

Build language-based AI systems: information extraction, classification, semantic search, retrieval-augmented
generation (RAG) and LLM-based agents. The NLP/LLM Engineer makes these systems accurate, grounded in sources and
measurable.

## Responsibilities

- Design RAG pipelines: document parsing, chunking, embeddings, vector search, reranking, generation.
- Write and evaluate prompts with structured outputs (JSON schemas).
- Fine-tune or adapt language models (LoRA, QLoRA) when prompting is not enough.
- Build evaluation sets and measure retrieval and generation quality.
- Develop tool-using agents and integrate them into products.

## Must-have requirements

- **M1** Python, with experience shipping NLP or LLM features.
- **M2** Transformer-based models and the Hugging Face ecosystem (transformers, tokenizers).
- **M3** Embeddings and semantic search (vector databases such as Qdrant, FAISS, Milvus or pgvector).
- **M4** Prompt engineering with structured outputs and an evaluation method.
- **M5** At least one of: text classification, named entity recognition, information extraction.

## Nice-to-have requirements

- **N1** RAG in production, including reranking (cross-encoders) and hybrid search (BM25 + dense).
- **N2** Fine-tuning with LoRA/QLoRA, PEFT, TRL.
- **N3** Agent frameworks: LangChain, LangGraph, or custom tool calling.
- **N4** Multilingual NLP, especially French and English.
- **N5** Multimodal models (vision-language, OCR pipelines).

## Equivalent and transferable skills

- Qdrant, Milvus, Weaviate, FAISS and pgvector are equivalent for M3.
- Classic NLP (spaCy, CRF, BiLSTM) counts as partial evidence for M2 if combined with recent transformer projects.
- An ML Engineer with no text experience usually meets M1 only: grade 1.
- Research experience (papers, theses) in NLP counts as evidence for M2 and M5.

## Experience by level

- **Junior**: NLP projects or a thesis using transformers; one deployed demo is a strong signal.
- **Mid-level**: 2–5 years, including at least one NLP or LLM system used by real users.
- **Senior**: 5+ years, has owned the evaluation strategy of an LLM product.

## Red flags

- "LLM experience" limited to calling a chat interface, with no evaluation or system design.
- No mention of how outputs were evaluated or how hallucinations were controlled.

## Role-specific interview questions

- "How did you choose the chunk size and the number of retrieved passages in your RAG system?"
- "How do you detect and reduce hallucinations in generated answers?"
- "When would you fine-tune rather than improve the prompt or the retrieval?"
