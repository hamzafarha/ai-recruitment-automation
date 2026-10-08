# Evaluation Framework

Quantitative evaluation is built into the design of the AI Recruitment Automation system to measure performance at every stage of the pipeline.

---

## 1. Metrics Taxonomy

### 1.1 Extraction Quality
Evaluates the accuracy of transforming unstructured CV text into the Pydantic `CandidateProfile` schema:
- **Precision:** Fraction of extracted fields that are correct.
- **Recall:** Fraction of ground-truth candidate details recovered.
- **F1 Score:** Harmonic mean of precision and recall.

### 1.2 Information Retrieval
Evaluates vector retrieval and candidate ranking in Qdrant:
- **Recall@K (K=3, 5, 10):** Fraction of relevant document chunks retrieved in top-$K$.
- **MRR (Mean Reciprocal Rank):** Position of the first relevant chunk in search results.
- **NDCG (Normalized Discounted Cumulative Gain):** Quality of ranking taking relevance grading into account.

### 1.3 RAG Quality (RAG Triad)
- **Context Relevance:** How relevant the retrieved chunks are to the user query.
- **Groundedness / Faithfulness:** Whether LLM claims are backed by cited retrieved chunks.
- **Answer Relevance:** How directly the generated report answers the recruiter's prompt.

### 1.4 Operational Telemetry
- **Latency (p50, p95, p99):** Ingestion, retrieval, and inference time in milliseconds.
- **Error Rate:** Percentage of failed extractions or tool executions.
- **Token Efficiency:** Total prompt/completion tokens consumed per candidate processed.
