# Retrieval evaluation — HR knowledge base

40 questions · 126 indexed chunks · BGE-M3 dense retrieval, optional role filter, optional BGE-reranker-v2-m3

| Configuration | Group | Questions | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |
|---|---|---|---|---|---|---|---|
| dense | all | 40 | 82% | 98% | 100% | 99% | 91% |
| dense | language=en | 30 | 80% | 100% | 100% | 99% | 90% |
| dense | language=fr | 10 | 90% | 90% | 100% | 100% | 92% |
| dense | cross_lingual | 11 | 82% | 100% | 100% | 100% | 91% |
| dense | role-specific questions | 13 | 85% | 100% | 100% | 100% | 92% |
| dense+rerank | all | 40 | 90% | 98% | 98% | 98% | 94% |
| dense+rerank | language=en | 30 | 93% | 100% | 100% | 100% | 97% |
| dense+rerank | language=fr | 10 | 80% | 90% | 90% | 90% | 85% |
| dense+rerank | cross_lingual | 11 | 82% | 100% | 100% | 100% | 89% |
| dense+rerank | role-specific questions | 13 | 92% | 100% | 100% | 100% | 96% |
| dense+role_filter | all | 40 | 85% | 98% | 100% | 99% | 92% |
| dense+role_filter | language=en | 30 | 83% | 100% | 100% | 99% | 92% |
| dense+role_filter | language=fr | 10 | 90% | 90% | 100% | 100% | 92% |
| dense+role_filter | cross_lingual | 11 | 82% | 100% | 100% | 100% | 91% |
| dense+role_filter | role-specific questions | 13 | 92% | 100% | 100% | 100% | 96% |
| dense+role_filter+rerank | all | 40 | 90% | 98% | 98% | 98% | 94% |
| dense+role_filter+rerank | language=en | 30 | 93% | 100% | 100% | 100% | 97% |
| dense+role_filter+rerank | language=fr | 10 | 80% | 90% | 90% | 90% | 85% |
| dense+role_filter+rerank | cross_lingual | 11 | 82% | 100% | 100% | 100% | 89% |
| dense+role_filter+rerank | role-specific questions | 13 | 92% | 100% | 100% | 100% | 96% |

Best configuration (MRR): **dense+rerank**.

Questions where **dense+rerank** misses the expected document in the top 3:
- Q37 (fr) "Comment évaluer le leadership d'un candidat à partir de son CV ?" — expected ['KB-SD-002'], got ['KB-RR-002', 'KB-RR-003', 'KB-RR-001']
