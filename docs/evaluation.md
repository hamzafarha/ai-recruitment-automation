# Evaluation

Every evaluation runs on Kaggle's GPUs (`python scripts/kaggle_run.py all <job>`) on the frozen evaluation set
`data/processed/eval_v1/` ([datasheet](../data/processed/eval_v1/README.md)): tune on **dev**, report **test**.
Results and their reading: [final report, section 4](reports/final_report.md). Reports are written to
`data/evaluation/<job>/report.md` and shown on the interface's **Evaluation** page.

| Objective | Job | What is measured |
|---|---|---|
| DSO1 — CV extraction | `extraction`, `extraction-text`, `extraction-vlm`, `extraction-ft` | field-level F1 for experience, education and skills; exact dates; identity accuracy and hallucinations; seconds per CV — by input type, source and language |
| DSO1 — fine-tuning | `finetune` | QLoRA training run (loss, steps, held-out check); then `extraction-ft` = same evaluation with the adapter |
| DSO2 — retrieval (RAG) | `rag` | 40 questions (EN/FR, cross-lingual) on the HR knowledge base: Hit@1/3/5, Recall@5, MRR for dense, + reranker, + role filter, both |
| DSO3 — matching | `matching` | 36 graded CV-job pairs: grade accuracy, within-one accuracy, MAE, quadratic weighted kappa, Spearman; must-have status accuracy and macro-F1; fast matcher thresholds calibrated on dev |
| DSO4 — screening agent | `screening` | the application pipeline's agent on the same 36 pairs as DSO3, next to the fixed analysis (same model): grade and must-have status metrics, plus Qwen and tool calls per pair, planning rounds, claims changed by its checks, fallbacks. The fixed analysis' saved answers are reused, so only the agent runs |
| DSO4 — AI assistant | `agent` | 20 scripted questions: right first tool, expected answer, grounded candidate names, refusals of discriminatory requests, tool/model calls and seconds |

Code: `src/recruitment_ai/evaluation/` (metrics, comparison of runs), `scripts/eval_*.py`, `scripts/kaggle/`.
