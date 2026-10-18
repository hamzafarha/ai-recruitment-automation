# Matching evaluation (DSO3, and the screening agent: DSO4)

36 graded CV-job pairs of eval_v1 (dev 14, test 22). Reference: silver labels made by applying the rubric KB-RR-002 (grade 0-3 and met / partial / missing per must-have). Candidate profiles are the answer keys, so this measures matching alone, without extraction errors.

**Semantic thresholds calibrated on dev** (reranker probability): demonstrated >= 0.15, equivalent >= 0.02.

## Test pairs (final numbers)

| System | Pairs | Grade accuracy | Within one grade | MAE | Quadratic kappa | Spearman | Must-have status accuracy | Must-have macro-F1 |
|---|---|---|---|---|---|---|---|---|
| semantic (default thresholds) | 22 | 41% | 82% | 0.77 | 0.40 | 0.77 | 57% | 0.50 |
| semantic (calibrated on dev) | 22 | 41% | 82% | 0.77 | 0.40 | 0.76 | 60% | 0.53 |
| llm (Qwen2.5-7B + RAG) | 22 | 36% | 95% | 0.68 | 0.54 | 0.76 | 55% | 0.41 |
| agent (LangGraph screening) | 22 | 41% | 77% | 0.91 | 0.18 | 0.44 | 55% | 0.47 |

## Dev pairs (used for calibration)

| System | Pairs | Grade accuracy | Within one grade | MAE | Quadratic kappa | Spearman | Must-have status accuracy | Must-have macro-F1 |
|---|---|---|---|---|---|---|---|---|
| semantic (default thresholds) | 14 | 50% | 86% | 0.64 | 0.17 | 0.71 | 64% | 0.62 |
| semantic (calibrated on dev) | 14 | 50% | 86% | 0.64 | 0.17 | 0.76 | 66% | 0.64 |
| llm (Qwen2.5-7B + RAG) | 14 | 50% | 93% | 0.57 | 0.36 | 0.70 | 48% | 0.41 |
| agent (LangGraph screening) | 14 | 43% | 93% | 0.64 | 0.37 | 0.50 | 57% | 0.50 |

## All pairs

| System | Pairs | Grade accuracy | Within one grade | MAE | Quadratic kappa | Spearman | Must-have status accuracy | Must-have macro-F1 |
|---|---|---|---|---|---|---|---|---|
| semantic (default thresholds) | 36 | 44% | 83% | 0.72 | 0.34 | 0.73 | 60% | 0.56 |
| semantic (calibrated on dev) | 36 | 44% | 83% | 0.72 | 0.34 | 0.74 | 62% | 0.58 |
| llm (Qwen2.5-7B + RAG) | 36 | 42% | 94% | 0.64 | 0.49 | 0.74 | 52% | 0.41 |
| agent (LangGraph screening) | 36 | 42% | 83% | 0.81 | 0.23 | 0.45 | 56% | 0.48 |

## Per pair — semantic (default thresholds)

| Pair | Split | Job | CV | Reference | Predicted | Score | Must-haves agreeing |
|---|---|---|---|---|---|---|---|
| P01 | dev | backend_developer | syn04 | 1 | 1 | 28.3 | 6/10 |
| P02 | dev | backend_developer | syn08 | 2 | 1 | 49 | 8/10 |
| P03 | test | backend_developer | resume_072d5c5e36d5bf4a | 1 | 1 | 31.3 | 4/10 |
| P04 | test | backend_developer | syn07 | 2 | 1 | 40.3 | 4/10 |
| P05 | dev | backend_developer | resume_de422f4e08059b5c | 1 | 1 | 37.9 | 7/10 |
| P06 | dev | backend_developer | syn12 | 1 | 0 | 0 | 8/10 |
| P07 | test | backend_developer | resume_10355e4023a15b00 | 1 | 1 | 29.8 | 3/10 |
| P08 | test | backend_developer | resume_5e6dafcd0ad6e829 | 1 | 0 | 8.1 | 7/10 |
| P09 | test | data_analyst_scientist | syn03 | 1 | 1 | 30.2 | 4/5 |
| P10 | test | data_analyst_scientist | resume_641f95f66aa9bdf6 | 0 | 0 | 0 | 5/5 |
| P11 | test | data_analyst_scientist | resume_4d073f3dad6b06af | 1 | 1 | 25 | 3/5 |
| P12 | dev | data_analyst_scientist | resume_257780038de667ad | 1 | 1 | 42.9 | 3/5 |
| P13 | dev | data_analyst_scientist | syn10 | 1 | 1 | 28.3 | 3/5 |
| P14 | test | data_analyst_scientist | syn09 | 3 | 2 | 63.7 | 3/5 |
| P15 | test | data_analyst_scientist | resume_047360d8d19a0da9 | 1 | 0 | 17.7 | 2/5 |
| P16 | dev | devops_cloud | syn04 | 1 | 1 | 32.6 | 7/9 |
| P17 | test | devops_cloud | resume_a23abb2ec16c9adf | 1 | 0 | 9.4 | 8/9 |
| P18 | dev | devops_cloud | syn06 | 2 | 1 | 49 | 4/9 |
| P19 | test | devops_cloud | resume_7ea43a25f45978af | 2 | 2 | 74 | 6/9 |
| P20 | dev | devops_cloud | resume_b4c8313e592846fb | 1 | 0 | 24 | 6/9 |
| P21 | test | devops_cloud | resume_7dd08e298826bb63 | 1 | 0 | 7.1 | 5/9 |
| P22 | test | devops_cloud | syn05 | 3 | 1 | 49 | 5/9 |
| P23 | test | ml_engineer | syn01 | 3 | 1 | 25 | 1/5 |
| P24 | test | ml_engineer | syn03 | 3 | 1 | 49 | 3/5 |
| P25 | dev | ml_engineer | syn04 | 1 | 1 | 28.3 | 4/5 |
| P26 | test | ml_engineer | syn11 | 3 | 1 | 48.2 | 2/5 |
| P27 | dev | ml_engineer | resume_586ec4393d3caf9b | 3 | 1 | 46 | 2/5 |
| P28 | test | ml_engineer | resume_1e2b6603d0fdbc13 | 1 | 1 | 25 | 4/5 |
| P29 | test | ml_engineer | resume_377f4092a0bba616 | 1 | 0 | 10.6 | 3/5 |
| P30 | test | nlp_llm_engineer | resume_6a4a1759d0497497 | 1 | 1 | 28.6 | 6/8 |
| P31 | dev | nlp_llm_engineer | syn04 | 1 | 1 | 30.7 | 6/8 |
| P32 | test | nlp_llm_engineer | syn11 | 1 | 0 | 18.9 | 3/8 |
| P33 | dev | nlp_llm_engineer | syn02 | 3 | 1 | 42.5 | 2/8 |
| P34 | dev | nlp_llm_engineer | resume_71af00e1aadacd42 | 1 | 0 | 12.1 | 5/8 |
| P35 | test | nlp_llm_engineer | syn01 | 3 | 2 | 61.4 | 5/8 |
| P36 | test | nlp_llm_engineer | syn03 | 1 | 1 | 25 | 4/8 |

Average time per pair: 0.2 s.

## Per pair — semantic (calibrated on dev)

| Pair | Split | Job | CV | Reference | Predicted | Score | Must-haves agreeing |
|---|---|---|---|---|---|---|---|
| P01 | dev | backend_developer | syn04 | 1 | 1 | 28.3 | 6/10 |
| P02 | dev | backend_developer | syn08 | 2 | 1 | 49 | 9/10 |
| P03 | test | backend_developer | resume_072d5c5e36d5bf4a | 1 | 1 | 31.3 | 4/10 |
| P04 | test | backend_developer | syn07 | 2 | 1 | 45.4 | 6/10 |
| P05 | dev | backend_developer | resume_de422f4e08059b5c | 1 | 1 | 44 | 8/10 |
| P06 | dev | backend_developer | syn12 | 1 | 0 | 0 | 8/10 |
| P07 | test | backend_developer | resume_10355e4023a15b00 | 1 | 1 | 40 | 5/10 |
| P08 | test | backend_developer | resume_5e6dafcd0ad6e829 | 1 | 0 | 8.1 | 7/10 |
| P09 | test | data_analyst_scientist | syn03 | 1 | 1 | 30.2 | 4/5 |
| P10 | test | data_analyst_scientist | resume_641f95f66aa9bdf6 | 0 | 0 | 0 | 5/5 |
| P11 | test | data_analyst_scientist | resume_4d073f3dad6b06af | 1 | 1 | 25 | 3/5 |
| P12 | dev | data_analyst_scientist | resume_257780038de667ad | 1 | 1 | 42.9 | 3/5 |
| P13 | dev | data_analyst_scientist | syn10 | 1 | 1 | 28.3 | 3/5 |
| P14 | test | data_analyst_scientist | syn09 | 3 | 2 | 63.7 | 3/5 |
| P15 | test | data_analyst_scientist | resume_047360d8d19a0da9 | 1 | 0 | 17.7 | 2/5 |
| P16 | dev | devops_cloud | syn04 | 1 | 1 | 32.6 | 7/9 |
| P17 | test | devops_cloud | resume_a23abb2ec16c9adf | 1 | 0 | 9.4 | 8/9 |
| P18 | dev | devops_cloud | syn06 | 2 | 1 | 49 | 4/9 |
| P19 | test | devops_cloud | resume_7ea43a25f45978af | 2 | 2 | 74 | 6/9 |
| P20 | dev | devops_cloud | resume_b4c8313e592846fb | 1 | 0 | 24 | 6/9 |
| P21 | test | devops_cloud | resume_7dd08e298826bb63 | 1 | 0 | 11.8 | 4/9 |
| P22 | test | devops_cloud | syn05 | 3 | 1 | 49 | 5/9 |
| P23 | test | ml_engineer | syn01 | 3 | 1 | 25 | 1/5 |
| P24 | test | ml_engineer | syn03 | 3 | 1 | 49 | 3/5 |
| P25 | dev | ml_engineer | syn04 | 1 | 1 | 28.3 | 4/5 |
| P26 | test | ml_engineer | syn11 | 3 | 1 | 49 | 2/5 |
| P27 | dev | ml_engineer | resume_586ec4393d3caf9b | 3 | 1 | 46 | 2/5 |
| P28 | test | ml_engineer | resume_1e2b6603d0fdbc13 | 1 | 1 | 25 | 4/5 |
| P29 | test | ml_engineer | resume_377f4092a0bba616 | 1 | 0 | 10.6 | 3/5 |
| P30 | test | nlp_llm_engineer | resume_6a4a1759d0497497 | 1 | 1 | 38.1 | 7/8 |
| P31 | dev | nlp_llm_engineer | syn04 | 1 | 1 | 30.7 | 6/8 |
| P32 | test | nlp_llm_engineer | syn11 | 1 | 0 | 18.9 | 3/8 |
| P33 | dev | nlp_llm_engineer | syn02 | 3 | 1 | 47.2 | 2/8 |
| P34 | dev | nlp_llm_engineer | resume_71af00e1aadacd42 | 1 | 0 | 14.4 | 5/8 |
| P35 | test | nlp_llm_engineer | syn01 | 3 | 2 | 68.5 | 6/8 |
| P36 | test | nlp_llm_engineer | syn03 | 1 | 1 | 34.2 | 4/8 |

Average time per pair: 0.2 s.

## Per pair — llm (Qwen2.5-7B + RAG)

| Pair | Split | Job | CV | Reference | Predicted | Score | Must-haves agreeing |
|---|---|---|---|---|---|---|---|
| P01 | dev | backend_developer | syn04 | 1 | 0 | 20.2 | 5/10 |
| P02 | dev | backend_developer | syn08 | 2 | 1 | 49 | 6/10 |
| P03 | test | backend_developer | resume_072d5c5e36d5bf4a | 1 | 1 | 43.5 | 4/10 |
| P04 | test | backend_developer | syn07 | 2 | 1 | 45.5 | 3/10 |
| P05 | dev | backend_developer | resume_de422f4e08059b5c | 1 | 1 | 44.5 | 6/10 |
| P06 | dev | backend_developer | syn12 | 1 | 0 | 17.2 | 7/10 |
| P07 | test | backend_developer | resume_10355e4023a15b00 | 1 | 1 | 38.5 | 4/10 |
| P08 | test | backend_developer | resume_5e6dafcd0ad6e829 | 1 | 0 | 13.2 | 7/10 |
| P09 | test | data_analyst_scientist | syn03 | 1 | 1 | 49 | 1/5 |
| P10 | test | data_analyst_scientist | resume_641f95f66aa9bdf6 | 0 | 0 | 0 | 5/5 |
| P11 | test | data_analyst_scientist | resume_4d073f3dad6b06af | 1 | 0 | 15.9 | 3/5 |
| P12 | dev | data_analyst_scientist | resume_257780038de667ad | 1 | 1 | 49 | 2/5 |
| P13 | dev | data_analyst_scientist | syn10 | 1 | 1 | 33.6 | 2/5 |
| P14 | test | data_analyst_scientist | syn09 | 3 | 2 | 74 | 5/5 |
| P15 | test | data_analyst_scientist | resume_047360d8d19a0da9 | 1 | 1 | 26.6 | 3/5 |
| P16 | dev | devops_cloud | syn04 | 1 | 0 | 10.6 | 4/9 |
| P17 | test | devops_cloud | resume_a23abb2ec16c9adf | 1 | 0 | 10.6 | 5/9 |
| P18 | dev | devops_cloud | syn06 | 2 | 1 | 49 | 3/9 |
| P19 | test | devops_cloud | resume_7ea43a25f45978af | 2 | 1 | 49 | 5/9 |
| P20 | dev | devops_cloud | resume_b4c8313e592846fb | 1 | 1 | 34.2 | 2/9 |
| P21 | test | devops_cloud | resume_7dd08e298826bb63 | 1 | 0 | 10.6 | 4/9 |
| P22 | test | devops_cloud | syn05 | 3 | 2 | 60.2 | 5/9 |
| P23 | test | ml_engineer | syn01 | 3 | 2 | 74 | 5/5 |
| P24 | test | ml_engineer | syn03 | 3 | 2 | 74 | 5/5 |
| P25 | dev | ml_engineer | syn04 | 1 | 1 | 33.6 | 3/5 |
| P26 | test | ml_engineer | syn11 | 3 | 2 | 74 | 5/5 |
| P27 | dev | ml_engineer | resume_586ec4393d3caf9b | 3 | 2 | 74 | 5/5 |
| P28 | test | ml_engineer | resume_1e2b6603d0fdbc13 | 1 | 2 | 70.8 | 1/5 |
| P29 | test | ml_engineer | resume_377f4092a0bba616 | 1 | 2 | 70.8 | 0/5 |
| P30 | test | nlp_llm_engineer | resume_6a4a1759d0497497 | 1 | 1 | 29.5 | 5/8 |
| P31 | dev | nlp_llm_engineer | syn04 | 1 | 1 | 44.9 | 0/8 |
| P32 | test | nlp_llm_engineer | syn11 | 1 | 1 | 27.1 | 4/8 |
| P33 | dev | nlp_llm_engineer | syn02 | 3 | 1 | 47.2 | 4/8 |
| P34 | dev | nlp_llm_engineer | resume_71af00e1aadacd42 | 1 | 1 | 29.5 | 4/8 |
| P35 | test | nlp_llm_engineer | syn01 | 3 | 1 | 49 | 5/8 |
| P36 | test | nlp_llm_engineer | syn03 | 1 | 1 | 34.2 | 3/8 |

## Per pair — agent (LangGraph screening)

| Pair | Split | Job | CV | Reference | Predicted | Score | Must-haves agreeing |
|---|---|---|---|---|---|---|---|
| P01 | dev | backend_developer | syn04 | 1 | 0 | 19.2 | 8/10 |
| P02 | dev | backend_developer | syn08 | 2 | 1 | 39.5 | 6/10 |
| P03 | test | backend_developer | resume_072d5c5e36d5bf4a | 1 | 1 | 31.3 | 4/10 |
| P04 | test | backend_developer | syn07 | 2 | 1 | 32.4 | 5/10 |
| P05 | dev | backend_developer | resume_de422f4e08059b5c | 1 | 1 | 34.4 | 6/10 |
| P06 | dev | backend_developer | syn12 | 1 | 0 | 4 | 9/10 |
| P07 | test | backend_developer | resume_10355e4023a15b00 | 1 | 1 | 40 | 5/10 |
| P08 | test | backend_developer | resume_5e6dafcd0ad6e829 | 1 | 0 | 8.1 | 7/10 |
| P09 | test | data_analyst_scientist | syn03 | 1 | 0 | 10.6 | 3/5 |
| P10 | test | data_analyst_scientist | resume_641f95f66aa9bdf6 | 0 | 0 | 4.5 | 5/5 |
| P11 | test | data_analyst_scientist | resume_4d073f3dad6b06af | 1 | 0 | 15.9 | 3/5 |
| P12 | dev | data_analyst_scientist | resume_257780038de667ad | 1 | 1 | 40.7 | 2/5 |
| P13 | dev | data_analyst_scientist | syn10 | 1 | 1 | 29.8 | 3/5 |
| P14 | test | data_analyst_scientist | syn09 | 3 | 2 | 72.6 | 4/5 |
| P15 | test | data_analyst_scientist | resume_047360d8d19a0da9 | 1 | 1 | 30.1 | 5/5 |
| P16 | dev | devops_cloud | syn04 | 1 | 0 | 7.1 | 5/9 |
| P17 | test | devops_cloud | resume_a23abb2ec16c9adf | 1 | 0 | 9.4 | 8/9 |
| P18 | dev | devops_cloud | syn06 | 2 | 1 | 49 | 4/9 |
| P19 | test | devops_cloud | resume_7ea43a25f45978af | 2 | 2 | 74 | 6/9 |
| P20 | dev | devops_cloud | resume_b4c8313e592846fb | 1 | 1 | 37.8 | 5/9 |
| P21 | test | devops_cloud | resume_7dd08e298826bb63 | 1 | 0 | 11.8 | 4/9 |
| P22 | test | devops_cloud | syn05 | 3 | 1 | 45.4 | 3/9 |
| P23 | test | ml_engineer | syn01 | 3 | 0 | 14.9 | 1/5 |
| P24 | test | ml_engineer | syn03 | 3 | 0 | 4.3 | 0/5 |
| P25 | dev | ml_engineer | syn04 | 1 | 1 | 26.6 | 2/5 |
| P26 | test | ml_engineer | syn11 | 3 | 1 | 30.9 | 1/5 |
| P27 | dev | ml_engineer | resume_586ec4393d3caf9b | 3 | 2 | 74 | 4/5 |
| P28 | test | ml_engineer | resume_1e2b6603d0fdbc13 | 1 | 1 | 25 | 4/5 |
| P29 | test | ml_engineer | resume_377f4092a0bba616 | 1 | 1 | 29.8 | 1/5 |
| P30 | test | nlp_llm_engineer | resume_6a4a1759d0497497 | 1 | 1 | 38.1 | 7/8 |
| P31 | dev | nlp_llm_engineer | syn04 | 1 | 0 | 13 | 2/8 |
| P32 | test | nlp_llm_engineer | syn11 | 1 | 0 | 13 | 3/8 |
| P33 | dev | nlp_llm_engineer | syn02 | 3 | 1 | 25 | 2/8 |
| P34 | dev | nlp_llm_engineer | resume_71af00e1aadacd42 | 1 | 1 | 25 | 5/8 |
| P35 | test | nlp_llm_engineer | syn01 | 3 | 1 | 49 | 5/8 |
| P36 | test | nlp_llm_engineer | syn03 | 1 | 1 | 27.4 | 3/8 |

Average time per pair: 90.0 s.

## Screening agent (DSO4): what it did

- 36 pairs; per pair 2.92 Qwen calls and 16.94 tool calls on average
- planning rounds by Qwen (weak evidence after the systematic first round): 1.92 per pair, on 36 pairs
- checks by code that changed the model's judgement: 73 on 24 pairs (evidence the model left out: 3, quote not supporting the requirement: 68, skill only listed: 2)
- analyses that fell back to the fast estimate: 7
- Compare the **agent** line with the **llm** line above: same model, same pairs; the difference is the agent (retrieval per requirement, iterations, checks).

## How to read

- **Within one grade** matters most for screening: a grade-2 candidate predicted 3 (or 1) still reaches the recruiter's review; 0 vs 3 would not.
- **Quadratic kappa** penalises big disagreements more than small ones (1 = perfect, 0 = chance).
- The labels are silver (model-made with the written rubric): treat differences of a few points as noise; adding human labels on the same pairs is the next step.

## Notes

- 7 screening agent analyses failed and fell back to the semantic estimate
