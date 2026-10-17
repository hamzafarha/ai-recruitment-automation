# QLoRA fine-tuning report

- Status: **complete**
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Training CVs: 1000 from the Resume Parser train split (silver labels; evaluation CVs excluded)
- Steps: 110/110 on 2 GPU(s)
- Training loss: 0.3985 at the start → 0.1824 at the end
- Quick check on 5 held-out CVs: overall F1 0.7924666928919167

Details: `params.json`, `data_report.json`, `training_state.json`, `smoke_test.json`.

Next: evaluate on the frozen evaluation set: `python scripts/kaggle_run.py all extraction-ft`.
