"""scripts/mlflow_sync.py: what it logs into MLflow from data/evaluation/ (without MLflow itself)."""

from scripts.mlflow_sync import EXPERIMENTS, flatten, metric_key, planned_runs


def test_metric_names_are_valid_for_mlflow():
    assert metric_key("language=fr", "hit@1") == "language_fr.hit_at_1"
    assert metric_key("dense+rerank", "all", "mrr") == "dense_plus_rerank.all.mrr"
    assert metric_key("category", "role (specific)") == "category.role _specific"
    assert flatten({"a": {"hit@3": 0.5, "ok": True, "name": "x"}, "n": 2}) == {"a.hit_at_3": 0.5, "n": 2.0}


def test_every_evaluation_becomes_a_run():
    runs = planned_runs()
    assert {r["experiment"] for r in runs} == set(EXPERIMENTS)
    keys = [r["key"] for r in runs]
    assert len(keys) == len(set(keys))  # one run per evaluation, found again by its key at the next sync
    base = next(r for r in runs if r["key"] == "extraction/ocr_llm_Qwen2.5-7B-Instruct_dev")
    tuned = next(r for r in runs if r["key"] == "extraction/ocr_llm_Qwen2.5-7B-Instruct-finetuned_dev")
    assert base["tags"]["model.variant"] == "base" and tuned["tags"]["model.variant"] == "fine-tuned (QLoRA)"
    assert "overall_f1" in base["metrics"] and "by.input_type.scanned_pdf.overall_f1" in tuned["metrics"]
    finetune = next(r for r in runs if r["experiment"] == "finetune")
    assert finetune["steps"]["loss"][0][0] == 5 and finetune["params"]["LORA_R"] == 16
    rerank = next(r for r in runs if r["key"] == "rag/dense+rerank")
    assert rerank["metrics"]["all.hit_at_1"] > 0 and rerank["params"]["reranker"].startswith("BAAI/")
    agent = next(r for r in runs if r["key"].startswith("matching/agent"))
    assert "test.grades.accuracy" in agent["metrics"] and "test.requirements.macro_f1" in agent["metrics"]
    assert all(all("@" not in k and "=" not in k for k in r["metrics"]) for r in runs)
