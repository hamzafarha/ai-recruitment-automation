"""Tests for comparing two extraction evaluation runs (the app's Evaluation page)."""

import csv
import json

from recruitment_ai.evaluation.compare import (
    compare_documents,
    compare_groups,
    compare_metrics,
    default_pair,
    list_runs,
    load_run,
)


def _write_run(root, name, overall_f1, skills_recall, docs):
    folder = root / name
    folder.mkdir(parents=True)
    overall = {"cvs": len(docs), "overall_f1": overall_f1, "experience": {"f1": 0.88, "date_exact": 0.96},
               "education": {"f1": 0.92}, "skills": {"precision": 0.68, "recall": skills_recall, "f1": 0.56},
               "identity": {"accuracy": 0.93}}
    by = {"input_type": {"scanned_pdf": {"cvs": 2, "overall_f1": overall_f1 - 0.05}}}
    (folder / "metrics.json").write_text(json.dumps({"overall": overall, "by": by}), encoding="utf-8")
    with (folder / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "source", "input_type", "language", "error", "seconds", "exp_f1", "edu_f1",
                         "skills_f1", "identity_accuracy"])
        for doc_id, f1, error in docs:
            writer.writerow([doc_id, "real", "scanned_pdf", "en", error, 60, f1, f1, f1, ""])


def test_runs_are_paired_and_compared(tmp_path):
    _write_run(tmp_path, "ocr_llm_Qwen2.5-7B-Instruct_dev", 0.79, 0.48,
               [("a", 0.5, False), ("b", 0.9, False), ("c", 0.0, True)])
    _write_run(tmp_path, "ocr_llm_Qwen2.5-7B-Instruct-finetuned_dev", 0.85, 0.70,
               [("a", 0.8, False), ("b", 0.85, False), ("c", 0.7, False)])
    names = list_runs(tmp_path)
    before_name, after_name = default_pair(names)
    assert before_name == "ocr_llm_Qwen2.5-7B-Instruct_dev" and after_name.endswith("-finetuned_dev")
    before, after = load_run(tmp_path, before_name), load_run(tmp_path, after_name)

    metrics = {m["metric"]: m for m in compare_metrics(before, after)}
    assert round(metrics["Skills recall"]["delta"], 2) == 0.22 and round(metrics["Overall F1"]["delta"], 2) == 0.06

    groups = compare_groups(before, after, "input_type")
    assert groups[0]["input_type"] == "scanned_pdf" and groups[0]["documents"] == 2

    docs = compare_documents(before, after)
    assert [d["doc_id"] for d in docs] == ["c", "a", "b"]  # largest improvement first
    assert docs[0]["error_before"] and not docs[0]["error_after"]


def test_no_finetuned_run_yet(tmp_path):
    assert default_pair([]) == (None, None)
    assert default_pair(["ocr_llm_Qwen2.5-7B-Instruct_dev"]) == ("ocr_llm_Qwen2.5-7B-Instruct_dev", None)
    assert list_runs(tmp_path / "missing") == []
