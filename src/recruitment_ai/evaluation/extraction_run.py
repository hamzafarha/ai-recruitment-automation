"""Run CV extraction on the evaluation set and score it (Step 2, DSO1).

Engines:
- ``ocr_llm``  : the real pipeline: document -> text (text layer / OCR / python-docx) -> Qwen
- ``text_llm`` : real CVs only, from the dataset's own clean text (no OCR): Qwen alone
- ``vlm``      : PDFs only, page images read by Qwen2.5-VL (no OCR)

Predictions are saved one file per document (resumable); predictions of real CVs contain
personal data and live in a git-ignored folder. Only scores and the report are versioned.
"""

import csv
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from recruitment_ai.evaluation.eval_set import read_rows, synthetic_split
from recruitment_ai.evaluation.extraction_eval import score_profile, summarize

ENGINES = ("ocr_llm", "text_llm", "vlm")
_SPLIT_DIRS = {"dev": "validation", "test": "test"}


def extraction_documents(
    eval_dir: Path, split: str, engine: str, sources: tuple[str, ...] = ("real", "synthetic"),
    raw_pdf_dir: Path = Path("data/raw/resumes/resume_parser/hf/raw"),
    interim_dir: Path = Path("data/interim/resume_parser"),
) -> list[dict[str, Any]]:
    """Every document of the split that ``engine`` can process, with its answer key."""
    docs: list[dict[str, Any]] = []
    if "real" in sources:
        for row in read_rows(eval_dir / "cv_pool.csv"):
            if row["eval_split"] != split:
                continue
            gold = json.loads((interim_dir / _SPLIT_DIRS[split] / f"{row['resume_id']}.json").read_text(encoding="utf-8"))
            docs.append({
                "doc_id": row["resume_id"], "cv_id": row["resume_id"], "source": "real", "language": "en",
                "input_type": "text" if engine == "text_llm" else "scanned_pdf",
                "path": raw_pdf_dir / row["category"] / f"{row['resume_id']}.pdf",
                "gold": gold, "score_identity": row["template_identity"] != "True",
            })
    if "synthetic" in sources and engine != "text_llm":
        for path in sorted((eval_dir / "synthetic" / "profiles").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            meta = data["meta"]
            if synthetic_split(meta["id"]) != split:
                continue
            for fmt in meta["formats"]:
                if engine == "vlm" and fmt != "pdf":
                    continue
                docs.append({
                    "doc_id": f"{meta['id']}_{fmt}", "cv_id": meta["id"], "source": "synthetic",
                    "language": meta["language"], "input_type": "text_pdf" if fmt == "pdf" else "docx",
                    "path": eval_dir / "synthetic" / "files" / f"{meta['id']}.{fmt}",
                    "gold": data["profile"], "score_identity": True,
                })
    return docs


def _dataset_texts(curated_dir: Path, split: str, ids: set[str]) -> dict[str, str]:
    from recruitment_ai.data.resume_parser import iter_split

    return {r["resume_id"]: r["resume_text"] for r in iter_split(curated_dir / f"{_SPLIT_DIRS[split]}.jsonl")
            if r["resume_id"] in ids}


def run_extraction(
    docs: list[dict[str, Any]], engine: str, out_dir: Path, extractor: Any,
    curated_dir: Path = Path("data/raw/resumes/resume_parser/hf/curated"),
    split: str = "dev", overwrite: bool = False, log: Any = print,
) -> list[dict[str, Any]]:
    """Predict (or reuse) and score every document; returns one result row per document."""
    from recruitment_ai.extraction.extractor import pdf_page_images

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    texts = _dataset_texts(curated_dir, split, {d["cv_id"] for d in docs}) if engine == "text_llm" else {}
    rows = []
    for i, doc in enumerate(docs, 1):
        pred_path = pred_dir / f"{doc['doc_id']}.json"
        if pred_path.exists() and not overwrite:
            record = json.loads(pred_path.read_text(encoding="utf-8"))
        else:
            start = time.perf_counter()
            try:
                if engine == "ocr_llm":
                    profile, stats = extractor.extract_document(doc["path"], doc["cv_id"])
                elif engine == "text_llm":
                    profile, stats = extractor.extract_text(texts[doc["cv_id"]], doc["cv_id"])
                else:
                    profile, stats = extractor.extract_images(pdf_page_images(doc["path"]), doc["cv_id"])
                record = {"prediction": profile.model_dump(exclude={"created_at"}), "stats": stats, "error": None}
            except Exception as exc:  # one failure must not stop a long run; it is scored as an empty prediction
                record = {"prediction": {}, "stats": {}, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc(limit=3)}
            record["total_seconds"] = round(time.perf_counter() - start, 2)
            pred_path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        scores = score_profile(doc["gold"], record["prediction"], extractor.normalizer, doc["score_identity"])
        rows.append({"doc": doc, "record": record, "scores": scores})
        status = "ERROR " + record["error"][:60] if record["error"] else f"{record.get('total_seconds', 0):.0f}s"
        log(f"  [{i}/{len(docs)}] {doc['doc_id']} ({doc['input_type']}): {status}")
    return rows


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def write_outputs(rows: list[dict[str, Any]], out_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """results.csv (per document), metrics.json (overall + by group) and report.md."""
    groups: dict[str, dict[str, list]] = {"source": defaultdict(list), "input_type": defaultdict(list),
                                          "language": defaultdict(list)}
    for r in rows:
        for g in groups:
            groups[g][r["doc"][g]].append(r["scores"])
    errors = [r["doc"]["doc_id"] for r in rows if r["record"]["error"]]
    seconds = [r["record"].get("total_seconds", 0) for r in rows if not r["record"]["error"]]
    metrics = {
        "config": config,
        "documents": len(rows), "errors": len(errors), "error_docs": errors,
        "mean_seconds_per_document": round(sum(seconds) / len(seconds), 1) if seconds else None,
        "overall": summarize([r["scores"] for r in rows]),
        "by": {g: {k: summarize(v) for k, v in sorted(groups[g].items())} for g in groups},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "source", "input_type", "language", "error", "seconds",
                         "exp_f1", "edu_f1", "skills_f1", "identity_accuracy"])
        for r in rows:
            s = summarize([r["scores"]])
            writer.writerow([r["doc"]["doc_id"], r["doc"]["source"], r["doc"]["input_type"], r["doc"]["language"],
                             bool(r["record"]["error"]), r["record"].get("total_seconds", ""),
                             s["experience"]["f1"], s["education"]["f1"], s["skills"]["f1"], s["identity"]["accuracy"]])

    def line(name: str, m: dict[str, Any]) -> str:
        return (f"| {name} | {m['cvs']} | {_fmt(m['experience']['f1'])} | {_fmt(m['experience']['date_exact'])} | "
                f"{_fmt(m['education']['f1'])} | {_fmt(m['skills']['precision'])} | {_fmt(m['skills']['recall'])} | "
                f"{_fmt(m['skills']['f1'])} | {_fmt(m['identity']['accuracy'])} | {m['identity']['hallucinations']} |")

    header = ("| Group | Docs | Experience F1 | Dates exact | Education F1 | Skills P | Skills R | Skills F1 | "
              "Identity acc. | Identity halluc. |\n|---|---|---|---|---|---|---|---|---|---|")
    report = [
        f"# Extraction evaluation — {config['run_name']}", "",
        f"Engine `{config['engine']}` · model `{config['model']}` · split `{config['split']}` · "
        f"{len(rows)} documents · {len(errors)} errors · {metrics['mean_seconds_per_document']} s per document", "",
        header, line("**All**", metrics["overall"]),
    ]
    for g in groups:
        for k, m in metrics["by"][g].items():
            report.append(line(f"{g} = {k}", m))
    report += ["", f"Overall F1 (mean of experience, education, skills): {_fmt(metrics['overall']['overall_f1'])}"]
    if errors:
        report += ["", "Errors: " + ", ".join(errors)]
    (out_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics
