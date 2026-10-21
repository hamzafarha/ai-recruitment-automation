"""Evaluation results of every part of the system (data/evaluation/, runs made on Kaggle's GPUs).

Read-only views for the app's Evaluation page (and the same files are logged into MLflow by
scripts/mlflow_sync.py):
- extraction (DSO1): Qwen2.5-7B base vs Qwen2.5-7B + our QLoRA adapter on the same frozen CVs, per
  field and per document type (text PDF, scanned PDF read by OCR, Word);
- finetune (DSO1): the QLoRA training run (loss curve, hyper-parameters, training data, smoke test);
- rag (DSO2): BGE-M3 dense retrieval in Qdrant vs + the BGE reranker (Hit@k, MRR, recall, per language);
- matching (DSO3) and screening (DSO4): the fast semantic matcher, the fixed Qwen analysis and the
  LangGraph screening agent on 36 graded CV-job pairs;
- assistant (DSO4): the LangGraph assistant on 20 recruiter questions;
- pipeline: live timings and outcomes of every step of the application pipeline (PostgreSQL).
"""

import csv
import importlib.util
import json
from pathlib import Path
from statistics import mean
from typing import Any

import httpx
from sqlalchemy.orm import Session

from recruitment_ai.evaluation.compare import (
    compare_documents,
    compare_groups,
    compare_metrics,
    default_pair,
    list_runs,
    load_run,
)
from recruitment_ai.services.errors import NotFoundError

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "data" / "evaluation"
SECTIONS = ("experience", "education", "skills")

# what each evaluation measures, and with which technologies (Evaluation page, MLflow tags)
COMPONENTS: list[dict[str, Any]] = [
    {"key": "extraction", "dso": "DSO1", "title": "CV extraction — base vs fine-tuned",
     "technologies": ["PyMuPDF", "python-docx", "RapidOCR (PaddleOCR models)", "Qwen2.5-7B-Instruct",
                      "QLoRA adapter (PEFT)", "Hugging Face Transformers"],
     "metrics": "Precision, recall and F1 of the positions, degrees and skills found; exact dates; identity "
                "accuracy and invented identity fields (hallucinations); seconds per CV",
     "data": "frozen evaluation set eval_v1, dev split (71 CVs: real and synthetic, text PDF, scans, Word)"},
    {"key": "finetune", "dso": "DSO1", "title": "QLoRA fine-tuning run",
     "technologies": ["Qwen2.5-7B-Instruct", "QLoRA (4-bit, bitsandbytes)", "PEFT LoRA r=16", "TRL", "Kaggle 2x T4"],
     "metrics": "Training loss per step; quick F1 check on held-out CVs",
     "data": "1000 CVs of the Resume Parser train split (silver labels), evaluation CVs excluded"},
    {"key": "rag", "dso": "DSO2", "title": "Retrieval (RAG) on the HR knowledge base",
     "technologies": ["BAAI/bge-m3 embeddings", "Qdrant (cosine)", "BAAI/bge-reranker-v2-m3"],
     "metrics": "Hit@1/3/5 (expected document in the top k), MRR, recall@k; per language and cross-lingual",
     "data": "40 recruiter questions in English and French over 15 HR documents and 5 job postings"},
    {"key": "matching", "dso": "DSO3", "title": "Candidate-job matching",
     "technologies": ["BGE-M3 + BGE reranker (semantic matcher)", "Qwen2.5-7B + RAG (fixed analysis)",
                      "LangGraph screening agent"],
     "metrics": "Grade accuracy (0-3), within one grade, MAE, quadratic kappa, Spearman; must-have status "
                "accuracy and macro-F1",
     "data": "36 graded CV-job pairs (silver labels from the rubric KB-RR-002), dev for tuning, test final"},
    {"key": "screening", "dso": "DSO4", "title": "Screening agent (application pipeline)",
     "technologies": ["LangGraph", "Qwen2.5-7B-Instruct", "CV search / HR knowledge search / reranking tools"],
     "metrics": "Same grades as matching, plus model and tool calls per pair, planning rounds, claims changed "
                "by the checks, fallbacks",
     "data": "the 36 graded pairs of the matching evaluation"},
    {"key": "assistant", "dso": "DSO4", "title": "AI assistant (LangGraph + 8 tools)",
     "technologies": ["LangGraph", "Qwen2.5-7B-Instruct", "8 tools (PostgreSQL, Qdrant, reranker, matcher)"],
     "metrics": "Task success, right first tool, expected answer, grounded names, refusals of discriminatory "
                "requests; per category",
     "data": "20 recruiter questions on the demo data"},
]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _get(data: Any, *path: str) -> Any:
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


# --------------------------------------------------------------------------- DSO1 extraction


def extraction(before: str | None = None, after: str | None = None) -> dict[str, Any]:
    root = EVAL_DIR / "extraction"
    names = list_runs(root)
    if not names:
        return {"available": False, "runs": []}
    default_before, default_after = default_pair(names)
    before = before or default_before
    after = after or default_after
    for name in (before, after):
        if name and name not in names:
            raise NotFoundError(f"No extraction run {name!r}: use one of {', '.join(names)}")
    runs = []
    for name in names:
        m = load_run(root, name)["metrics"]
        runs.append({"name": name, "config": m.get("config") or {}, "documents": m.get("documents"),
                     "errors": m.get("errors"), "mean_seconds": m.get("mean_seconds_per_document"),
                     "overall_f1": _get(m, "overall", "overall_f1"),
                     "adapter": bool((m.get("config") or {}).get("adapter"))})
    out: dict[str, Any] = {"available": True, "runs": runs, "before": before, "after": after}
    if not (before and after and before != after):
        return out
    b, a = load_run(root, before), load_run(root, after)
    mb, ma = b["metrics"], a["metrics"]
    out.update(
        metrics=compare_metrics(b, a),
        groups={g: compare_groups(b, a, g) for g in ("input_type", "source", "language")},
        documents=compare_documents(b, a),
        sections=[{"section": s, **{f"{run}_{k}": _get(m, "overall", s, k) for run, m in (("before", mb), ("after", ma))
                                    for k in ("precision", "recall", "f1")}} for s in SECTIONS],
        identity={run: _get(m, "overall", "identity") for run, m in (("before", mb), ("after", ma))},
        speed={run: m.get("mean_seconds_per_document") for run, m in (("before", mb), ("after", ma))},
        errors={run: m.get("errors") for run, m in (("before", mb), ("after", ma))},
        config={run: m.get("config") for run, m in (("before", mb), ("after", ma))},
        reports={run: _text(root / name / "report.md") for run, name in (("before", before), ("after", after))},
    )
    return out


def finetune() -> dict[str, Any]:
    folder = EVAL_DIR / "finetune"
    state = _json(folder / "training_state.json") or {}
    smoke = _json(folder / "smoke_test.json") or {}
    return {"available": bool(state or (folder / "report.md").exists()),
            "status": state.get("status"), "global_step": state.get("global_step"), "max_steps": state.get("max_steps"),
            "examples": state.get("examples"), "skipped_too_long": state.get("skipped_too_long"),
            "gpu_processes": state.get("gpu_processes"), "minutes": state.get("minutes_this_session"),
            "losses": state.get("losses") or [], "config": _json(folder / "train_config.json") or state.get("config"),
            "params": _json(folder / "params.json"), "data_report": _json(folder / "data_report.json"),
            "environment": _json(folder / "environment.json"),
            "smoke_test": {k: v for k, v in smoke.items() if k != "per_cv"} if smoke else None,
            "report": _text(folder / "report.md")}


# --------------------------------------------------------------------------- DSO2 retrieval


def rag() -> dict[str, Any]:
    folder = EVAL_DIR / "rag"
    metrics = _json(folder / "metrics.json")
    if not metrics:
        return {"available": False}
    return {"available": True, "configurations": list(metrics), "metrics": metrics,
            "per_query": _csv(folder / "per_query.csv"), "report": _text(folder / "report.md")}


# --------------------------------------------------------------------------- DSO3-4 matching and screening


def matching() -> dict[str, Any]:
    folder = EVAL_DIR / "matching"
    metrics = _json(folder / "metrics.json")
    if not metrics:
        return {"available": False}
    return {"available": True, "systems": metrics.get("systems") or {}, "thresholds": metrics.get("thresholds"),
            "pairs": metrics.get("pairs"), "notes": metrics.get("notes") or [], "agent": metrics.get("agent"),
            "models": metrics.get("models"), "per_pair": _csv(folder / "pairs.csv"),
            "report": _text(folder / "report.md")}


def assistant() -> dict[str, Any]:
    folder = EVAL_DIR / "agent"
    metrics = _json(folder / "metrics.json")
    if not metrics:
        return {"available": False}
    scenarios = []
    try:
        for line in (folder / "runs.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            scenario, result, score = row.get("scenario") or {}, row.get("result") or {}, row.get("score") or {}
            scenarios.append({"id": scenario.get("id"), "category": scenario.get("category"),
                              "question": scenario.get("question"), "expected_tools": scenario.get("expected_tools"),
                              "tools": [s.get("tool") for s in result.get("steps") or []],
                              "answer": (result.get("answer") or "")[:1500], "seconds": result.get("seconds"),
                              "score": {k: v for k, v in score.items() if k not in ("answer", "question")}})
    except (OSError, ValueError):
        pass
    return {"available": True, "model": metrics.get("model"), "summary": metrics.get("summary"),
            "scenarios": scenarios, "report": _text(folder / "report.md")}


# --------------------------------------------------------------------------- live pipeline


def pipeline_stats(db: Session) -> dict[str, Any]:
    """Timings and outcomes of every step over the stored applications (the live system, not a benchmark)."""
    from recruitment_ai.database.models import Application, Email
    from recruitment_ai.services.pipeline import STEPS

    applications = db.query(Application).all()
    rows = []
    for step in STEPS:
        entries = [e for a in applications for e in (a.steps or []) if e.get("step") == step.name]
        seconds = [float(e["seconds"]) for e in entries if e.get("status") == "done" and e.get("seconds") is not None]
        rows.append({"step": step.name, "label": step.label, "part": step.part, "runs": len(entries),
                     "done": sum(e.get("status") == "done" for e in entries),
                     "failed": sum(e.get("status") == "failed" for e in entries),
                     "waiting": sum(e.get("status") == "waiting" for e in entries),
                     "mean_seconds": round(mean(seconds), 2) if seconds else None,
                     "max_seconds": round(max(seconds), 2) if seconds else None})
    statuses: dict[str, int] = {}
    for a in applications:
        statuses[a.status] = statuses.get(a.status, 0) + 1
    emails: dict[str, int] = {}
    for (status,) in db.query(Email.status).all():
        emails[status] = emails.get(status, 0) + 1
    closed = [a for a in applications if a.status == "closed"]
    return {"applications": len(applications), "by_status": statuses, "steps": rows, "emails": emails,
            "decisions": {d: sum(a.decision == d for a in closed) for d in ("interview", "shortlist", "reject")}}


# --------------------------------------------------------------------------- overview and MLflow


def overview() -> dict[str, Any]:
    """The headline number of every evaluation, for the Evaluation page's first view."""
    ex = extraction()
    head: dict[str, Any] = {}
    if ex.get("metrics"):
        overall = next(r for r in ex["metrics"] if r["metric"] == "Overall F1")
        head["extraction"] = {"before": overall["before"], "after": overall["after"], "delta": overall["delta"],
                              "label": "Overall F1 (dev)"}
    ft = finetune()
    if ft.get("losses"):
        head["finetune"] = {"first_loss": ft["losses"][0]["loss"], "last_loss": ft["losses"][-1]["loss"],
                            "steps": ft.get("global_step"), "label": "Training loss"}
    r = rag()
    if r.get("available"):
        head["rag"] = {name: {"hit@1": _get(m, "all", "hit@1"), "mrr": _get(m, "all", "mrr")}
                       for name, m in r["metrics"].items()}
    m = matching()
    if m.get("available"):
        head["matching"] = {name: _get(s, "test", "grades", "accuracy") for name, s in m["systems"].items()}
        head["screening"] = m.get("agent")
    a = assistant()
    if a.get("available"):
        head["assistant"] = _get(a, "summary", "overall")
    return {"components": COMPONENTS, "headline": head, "folder": str(EVAL_DIR)}


def mlflow_status(settings: Any) -> dict[str, Any]:
    """Is MLflow installed, is its interface running, which experiments does it hold."""
    out: dict[str, Any] = {"installed": importlib.util.find_spec("mlflow") is not None,
                           "tracking_uri": settings.mlflow_tracking_uri, "ui_url": settings.mlflow_ui_url,
                           "reachable": False, "experiments": []}
    base = settings.mlflow_ui_url.rstrip("/")
    try:
        response = httpx.post(f"{base}/api/2.0/mlflow/experiments/search", json={"max_results": 100}, timeout=3)
        out["reachable"] = response.status_code < 400
        if out["reachable"]:
            out["experiments"] = [{"id": e.get("experiment_id"), "name": e.get("name"),
                                   "link": f"{base}/#/experiments/{e.get('experiment_id')}"}
                                  for e in response.json().get("experiments") or [] if e.get("name") != "Default"]
    except (httpx.HTTPError, ValueError):
        pass
    return out
