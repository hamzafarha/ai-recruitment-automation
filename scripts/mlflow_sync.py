"""Log every evaluation of data/evaluation/ into MLflow (experiment tracking), to browse and compare them
in the MLflow interface.

    python scripts/mlflow_sync.py                 # logs what changed since the last sync
    python scripts/mlflow_sync.py --force         # logs everything again
    python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000   -> http://localhost:5000

The evaluations run on Kaggle's GPUs (scripts/kaggle_run.py) and are downloaded into data/evaluation/;
this script reads those files (no model, no GPU) and creates one MLflow experiment per part:

- "DSO1 - CV extraction (Qwen2.5 base vs QLoRA)": one run per extraction run (base model, fine-tuned
  adapter, smoke test): F1/precision/recall per field, identity accuracy, hallucinations, seconds per CV,
  the same per document type, source and language;
- "DSO1 - QLoRA fine-tuning": the training run, its loss at every logged step (a curve in MLflow), its
  hyper-parameters, training data report and smoke test;
- "DSO2 - RAG retrieval": dense (BGE-M3 + Qdrant) and dense + BGE reranker: Hit@k, MRR, recall@k per slice;
- "DSO3 - Matching": the semantic matcher (default / calibrated thresholds), the fixed Qwen analysis and
  the screening agent, per split (dev, test, all);
- "DSO4 - Screening agent": its tool and model calls, planning rounds, checks, fallbacks;
- "DSO4 - AI assistant": success, tool choice, answers, grounding, refusals, per category.
Each run keeps its source files as artifacts (metrics.json, report.md, CSVs). A run is identified by the
tag ``eval_key`` and re-logged only when its files changed (tag ``source_hash``), or with --force.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "evaluation"
for path in (ROOT / "src",):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

EXPERIMENTS = {
    "extraction": "DSO1 - CV extraction (Qwen2.5 base vs QLoRA)",
    "finetune": "DSO1 - QLoRA fine-tuning",
    "rag": "DSO2 - RAG retrieval (BGE-M3 + Qdrant + reranker)",
    "matching": "DSO3 - Matching (semantic, Qwen analysis, agent)",
    "screening": "DSO4 - Screening agent",
    "assistant": "DSO4 - AI assistant (LangGraph + tools)",
}
KEY = re.compile(r"[^\w.\- /]")


# --------------------------------------------------------------------------- pure helpers (unit-tested)


def metric_key(*parts: str) -> str:
    """An MLflow-safe metric name: "hit@1" -> "hit_at_1", "language=fr" -> "language_fr"."""
    text = ".".join(p for p in parts if p).replace("@", "_at_").replace("=", "_").replace("+", "_plus_")
    return KEY.sub("_", text).strip("._")[:250]


def flatten(data: Any, prefix: str = "") -> dict[str, float]:
    """Every number of a nested dict as a flat {metric key: value} (booleans and texts are skipped)."""
    out: dict[str, float] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            out.update(flatten(value, metric_key(prefix, str(key))))
    elif isinstance(data, int | float) and not isinstance(data, bool):
        out[prefix] = float(data)
    return out


def files_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in paths if p.is_file()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def planned_runs() -> list[dict[str, Any]]:
    """What will be logged: experiment, run name, params, metrics, step metrics, artifacts, tags."""
    runs: list[dict[str, Any]] = []
    # DSO1 extraction
    for metrics_file in sorted((EVAL / "extraction").glob("*/metrics.json")):
        m = _json(metrics_file) or {}
        folder, config = metrics_file.parent, m.get("config") or {}
        values = flatten(m.get("overall") or {})
        values |= flatten(m.get("by") or {}, "by")
        values |= {k: float(v) for k, v in (("documents", m.get("documents")), ("errors", m.get("errors")),
                                           ("seconds_per_cv", m.get("mean_seconds_per_document")))
                   if isinstance(v, int | float)}
        runs.append({"experiment": "extraction", "key": f"extraction/{folder.name}",
                     "name": ("Qwen2.5-7B + QLoRA adapter" if config.get("adapter") else
                              "Qwen2.5-7B base") + f" - {config.get('split') or folder.name}"
                     if folder.name != "smoke" else "smoke test",
                     "params": {k: config.get(k) for k in ("engine", "model", "adapter", "split", "sources",
                                                           "device", "dtype", "planned_documents")},
                     "metrics": values, "artifacts": [folder / "metrics.json", folder / "report.md",
                                                      folder / "results.csv"],
                     "tags": {"model.variant": "fine-tuned (QLoRA)" if config.get("adapter") else "base",
                              "technologies": "PyMuPDF, python-docx, RapidOCR, Qwen2.5-7B-Instruct, PEFT"}})
    # DSO1 fine-tuning
    folder = EVAL / "finetune"
    state = _json(folder / "training_state.json")
    if state:
        params = (_json(folder / "params.json") or {}) | {f"train.{k}": v for k, v in
                                                          (_json(folder / "train_config.json") or {}).items()
                                                          if k not in ("model_path", "train_file", "output_dir",
                                                                       "adapter_dir", "state_file", "stop_at")}
        data_report = _json(folder / "data_report.json") or {}
        smoke = _json(folder / "smoke_test.json") or {}
        losses = state.get("losses") or []
        values = {k: float(v) for k, v in (("examples", state.get("examples")),
                                           ("skipped_too_long", state.get("skipped_too_long")),
                                           ("steps", state.get("global_step")),
                                           ("minutes", state.get("minutes_this_session")),
                                           ("final_loss", losses[-1]["loss"] if losses else None),
                                           ("first_loss", losses[0]["loss"] if losses else None))
                  if isinstance(v, int | float)}
        values |= flatten({k: v for k, v in data_report.items() if k not in ("train_categories", "params")}, "data")
        values |= flatten({k: v for k, v in smoke.items() if k != "per_cv"}, "smoke")
        runs.append({"experiment": "finetune", "key": "finetune/qlora", "name": "QLoRA - Qwen2.5-7B-Instruct",
                     "params": params, "metrics": values,
                     "steps": {"loss": [(int(x["step"]), float(x["loss"])) for x in losses]},
                     "artifacts": [folder / n for n in ("report.md", "params.json", "train_config.json",
                                                        "data_report.json", "smoke_test.json", "environment.json",
                                                        "training_state.json")],
                     "tags": {"technologies": "QLoRA 4-bit (bitsandbytes), PEFT LoRA, TRL, Kaggle 2x T4",
                              "labels": str(data_report.get("labels") or "")}})
    # DSO2 retrieval
    rag = _json(EVAL / "rag" / "metrics.json")
    for name, slices in (rag or {}).items():
        runs.append({"experiment": "rag", "key": f"rag/{name}", "name": name,
                     "params": {"embedding": "BAAI/bge-m3", "vector_store": "Qdrant (cosine)",
                                "reranker": "BAAI/bge-reranker-v2-m3" if "rerank" in name else "none"},
                     "metrics": flatten(slices),
                     "artifacts": [EVAL / "rag" / "metrics.json", EVAL / "rag" / "per_query.csv",
                                   EVAL / "rag" / "report.md"],
                     "tags": {"technologies": "BGE-M3, Qdrant" + (", BGE reranker" if "rerank" in name else "")}})
    # DSO3 matching, DSO4 screening agent
    matching = _json(EVAL / "matching" / "metrics.json") or {}
    files = [EVAL / "matching" / n for n in ("metrics.json", "pairs.csv", "requirements.csv", "report.md",
                                             "thresholds.json")]
    for name, splits in (matching.get("systems") or {}).items():
        values = {}
        for split, summary in splits.items():
            grades = {k: v for k, v in (summary.get("grades") or {}).items() if k != "confusion"}
            values |= flatten(grades, f"{split}.grades")
            values |= flatten(summary.get("requirements") or {}, f"{split}.requirements")
        runs.append({"experiment": "matching", "key": f"matching/{name}", "name": name,
                     "params": {"pairs": matching.get("pairs"), **{f"model.{k}": v for k, v in
                                                                   (matching.get("models") or {}).items()},
                                **{f"threshold.{k}": v for k, v in (matching.get("thresholds") or {}).items()}},
                     "metrics": values, "artifacts": files, "tags": {"system": name.split(" ")[0]}})
    if matching.get("agent"):
        agent_system = next((s for n, s in (matching.get("systems") or {}).items() if n.startswith("agent")), {})
        values = flatten({k: v for k, v in matching["agent"].items() if k != "checks_by_kind"})
        values |= flatten(matching["agent"].get("checks_by_kind") or {}, "checks")
        values |= flatten({k: v for k, v in ((agent_system.get("test") or {}).get("grades") or {}).items()
                           if k != "confusion"}, "test.grades")
        runs.append({"experiment": "screening", "key": "screening/agent", "name": "LangGraph screening agent",
                     "params": {"llm": (matching.get("models") or {}).get("llm"), "max_rounds": 2,
                                "max_calls_per_round": 4, "evidence_per_requirement": 3},
                     "metrics": values, "artifacts": files,
                     "tags": {"technologies": "LangGraph, Qwen2.5-7B, CV search, HR knowledge search, reranker"}})
    # DSO4 assistant
    assistant = _json(EVAL / "agent" / "metrics.json")
    if assistant:
        summary = assistant.get("summary") or {}
        values = flatten(summary.get("overall") or {}, "overall") | flatten(summary.get("refusals") or {}, "refusals")
        values |= flatten(summary.get("by_category") or {}, "category")
        runs.append({"experiment": "assistant", "key": "assistant/langgraph", "name": "LangGraph assistant",
                     "params": {"model": assistant.get("model"), "scenarios": assistant.get("scenarios")},
                     "metrics": values, "artifacts": [EVAL / "agent" / n for n in ("metrics.json", "report.md",
                                                                                   "runs.jsonl")],
                     "tags": {"technologies": "LangGraph, Qwen2.5-7B, 8 tools"}})
    return runs


# --------------------------------------------------------------------------- MLflow


def log_runs(runs: list[dict[str, Any]], tracking_uri: str, force: bool = False) -> list[str]:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    artifacts_root = (ROOT / "mlruns" / "artifacts").resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    done = []
    for run in runs:
        name = EXPERIMENTS[run["experiment"]]
        experiment = client.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(
            name, artifact_location=(artifacts_root / run["experiment"]).as_uri(),
            tags={"project": "ai-recruitment-automation", "source": "data/evaluation"})
        digest = hashlib.sha256((files_hash(run["artifacts"]) + json.dumps(run["metrics"], sort_keys=True))
                                .encode()).hexdigest()[:16]
        existing = client.search_runs([experiment_id], filter_string=f"tags.eval_key = '{run['key']}'")
        if existing and not force and existing[0].data.tags.get("source_hash") == digest:
            print(f"  = {name} / {run['name']} (unchanged)")
            continue
        for old in existing:
            client.delete_run(old.info.run_id)
        with mlflow.start_run(experiment_id=experiment_id, run_name=run["name"],
                              tags={"eval_key": run["key"], "source_hash": digest, **run.get("tags", {})}):
            mlflow.log_params({k: str(v)[:500] for k, v in run["params"].items() if v is not None})
            mlflow.log_metrics(run["metrics"])
            for key, points in (run.get("steps") or {}).items():
                for step, value in points:
                    mlflow.log_metric(key, value, step=step)
            for path in run["artifacts"]:
                if path.is_file():
                    mlflow.log_artifact(str(path))
        done.append(f"{name} / {run['name']}")
        print(f"  + {name} / {run['name']} ({len(run['metrics'])} metrics)")
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="log every run again")
    parser.add_argument("--tracking-uri", default=None, help="default: MLFLOW_TRACKING_URI or sqlite:///mlruns/mlflow.db")
    parser.add_argument("--dry-run", action="store_true", help="print what would be logged, without MLflow")
    args = parser.parse_args()
    os.chdir(ROOT)  # a relative tracking URI (sqlite:///mlruns/mlflow.db) is always the project's
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    runs = planned_runs()
    if not runs:
        sys.exit("No evaluation results in data/evaluation/: run or fetch them first (scripts/kaggle_run.py)")
    if args.dry_run:
        for run in runs:
            print(f"{EXPERIMENTS[run['experiment']]} / {run['name']}: {len(run['metrics'])} metrics, "
                  f"{len(run['params'])} params, {sum(p.is_file() for p in run['artifacts'])} artifacts")
        return
    try:
        import mlflow  # noqa: F401
    except ImportError:
        sys.exit('MLflow is not installed: pip install -e ".[tracking]"')
    from recruitment_ai.config.settings import get_settings

    uri = args.tracking_uri or get_settings().mlflow_tracking_uri
    if uri.startswith("sqlite:///"):
        Path(uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    print(f"MLflow tracking store: {uri}")
    done = log_runs(runs, uri, force=args.force)
    print(f"\n{len(done)} run(s) logged, {len(runs) - len(done)} unchanged. Open the interface:\n"
          f"  python -m mlflow ui --backend-store-uri {uri} --port 5000   ->  http://localhost:5000")


if __name__ == "__main__":
    main()
