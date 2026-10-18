"""DSO4 — evaluate the AI assistant (LangGraph agent + tools) on scripted recruiter questions.

Runs on Kaggle's GPUs: python scripts/kaggle_run.py all agent (never on our PCs).

The job builds a fresh workspace: SQLite database, embedded Qdrant, the HR knowledge base, the
5 demo jobs and the 12 synthetic candidates (the same "Load demo data" as the app), then asks
every question of data/processed/agent_eval/scenarios.json with the real models.

Usage (from the project root, on a GPU machine; the Kaggle job sets the environment):
  DATABASE_URL=sqlite:////tmp/agent.db QDRANT_LOCATION=:memory: LLM_PROVIDER=huggingface \
  python scripts/eval_agent.py [--limit 5]

Outputs in data/evaluation/agent/: report.md, metrics.json, runs.jsonl (questions, tool traces, answers).
"""

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from recruitment_ai.config.settings import get_settings
from recruitment_ai.evaluation.agent_eval import (
    load_scenarios,
    report_markdown,
    score_scenario,
    summarize,
)
from recruitment_ai.utils.logging import configure_logging

SCENARIOS = Path("data/processed/agent_eval/scenarios.json")
OUT = Path("data/evaluation/agent")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only the first N scenarios")
    args = parser.parse_args()
    configure_logging(level="WARNING")
    settings = get_settings()
    if not settings.database_url.startswith("sqlite"):
        sys.exit("Use a throw-away SQLite database (DATABASE_URL=sqlite:////tmp/agent.db): the evaluation loads demo "
                 "data and stores its runs.")

    from recruitment_ai.database.connection import Base, get_engine, get_session_factory
    from recruitment_ai.database.repositories import CandidateRepository
    from recruitment_ai.services import assistant, demo
    from recruitment_ai.services.container import Container

    Base.metadata.create_all(get_engine())
    db = get_session_factory()()
    container = Container(settings)
    start = time.perf_counter()
    setup = demo.setup(db, container)
    print(f"  workspace ready in {time.perf_counter() - start:.0f}s: {setup}", flush=True)
    names = [c.full_name for c in CandidateRepository(db).list(limit=1000) if c.full_name]

    scenarios = load_scenarios(SCENARIOS)
    scenarios = scenarios[: args.limit] if args.limit else scenarios
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    with (OUT / "runs.jsonl").open("w", encoding="utf-8") as runs:
        for i, scenario in enumerate(scenarios, 1):
            try:
                result = assistant.run_dict(assistant.ask(db, container, scenario["question"]))
            except Exception as exc:  # one failure must not stop the evaluation
                db.rollback()
                result = {"answer": "", "steps": [], "warnings": [f"{type(exc).__name__}: {exc}"], "seconds": None,
                          "llm_calls": 0}
            row = score_scenario(scenario, result, names)
            rows.append(row)
            runs.write(json.dumps({"scenario": scenario, "result": result, "score": row}, ensure_ascii=False,
                                  default=str) + "\n")
            print(f"  [{i}/{len(scenarios)}] {scenario['id']} {'OK ' if row['success'] else 'NO '} "
                  f"tools={','.join(row['tools']) or '-'} ({row['seconds']}s)", flush=True)

    summary = summarize(rows)
    model = getattr(container.llm, "model", None)
    (OUT / "metrics.json").write_text(json.dumps({"summary": summary, "model": model, "scenarios": len(rows)},
                                                 indent=1), encoding="utf-8")
    (OUT / "report.md").write_text(report_markdown(summary, rows, model), encoding="utf-8")
    o = summary["overall"]
    print(f"  success {o['success']:.0%} | tool choice {o['tool_choice']:.0%} | answers {o['answer']:.0%} | "
          f"{o['mean_seconds']}s per question\n  -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
