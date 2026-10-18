"""DSO3 / DSO4 — evaluate candidate-job matching on the 36 graded pairs of eval_v1.

Runs on Kaggle's GPUs: python scripts/kaggle_run.py all matching (semantic + fixed LLM analysis) or
all screening (the same + the application pipeline's screening agent), never on our PCs.

Usage (from the project root, on a GPU machine):
  python scripts/eval_matching.py                       # semantic (calibrated on dev) + full LLM analysis
  python scripts/eval_matching.py --methods semantic,llm,agent   # + the screening agent (DSO4)
  python scripts/eval_matching.py --methods semantic    # the fast part only (a few minutes)
  python scripts/eval_matching.py --limit 6             # quick check on 6 pairs
  python scripts/eval_matching.py --max-minutes 150     # no new LLM analysis after this time (resumable)

Saved analyses (predictions/<method>/<pair>.json) are reused unless --overwrite: the "screening" job
ships the fixed analysis' saved predictions, so only the agent runs on the GPU. The agent is given
the pair's job (the task "the job applied to"); its checks use the thresholds calibrated on dev.

Needs the real CVs' text (data/raw/resumes/resume_parser/hf/curated/*.jsonl, downloaded by the Kaggle
job from the public dataset), Qdrant (QDRANT_LOCATION=:memory: = embedded) and the models.
Outputs in data/evaluation/matching/: report.md, metrics.json, pairs.csv, requirements.csv,
thresholds.json (used by the app's fast matching) and predictions/ (git-ignored: CV excerpts).
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from recruitment_ai.config.settings import get_settings
from recruitment_ai.evaluation.freeze import verify
from recruitment_ai.evaluation.matching_eval import (
    DEMONSTRATED_GRID,
    EQUIVALENT_GRID,
    agent_summary,
    load_candidates,
    load_jobs,
    load_pairs,
    objective,
    pair_rows,
    report_markdown,
    split_of,
    summarize,
)
from recruitment_ai.utils.logging import configure_logging

EVAL_DIR = Path("data/processed/eval_v1")
INTERIM = Path("data/interim/resume_parser")
CURATED = Path("data/raw/resumes/resume_parser/hf/curated")
KNOWLEDGE = Path("data/knowledge_base")
OUT = Path("data/evaluation/matching")


def job_payload(job: dict[str, Any]) -> dict[str, Any]:
    return {"id": job["id"], "title": job["title"], "company": (job.get("source") or {}).get("company"),
            "role": job["role"], "description": job.get("description") or "", "requirements": job["requirements"]}


def write_csvs(results: dict[str, dict[str, Any]]) -> None:
    with (OUT / "pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "pair_id", "split", "job_role", "cv_id", "cv_source", "reference_grade",
                         "predicted_grade", "score", "warnings"])
        for name, result in results.items():
            for r in result["rows"]:
                writer.writerow([name, r["pair_id"], r["split"], r["job_role"], r["cv_id"], r["cv_source"],
                                 r["gold_grade"], r["pred_grade"], r["score"], " | ".join(r["warnings"])])
    with (OUT / "requirements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "pair_id", "split", "requirement", "reference_status", "predicted_status",
                         "evidence_level"])
        for name, result in results.items():
            for r in result["rows"]:
                for q in r["requirements"]:
                    writer.writerow([name, r["pair_id"], r["split"], q["id"], q["gold"], q["pred"], q["level"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", default="semantic,llm", help="comma-separated: semantic, llm, agent")
    parser.add_argument("--model", help="Hugging Face model id of the LLM (default: HUGGINGFACE_MODEL)")
    parser.add_argument("--limit", type=int, help="only the first N pairs")
    parser.add_argument("--max-minutes", type=float, help="time budget for the LLM analyses (then stop, resumable)")
    parser.add_argument("--overwrite", action="store_true", help="re-run LLM analyses already saved")
    args = parser.parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if unknown := [m for m in methods if m not in ("semantic", "llm", "agent")]:
        parser.error(f"unknown method(s): {', '.join(unknown)}")

    configure_logging(level="WARNING")
    settings = get_settings()
    # real CVs are read from their dataset text and answer key, never from their PDFs (not on Kaggle)
    changes = verify(EVAL_DIR / "manifest.csv", Path("."), skip_kinds={"real_cv_pdf"})
    if any(changes.values()):
        sys.exit(f"The evaluation set changed since it was frozen: {changes}. Restore it first.")
    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None

    from recruitment_ai.data.resume_parser import iter_split
    from recruitment_ai.embeddings.service import SentenceTransformerEmbeddingService
    from recruitment_ai.extraction.extractor import default_normalizer
    from recruitment_ai.ingestion.parser import UnifiedDocumentParser
    from recruitment_ai.matching.semantic import (
        SemanticMatcher,
        Thresholds,
        load_thresholds,
    )
    from recruitment_ai.reranking.reranker import OptionalReranker

    pairs = load_pairs(EVAL_DIR)
    pairs = pairs[: args.limit] if args.limit else pairs
    jobs = {role: job_payload(job) for role, job in load_jobs(EVAL_DIR).items()}
    needed = {p["cv_id"] for p in pairs if p["cv_source"] == "real"}
    texts: dict[str, str] = {}
    for split_dir in ("validation", "test"):
        path = CURATED / f"{split_dir}.jsonl"
        if path.exists():
            texts.update({r["resume_id"]: r["resume_text"] for r in iter_split(path) if r["resume_id"] in needed})
    notes = []
    if missing := needed - set(texts):
        notes.append(f"{len(missing)} real CVs without their dataset text: their answer key was used as text")
    document_parser = UnifiedDocumentParser()
    candidates = load_candidates(pairs, EVAL_DIR, INTERIM, texts, lambda p: document_parser.parse(p).raw_text)
    print(f"  {len(pairs)} pairs ({sum(split_of(p) == 'dev' for p in pairs)} dev), {len(candidates)} CVs, "
          f"{len(jobs)} jobs", flush=True)

    embedder = SentenceTransformerEmbeddingService(settings)
    embedder.require_model()
    reranker = OptionalReranker(settings)
    if reranker.score_pairs([["Python", "Python developer"]]) is None:
        notes.append("BGE reranker unavailable: the semantic matching used BGE-M3 cosine similarity")
    # the dev-calibrated thresholds when they exist (calibrated again below when "semantic" runs)
    matcher = SemanticMatcher(embedder, reranker, default_normalizer(), load_thresholds())
    results: dict[str, dict[str, Any]] = {}
    calibrated: dict[str, Any] | None = None
    OUT.mkdir(parents=True, exist_ok=True)

    if "semantic" in methods:
        start = time.perf_counter()
        evidence: dict[str, Any] = {}
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in pairs:
            by_role[pair["job_role"]].append(pair)
        for role, role_pairs in by_role.items():  # one embedding + one reranking request per job
            found = matcher.score_requirements(jobs[role], [candidates[p["cv_id"]] for p in role_pairs])
            evidence.update({p["pair_id"]: e for p, e in zip(role_pairs, found, strict=True)})
        seconds = (time.perf_counter() - start) / max(len(pairs), 1)
        reranked = all(e["reranked"] for e in evidence.values())

        def reports(chosen: list[dict[str, Any]], t: Thresholds) -> dict[str, dict[str, Any]]:
            return {p["pair_id"]: matcher.report(matcher.assess(evidence[p["pair_id"]], t), jobs[p["job_role"]],
                                                 candidates[p["cv_id"]], seconds).model_dump(mode="json")
                    for p in chosen}

        dev = [p for p in pairs if split_of(p) == "dev"]
        best: tuple[float, Thresholds] | None = None
        for high in DEMONSTRATED_GRID:
            for low in EQUIVALENT_GRID:
                if low >= high:
                    continue
                t = (Thresholds(demonstrated=high, equivalent=low) if reranked
                     else Thresholds(cosine_demonstrated=round(0.3 + high * 0.6, 3),
                                     cosine_equivalent=round(0.3 + low * 0.6, 3)))
                score = objective(summarize(pair_rows(dev, reports(dev, t)))["dev"]) if dev else 0.0
                if best is None or score > best[0] + 1e-9:
                    best = (score, t)
        assert best is not None
        thresholds = best[1]
        calibrated = {**thresholds.to_dict(), "calibrated_on": f"{len(dev)} dev pairs",
                      "objective": round(best[0], 4), "reranked": reranked}
        (OUT / "thresholds.json").write_text(json.dumps(calibrated, indent=1), encoding="utf-8")
        matcher.thresholds = thresholds  # also the fallback of the LLM analysis below
        for name, t in (("semantic (default thresholds)", Thresholds()), ("semantic (calibrated on dev)", thresholds)):
            produced = reports(pairs, t)
            rows = pair_rows(pairs, produced)
            results[name] = {"summary": summarize(rows), "rows": rows, "seconds": seconds}
        pred_dir = OUT / "predictions" / "semantic"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for pair_id, report in reports(pairs, thresholds).items():
            (pred_dir / f"{pair_id}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                      encoding="utf-8")
        print(f"  semantic: {seconds:.2f} s per pair; thresholds {thresholds}", flush=True)

    analyzer = llm = None
    if "llm" in methods or "agent" in methods:
        from recruitment_ai.llm.huggingface_adapter import HuggingFaceAdapter
        from recruitment_ai.matching.analyzer import EvidenceAnalyzer
        from recruitment_ai.rag.search_index import SearchIndex
        from recruitment_ai.vectorstore.qdrant import QdrantVectorStore

        index = SearchIndex(QdrantVectorStore(settings), embedder, reranker, settings.embedding_dimension)
        print(f"  knowledge index: {index.build_knowledge(KNOWLEDGE)} chunks", flush=True)
        # the model is loaded at its first call: not at all when every analysis is already saved
        llm = HuggingFaceAdapter(settings, model=args.model or settings.huggingface_model)  # base model, no adapter
        analyzer = EvidenceAnalyzer(llm, knowledge_search=lambda q, role: index.search_knowledge(q, role=role, top_k=3),
                                    semantic=matcher)

    if "llm" in methods:
        pred_dir = OUT / "predictions" / "llm"
        pred_dir.mkdir(parents=True, exist_ok=True)
        produced, durations = {}, []
        for i, pair in enumerate(pairs, 1):
            path = pred_dir / f"{pair['pair_id']}.json"
            if path.exists() and not args.overwrite:
                produced[pair["pair_id"]] = json.loads(path.read_text(encoding="utf-8"))
                continue
            if deadline is not None and time.monotonic() > deadline:
                notes.append(f"time budget reached: {len(pairs) - i + 1} pairs without an LLM analysis")
                break
            start = time.perf_counter()
            report = analyzer.analyze(jobs[pair["job_role"]], candidates[pair["cv_id"]]).model_dump(mode="json")
            durations.append(time.perf_counter() - start)
            path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
            produced[pair["pair_id"]] = report
            print(f"  [{i}/{len(pairs)}] {pair['pair_id']} {pair['job_role']} {pair['cv_id']}: grade {report['grade']} "
                  f"(reference {pair['grade']}) in {durations[-1]:.0f}s"
                  + (" — " + report["warnings"][0][:80] if report["warnings"] else ""), flush=True)
        failures = sum(any("AI analysis failed" in w for w in r.get("warnings") or []) for r in produced.values())
        if failures:
            notes.append(f"{failures} LLM analyses failed and fell back to the semantic estimate")
        rows = pair_rows(pairs, produced)
        results["llm (Qwen2.5-7B + RAG)"] = {"summary": summarize(rows), "rows": rows,
                                            "seconds": sum(durations) / len(durations) if durations else None}

    agent_stats = None
    if "agent" in methods:
        from recruitment_ai.agents.screening import ScreeningAgent

        pred_dir = OUT / "predictions" / "agent"
        pred_dir.mkdir(parents=True, exist_ok=True)
        produced, durations = {}, []
        for i, pair in enumerate(pairs, 1):
            path = pred_dir / f"{pair['pair_id']}.json"
            if path.exists() and not args.overwrite:
                produced[pair["pair_id"]] = json.loads(path.read_text(encoding="utf-8"))
                continue
            if deadline is not None and time.monotonic() > deadline:
                notes.append(f"time budget reached: {len(pairs) - i + 1} pairs without a screening agent analysis")
                break
            job = jobs[pair["job_role"]]
            start = time.perf_counter()
            run = ScreeningAgent(llm, embedder, reranker, matcher, analyzer).run(candidates[pair["cv_id"]], [job],
                                                                                 job["id"])
            durations.append(time.perf_counter() - start)
            report = run["reports"][0].model_dump(mode="json")
            report["agent_run"] = {"llm_calls": run["llm_calls"], "tool_calls": len(run["tool_calls"]),
                                   "seconds": run["seconds"], "passages": run["passages"],
                                   "passages_from": run["passages_from"]}
            path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
            produced[pair["pair_id"]] = report
            print(f"  [{i}/{len(pairs)}] agent {pair['pair_id']}: grade {report['grade']} (reference {pair['grade']}), "
                  f"{run['llm_calls']} Qwen calls, {len(run['tool_calls'])} tool calls in {durations[-1]:.0f}s",
                  flush=True)
        failures = sum(any("AI analysis failed" in w for w in r.get("warnings") or []) for r in produced.values())
        if failures:
            notes.append(f"{failures} screening agent analyses failed and fell back to the semantic estimate")
        rows = pair_rows(pairs, produced)
        results["agent (LangGraph screening)"] = {"summary": summarize(rows), "rows": rows,
                                                  "seconds": sum(durations) / len(durations) if durations else None}
        agent_stats = agent_summary(list(produced.values()))

    metrics = {"systems": {name: r["summary"] for name, r in results.items()}, "thresholds": calibrated,
               "pairs": len(pairs), "notes": notes, "agent": agent_stats,
               "models": {"embedding": settings.embedding_model, "reranker": settings.reranking_model,
                          "llm": args.model or settings.huggingface_model}}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (OUT / "report.md").write_text(report_markdown(results, calibrated, pairs, notes, agent_stats),
                                   encoding="utf-8")
    write_csvs(results)
    for name, result in results.items():
        s = result["summary"]["test"]["grades"]
        if s.get("pairs"):
            print(f"  {name:32s} test: accuracy {s['accuracy']:.0%} | within one {s['within_one']:.0%} | "
                  f"QWK {s['qwk']}", flush=True)
    print(f"  -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
