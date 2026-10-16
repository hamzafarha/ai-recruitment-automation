"""Step 2 — evaluate CV extraction on the frozen evaluation set (eval_v1).

Usage (from the project root):
  python scripts/eval_extraction.py                         # real pipeline (OCR + Qwen) on dev
  python scripts/eval_extraction.py --engine text_llm       # Qwen on the dataset's clean text (no OCR)
  python scripts/eval_extraction.py --engine vlm            # Qwen2.5-VL reads the page images
  python scripts/eval_extraction.py --limit 5               # quick check on 5 documents
  python scripts/eval_extraction.py --split test            # final numbers (only once the system is tuned)

Outputs in data/evaluation/extraction/<run>/: report.md, metrics.json, results.csv, and the
predictions/ folder (git-ignored: it contains personal data). Interrupted runs resume.
"""

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from recruitment_ai.config.settings import get_settings
from recruitment_ai.evaluation.extraction_run import (
    ENGINES,
    extraction_documents,
    run_extraction,
    write_outputs,
)
from recruitment_ai.evaluation.freeze import verify
from recruitment_ai.utils.logging import configure_logging

EVAL_DIR = Path("data/processed/eval_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=ENGINES, default="ocr_llm")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--sources", default="real,synthetic", help="comma-separated: real, synthetic")
    parser.add_argument("--model", help="Ollama model (default: OLLAMA_MODEL, or OLLAMA_VISION_MODEL for vlm)")
    parser.add_argument("--limit", type=int, help="only the first N documents")
    parser.add_argument("--run-name", help="output folder name (default: <engine>_<model>_<split>)")
    parser.add_argument("--overwrite", action="store_true", help="re-run documents already predicted")
    args = parser.parse_args()

    configure_logging(level="WARNING")
    settings = get_settings()
    changes = verify(EVAL_DIR / "manifest.csv", Path("."))
    if any(changes.values()):
        sys.exit(f"The evaluation set changed since it was frozen: {changes}. Re-freeze or restore it first.")

    from recruitment_ai.extraction.extractor import CVExtractor
    from recruitment_ai.llm.ollama_adapter import OllamaAdapter

    model = args.model or (settings.ollama_vision_model if args.engine == "vlm" else settings.ollama_model)
    llm = OllamaAdapter(settings, model=model)
    extractor = CVExtractor(llm=llm, vision_llm=llm if args.engine == "vlm" else None)

    docs = extraction_documents(EVAL_DIR, args.split, args.engine, tuple(args.sources.split(",")))
    if args.limit:
        docs = docs[: args.limit]
    run_name = args.run_name or f"{args.engine}_{model.replace(':', '-')}_{args.split}"
    out_dir = Path("data/evaluation/extraction") / run_name
    print(f"\n== {run_name}: {len(docs)} documents -> {out_dir}")

    rows = run_extraction(docs, args.engine, out_dir, extractor, split=args.split, overwrite=args.overwrite)
    config = {"run_name": run_name, "engine": args.engine, "model": model, "split": args.split,
              "sources": args.sources, "limit": args.limit, "num_ctx": settings.ollama_num_ctx}
    metrics = write_outputs(rows, out_dir, config)
    overall = metrics["overall"]
    print(f"\n  overall F1 {overall['overall_f1']:.0%} | experience {overall['experience']['f1']:.0%} | "
          f"education {overall['education']['f1']:.0%} | skills {overall['skills']['f1']:.0%} | "
          f"errors {metrics['errors']} | {metrics['mean_seconds_per_document']} s/doc")
    print(f"  -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
