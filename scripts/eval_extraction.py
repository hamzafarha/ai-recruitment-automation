"""Step 2 — evaluate CV extraction on the frozen evaluation set (eval_v1).

Usage (from the project root):
  python scripts/eval_extraction.py                         # real pipeline (OCR + Qwen) on dev
  python scripts/eval_extraction.py --engine text_llm       # Qwen on the dataset's clean text (no OCR)
  python scripts/eval_extraction.py --engine vlm            # Qwen2.5-VL reads the page images
  python scripts/eval_extraction.py --limit 5               # quick check on 5 documents
  python scripts/eval_extraction.py --max-minutes 240       # stop starting documents after 4 h
  python scripts/eval_extraction.py --split test            # final numbers (only once the system is tuned)

Outputs in data/evaluation/extraction/<run>/: report.md, metrics.json, results.csv, and the
predictions/ folder (git-ignored: it contains personal data). Interrupted runs resume.
Models run with Hugging Face Transformers (downloaded on first use); see LLM_DEVICE / LLM_DTYPE.
"""

import argparse
import sys
import time
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
    parser.add_argument("--model", help="Hugging Face model id (default: HUGGINGFACE_MODEL, or MULTIMODAL_MODEL "
                                        "for vlm)")
    parser.add_argument("--adapter", help="LoRA adapter folder from the fine-tuning (Step 9); default: LLM_ADAPTER")
    parser.add_argument("--limit", type=int, help="only the first N documents")
    parser.add_argument("--max-minutes", type=float, help="time budget: no new document after this many minutes")
    parser.add_argument("--run-name", help="output folder name (default: <engine>_<model>_<split>)")
    parser.add_argument("--overwrite", action="store_true", help="re-run documents already predicted")
    args = parser.parse_args()

    configure_logging(level="WARNING")
    settings = get_settings()
    changes = verify(EVAL_DIR / "manifest.csv", Path("."))
    if any(changes.values()):
        sys.exit(f"The evaluation set changed since it was frozen: {changes}. Re-freeze or restore it first.")

    from recruitment_ai.extraction.extractor import CVExtractor
    from recruitment_ai.llm.huggingface_adapter import HuggingFaceAdapter

    deadline = time.monotonic() + args.max_minutes * 60 if args.max_minutes else None
    model = args.model or (settings.multimodal_model if args.engine == "vlm" else settings.huggingface_model)
    llm = HuggingFaceAdapter(settings, model=model, vision=args.engine == "vlm", adapter=args.adapter)
    extractor = CVExtractor(llm=llm, vision_llm=llm if args.engine == "vlm" else None)

    docs = extraction_documents(EVAL_DIR, args.split, args.engine, tuple(args.sources.split(",")))
    if args.limit:
        docs = docs[: args.limit]
    planned = len(docs)
    model_name = model.split("/")[-1] + ("-finetuned" if llm.adapter else "")
    run_name = args.run_name or f"{args.engine}_{model_name}_{args.split}"
    out_dir = Path("data/evaluation/extraction") / run_name
    print(f"\n== {run_name}: {len(docs)} documents -> {out_dir}")

    rows = run_extraction(docs, args.engine, out_dir, extractor, split=args.split, overwrite=args.overwrite,
                          deadline=deadline)
    config = {"run_name": run_name, "engine": args.engine, "model": model, "adapter": llm.adapter, "split": args.split,
              "sources": args.sources, "limit": args.limit, "max_minutes": args.max_minutes,
              "planned_documents": planned, "device": llm.device, "dtype": settings.llm_dtype}
    metrics = write_outputs(rows, out_dir, config)
    overall = metrics["overall"]
    print(f"\n  overall F1 {overall['overall_f1']:.0%} | experience {overall['experience']['f1']:.0%} | "
          f"education {overall['education']['f1']:.0%} | skills {overall['skills']['f1']:.0%} | "
          f"errors {metrics['errors']} | {metrics['mean_seconds_per_document']} s/doc")
    print(f"  -> {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
