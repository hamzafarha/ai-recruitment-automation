"""Step 1c — build the evaluation set in data/processed/eval_v1/.

Stages (run in order; your review happens between them):
  python scripts/build_eval_set.py pool         # select CVs, download their PDFs -> cv_pool.csv
  python scripts/build_eval_set.py pool --no-download   # same, without fetching PDFs
  python scripts/build_eval_set.py shortlist    # 3 candidate job postings per role -> jd_shortlist.md
  python scripts/build_eval_set.py synthetic    # render synthetic CV profiles -> synthetic/files/ (PDF, DOCX)
  python scripts/build_eval_set.py pairs        # pairs to grade + CVs to check -> labels/pairs.csv, extraction_sample.csv
  (labels: model-made silver labels; the Streamlit labeling app was removed with the Streamlit interface)
  python scripts/build_eval_set.py autocheck    # automatic extraction checks -> labels/extraction_checks.csv
  python scripts/build_eval_set.py freeze       # check labels, checksums -> manifest.csv, labels/summary.json, README.md
  python scripts/build_eval_set.py verify       # has anything changed since the freeze?

Requires: python scripts/prepare_data.py (reads data/interim/).
The pool is fully rule-based (no manual review): re-running it gives the same result.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from datetime import UTC

from recruitment_ai.evaluation.eval_set import (
    EXTRACTION_COLUMNS,
    PAIR_COLUMNS,
    build_cv_pool,
    download_pdfs,
    load_jobs,
    read_rows,
    render_synthetic,
    sample_extraction_checks,
    select_pairs,
    shortlist_jobs,
    write_pool,
    write_rows,
    write_shortlist,
)
from recruitment_ai.evaluation.freeze import (
    collect_files,
    completeness,
    datasheet,
    label_summary,
    verify,
    write_manifest,
)
from recruitment_ai.utils.logging import configure_logging

EVAL_DIR = Path("data/processed/eval_v1")
INTERIM = Path("data/interim")
CURATED = Path("data/raw/resumes/resume_parser/hf/curated")
RESUME_HF = Path("data/raw/resumes/resume_parser/hf")


def run_pool(download: bool) -> None:
    pool, stats = build_cv_pool(INTERIM / "resume_parser" / "index.csv", CURATED)
    print(f"  selection: {stats}")
    if download:
        print(f"  downloading {len(pool)} PDFs (cached ones are skipped)...")
        print(f"  pdfs: {download_pdfs(pool, RESUME_HF)}")
    write_pool(pool, EVAL_DIR / "cv_pool.csv")

    for split in ("dev", "test"):
        rows = [r for r in pool if r["eval_split"] == split]
        print(f"  {split:4s}: {len(rows)} CVs | categories {dict(Counter(r['category'] for r in rows))}")
    print(f"  -> {EVAL_DIR / 'cv_pool.csv'}")


def run_shortlist() -> None:
    shortlist = shortlist_jobs(load_jobs(INTERIM / "tech_jobs" / "jobs_clean.json"))
    path = INTERIM / "tech_jobs" / "jd_shortlist.md"
    write_shortlist(shortlist, path)
    for role, jobs in shortlist.items():
        print(f"  {role:24s} " + " | ".join(f"[{j['id']}] {j['title'][:45]}" for j in jobs))
    print(f"  -> {path}")


def run_synthetic() -> None:
    rows = render_synthetic(EVAL_DIR / "synthetic")
    print(f"  rendered {len(rows)} files from {len({r['cv_id'] for r in rows})} profiles: "
          f"{dict(Counter(r['format'] for r in rows))} | languages {dict(Counter(r['language'] for r in rows))}")
    print(f"  -> {EVAL_DIR / 'synthetic' / 'files'}  (index: synthetic/index.csv)")


def run_pairs(force: bool) -> None:
    labels = EVAL_DIR / "labels"
    if not force and (read_rows(labels / "matches.csv") or read_rows(labels / "extraction_checks.csv")):
        print("  labels already exist: pairs.csv is kept so they stay valid (use --force to rebuild)")
        return
    pool = read_rows(EVAL_DIR / "cv_pool.csv")
    pairs = select_pairs(pool)
    write_rows(pairs, labels / "pairs.csv", PAIR_COLUMNS)
    checks = sample_extraction_checks(pool)
    write_rows(checks, labels / "extraction_sample.csv", EXTRACTION_COLUMNS)

    print(f"  {len(pairs)} pairs | per job {dict(Counter(p['job_role'] for p in pairs))}")
    print(f"  split {dict(Counter(p['eval_split'] for p in pairs))} | source {dict(Counter(p['cv_source'] for p in pairs))}")
    print(f"  {len(checks)} CVs for the extraction check | {dict(Counter(c['category'] for c in checks))}")
    print(f"  -> {labels / 'pairs.csv'}, {labels / 'extraction_sample.csv'}")


def run_autocheck() -> None:
    from recruitment_ai.data.resume_parser import iter_split
    from recruitment_ai.evaluation.labeling import (
        CHECK_COLUMNS,
        LabelingData,
        auto_check,
        upsert,
    )

    data = LabelingData()
    sample = data.extraction_sample()
    wanted = {s["resume_id"] for s in sample}
    texts = {r["resume_id"]: r["resume_text"] for r in iter_split(CURATED / "test.jsonl") if r["resume_id"] in wanted}
    results = Counter()
    for s in sample:
        row = auto_check(data.answer_key(s["resume_id"], "real", s["eval_split"]), texts[s["resume_id"]], s["resume_id"])
        upsert(data.checks_path, row, ("resume_id", "annotator"), CHECK_COLUMNS)
        results.update(f"{k}:{row[k]}" for k in ("identity", "experience", "education", "skills"))
    print(f"  {len(sample)} CVs checked automatically -> {data.checks_path}")
    for section in ("identity", "experience", "education", "skills"):
        print(f"  {section:10s} " + ", ".join(f"{k.split(':')[1]} {v}" for k, v in sorted(results.items()) if k.startswith(section)))


def run_freeze(allow_incomplete: bool) -> None:
    import json
    from datetime import datetime

    labels = EVAL_DIR / "labels"
    pairs, matches = read_rows(labels / "pairs.csv"), read_rows(labels / "matches.csv")
    sample, checks = read_rows(labels / "extraction_sample.csv"), read_rows(labels / "extraction_checks.csv")
    missing = completeness(pairs, matches, sample, checks)
    for name, items in missing.items():
        print(f"  {name}: {len(items)}{' -> ' + ', '.join(items[:12]) if items else ''}")
    complete = not any(missing.values())
    if not complete and not allow_incomplete:
        print("  not frozen: finish labeling first (or use --allow-incomplete for a draft)")
        return

    summary = label_summary(pairs, matches, checks)
    (labels / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = collect_files(EVAL_DIR, INTERIM / "resume_parser", RESUME_HF / "raw", Path("."))
    write_manifest(manifest, EVAL_DIR / "manifest.csv")
    status = "frozen" if complete else "draft (labels incomplete)"
    frozen_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    (EVAL_DIR / "README.md").write_text(datasheet(EVAL_DIR, summary, manifest, status, frozen_at), encoding="utf-8")

    print(f"  status: {status} | {len(manifest)} files under checksum "
          f"({sum(m['in_git'] == 'yes' for m in manifest)} in git, {sum(m['in_git'] == 'no' for m in manifest)} private)")
    print(f"  grades by job: {summary['grades_by_job']}")
    for a in summary["agreement"]:
        print(f"  agreement {a['annotators']}: exact {a['exact_agreement']:.0%}, weighted kappa {a['quadratic_weighted_kappa']:.2f}")
    print(f"  -> {EVAL_DIR / 'manifest.csv'}, {labels / 'summary.json'}, {EVAL_DIR / 'README.md'}")


def run_verify() -> None:
    result = verify(EVAL_DIR / "manifest.csv", Path("."))
    if not any(result.values()):
        print("  OK: every file matches the frozen manifest")
    for name, paths in result.items():
        if paths:
            print(f"  {name}: {len(paths)} -> " + ", ".join(paths[:10]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    pool = sub.add_parser("pool", help="select CVs and download their PDFs")
    pool.add_argument("--no-download", action="store_true", help="skip fetching PDFs")
    sub.add_parser("shortlist", help="shortlist job postings per role")
    sub.add_parser("synthetic", help="render synthetic CV profiles to PDF/DOCX")
    pairs = sub.add_parser("pairs", help="select the CV-job pairs and CVs to label")
    pairs.add_argument("--force", action="store_true", help="rebuild even if labels already exist")
    freeze = sub.add_parser("freeze", help="check labels, write manifest, summary and datasheet")
    freeze.add_argument("--allow-incomplete", action="store_true", help="write a draft before labeling is finished")
    sub.add_parser("autocheck", help="automatic extraction checks (no human needed)")
    sub.add_parser("verify", help="compare files with the frozen manifest")
    args = parser.parse_args()

    configure_logging(level="INFO")
    print(f"\n== {args.stage}")
    if args.stage == "pool":
        run_pool(download=not args.no_download)
    elif args.stage == "shortlist":
        run_shortlist()
    elif args.stage == "synthetic":
        run_synthetic()
    elif args.stage == "pairs":
        run_pairs(force=args.force)
    elif args.stage == "autocheck":
        run_autocheck()
    elif args.stage == "freeze":
        run_freeze(allow_incomplete=args.allow_incomplete)
    else:
        run_verify()


if __name__ == "__main__":
    main()
