"""Step 1b — prepare the raw datasets for the evaluation set.

Reads data/raw/ (never modified) and writes cleaned data to data/interim/:
  resume_parser/<split>/<id>.json  CandidateProfile references (converted labels)
  resume_parser/index.csv          quality flags per CV (leak, corrupted, template, tech...)
  tech_jobs/jobs_clean.json        deduplicated job postings with a target role
  skills/skill_lexicon.json        EN/FR skill dictionary (ESCO + custom extension)

Usage (from the project root):
  python scripts/prepare_data.py                 # everything
  python scripts/prepare_data.py --only jobs     # one part: resumes | jobs | lexicon
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from recruitment_ai.data.resume_parser import prepare_resume_parser
from recruitment_ai.data.skill_lexicon import SkillNormalizer, build_skill_lexicon
from recruitment_ai.data.tech_jobs import prepare_tech_jobs

INTERIM = Path("data/interim")


def run_resumes() -> None:
    counts = prepare_resume_parser()
    print(f"  converted labels: {counts}")
    with (INTERIM / "resume_parser" / "index.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["is_tech_category"] == "True"]
    def flag(cvs: list[dict[str, str]], name: str) -> int:
        return sum(r[name] == "True" for r in cvs)

    for split in ("validation", "test"):
        tech = [r for r in rows if r["split"] == split]
        print(
            f"  {split:10s} tech-category CVs: {len(tech)} | leak_train: {flag(tech, 'leak_train')} | "
            f"corrupted: {flag(tech, 'corrupted')} | template_identity: {flag(tech, 'template_identity')} | "
            f"not a tech job title (excluded from eval pool): {len(tech) - flag(tech, 'tech_title_match')}"
        )


def run_jobs() -> None:
    stats = prepare_tech_jobs()
    print(f"  {json.dumps(stats)}")


def run_lexicon() -> None:
    stats = build_skill_lexicon()
    print(f"  {json.dumps(stats)}")
    # Quick quality indicator: how many technical/tool skills of the tech test CVs are recognised?
    # (The long tail of rare skills is left to embedding similarity in the matching step.)
    normalizer = SkillNormalizer.from_file()
    index_path = INTERIM / "resume_parser" / "index.csv"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as f:
            tech_ids = {r["resume_id"] for r in csv.DictReader(f) if r["is_tech_category"] == "True"}
        skills = [
            s["name"]
            for p in (INTERIM / "resume_parser" / "test").glob("*.json")
            if p.stem in tech_ids
            for s in json.loads(p.read_text(encoding="utf-8"))["skills"]
            if s["category"] in ("Technical", "Tool")
        ]
        print(f"  lexicon recognises {normalizer.coverage(skills):.0%} of {len(skills)} technical/tool skills in tech test CVs")
    for raw in ("ReactJS", "Python 3", "apprentissage automatique", "AWS", "TensorFlow"):
        concept = normalizer.normalize(raw)
        print(f"    {raw!r:28} -> {concept.label_en + ' / ' + concept.label_fr if concept else None}")


STEPS = {"resumes": run_resumes, "jobs": run_jobs, "lexicon": run_lexicon}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=list(STEPS), help="run a single step")
    args = parser.parse_args()

    # The lexicon check reads converted resumes, so resumes run first
    for name in [args.only] if args.only else list(STEPS):
        print(f"\n== {name}")
        start = time.perf_counter()
        STEPS[name]()
        print(f"  done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
