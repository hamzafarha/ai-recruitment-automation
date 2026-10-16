"""Run the Step 2 extraction evaluation on a Kaggle GPU (7B models do not fit the local 4 GB GPU).

Usage (from the project root; needs the Kaggle token in ~/.kaggle/access_token):
  python scripts/kaggle_run.py all       # package + push + wait + fetch (everything)
  python scripts/kaggle_run.py package   # build the private dataset and the job in build/kaggle/
  python scripts/kaggle_run.py push      # upload the dataset (new version if it exists) and start the job
  python scripts/kaggle_run.py status    # job status
  python scripts/kaggle_run.py fetch     # download the reports into data/evaluation/extraction/

What is uploaded: code, the frozen evaluation set (ids, labels, answer keys, synthetic CVs) and the
skill lexicon, as a PRIVATE dataset. Real CV PDFs are not uploaded: the job downloads them from
the public Hugging Face dataset and checks them against the frozen manifest.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(".")
BUILD = ROOT / "build" / "kaggle"
DATASET_SLUG = "recruitment-ai-eval-v1"
KERNEL_SLUG = "recruitment-ai-extraction-eval"
KERNEL_FILE = ROOT / "scripts" / "kaggle" / "extraction_kernel.py"
RESULTS = ROOT / "data" / "evaluation" / "extraction"


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if check and result.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


def username() -> str:
    out = kaggle("config", "view").stdout
    return next(line.split(":", 1)[1].strip() for line in out.splitlines() if "username" in line)


def project_files() -> list[Path]:
    """Everything the Kaggle job needs, nothing personal beyond the (public-dataset) answer keys."""
    import csv

    files = [ROOT / "pyproject.toml", ROOT / "README.md", ROOT / "scripts" / "eval_extraction.py",
             ROOT / "data" / "interim" / "skills" / "skill_lexicon.json"]
    files += [p for p in (ROOT / "src").rglob("*") if p.is_file() and "__pycache__" not in p.parts
              and ".egg-info" not in str(p)]
    files += [p for p in (ROOT / "data" / "processed" / "eval_v1").rglob("*") if p.is_file()]
    split_dirs = {"dev": "validation", "test": "test"}
    with (ROOT / "data" / "processed" / "eval_v1" / "cv_pool.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            files.append(ROOT / "data" / "interim" / "resume_parser" / split_dirs[row["eval_split"]]
                         / f"{row['resume_id']}.json")
    return files


def package() -> str:
    user = username()
    if BUILD.exists():
        shutil.rmtree(BUILD)
    dataset_dir, kernel_dir = BUILD / "dataset", BUILD / "kernel"
    dataset_dir.mkdir(parents=True)
    kernel_dir.mkdir(parents=True)

    files = project_files()
    with tarfile.open(dataset_dir / "project.tar.gz", "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=f"project/{path.relative_to(ROOT).as_posix()}")
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}],
    }, indent=1), encoding="utf-8")

    shutil.copy(KERNEL_FILE, kernel_dir / "run.py")
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [], "kernel_sources": [],
    }, indent=1), encoding="utf-8")
    size = (dataset_dir / "project.tar.gz").stat().st_size / 1e6
    print(f"  packaged {len(files)} files ({size:.1f} MB) -> {dataset_dir}; job -> {kernel_dir}")
    return user


def push() -> None:
    user = username()
    exists = DATASET_SLUG in kaggle("datasets", "list", "--mine", "--csv").stdout
    if exists:
        kaggle("datasets", "version", "-p", str(BUILD / "dataset"), "-m", "update eval set / code")
    else:
        kaggle("datasets", "create", "-p", str(BUILD / "dataset"))
    print(f"  dataset {'new version' if exists else 'created'}: {user}/{DATASET_SLUG} (private)")
    for _ in range(60):  # wait until the upload is processed, otherwise the job would not see it
        if "ready" in kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False).stdout.lower():
            break
        time.sleep(10)
    out = kaggle("kernels", "push", "-p", str(BUILD / "kernel")).stdout
    print("  " + out.strip().replace("\n", "\n  "))


def status() -> str:
    out = kaggle("kernels", "status", f"{username()}/{KERNEL_SLUG}", check=False).stdout.strip()
    print("  " + out)
    return out.lower()


def wait(poll_seconds: int = 120) -> str:
    while True:
        state = status()
        if any(s in state for s in ("complete", "error", "cancel")):
            return state
        time.sleep(poll_seconds)


def fetch() -> None:
    target = BUILD / "output"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    kaggle("kernels", "output", f"{username()}/{KERNEL_SLUG}", "-p", str(target))
    runs = [p for p in (target / "extraction").iterdir() if p.is_dir()] if (target / "extraction").exists() else []
    for run in runs:
        shutil.copytree(run, RESULTS / run.name, dirs_exist_ok=True)
        print(f"  {run.name} -> {RESULTS / run.name}")
    if not runs:
        print(f"  no results found; see the job log in {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("all", "package", "push", "status", "fetch"))
    args = parser.parse_args()
    if args.command in ("all", "package"):
        package()
    if args.command in ("all", "push"):
        push()
    if args.command == "status":
        status()
    if args.command == "all":
        print(f"  final state: {wait()}")
    if args.command in ("all", "fetch"):
        fetch()


if __name__ == "__main__":
    main()
