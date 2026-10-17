"""Run the project's evaluation jobs on Kaggle's free GPUs (2x T4; quota: 30 h per week).

No model runs on our own computers: every model-heavy job goes through this script.
Remaining GPU quota: python -m kaggle quota

Usage (from the project root; needs the Kaggle token in ~/.kaggle/access_token):
  python scripts/kaggle_run.py all <job> [<job> ...]   # package + push + wait + fetch (jobs run in parallel)
  python scripts/kaggle_run.py package                 # build the private dataset (shared by all jobs)
  python scripts/kaggle_run.py upload                  # send the latest code to Kaggle, start no job
  python scripts/kaggle_run.py push <job> [<job> ...]  # upload the dataset (new version) and start the jobs
  python scripts/kaggle_run.py status <job> [<job> ...]
  python scripts/kaggle_run.py fetch <job> [<job> ...] # download the reports into data/evaluation/<folder>/

Jobs: extraction, extraction-text, extraction-vlm (Step 2, one engine each), rag (Step 3),
finetune (Step 9: QLoRA notebook) and extraction-ft (Step 9: evaluation of the fine-tuned model).

What is uploaded: code, knowledge base, the frozen evaluation set (ids, labels, answer keys,
synthetic CVs) and the skill lexicon, as one PRIVATE dataset. Real CV PDFs are not uploaded:
jobs that need them download them from the public Hugging Face dataset and check them against
the frozen manifest.
"""

import argparse
import csv
import io
import json
import os
import re
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
KERNELS = ROOT / "scripts" / "kaggle"
# job -> Kaggle job: slug, code (script or notebook), output folder (copied to data/evaluation/<output>),
# constants set in the code ("set"), and other jobs whose output is attached as an input ("inputs")
JOBS: dict[str, dict] = {
    "extraction": {"slug": "recruitment-ai-extraction-eval", "code": KERNELS / "extraction_kernel.py",
                   "output": "extraction", "set": {"ENGINE": "ocr_llm"}},
    "extraction-text": {"slug": "recruitment-ai-extraction-text-eval", "code": KERNELS / "extraction_kernel.py",
                        "output": "extraction", "set": {"ENGINE": "text_llm"}},
    "extraction-vlm": {"slug": "recruitment-ai-extraction-vlm-eval", "code": KERNELS / "extraction_kernel.py",
                       "output": "extraction", "set": {"ENGINE": "vlm"}},
    "rag": {"slug": "recruitment-ai-rag-eval", "code": KERNELS / "rag_kernel.py", "output": "rag"},
    "finetune": {"slug": "recruitment-ai-finetune", "code": ROOT / "notebooks" / "kaggle" / "finetune_qwen_qlora.ipynb",
                 "output": "finetune"},
    # the real pipeline again, with Qwen + the LoRA adapter produced by the "finetune" job
    "extraction-ft": {"slug": "recruitment-ai-extraction-ft-eval", "code": KERNELS / "extraction_kernel.py",
                      "output": "extraction", "set": {"ENGINE": "ocr_llm", "ADAPTER": "auto"}, "inputs": ["finetune"]},
}


def set_constants(code: str, values: dict[str, str]) -> str:
    """Give module-level constants (``NAME = ...`` lines) their value for one job."""
    for name, value in values.items():
        code = re.sub(rf"^{name} = .*$", f'{name} = "{value}"', code, count=1, flags=re.MULTILINE)
    return code


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # UTF-8 mode: on Windows the CLI would otherwise write job logs (progress bars, accents) as
    # cp1252 and crash with "'charmap' codec can't encode characters"
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", env=env)
    if check and result.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}")
    return result


def username() -> str:
    out = kaggle("config", "view").stdout
    return next(line.split(":", 1)[1].strip() for line in out.splitlines() if "username" in line)


def project_files() -> list[Path]:
    """Everything the jobs need; nothing personal beyond the (public-dataset) answer keys."""
    files = [ROOT / "pyproject.toml", ROOT / "README.md", ROOT / "scripts" / "eval_extraction.py",
             ROOT / "scripts" / "eval_rag.py", ROOT / "scripts" / "finetune_qlora.py",
             ROOT / "data" / "interim" / "skills" / "skill_lexicon.json"]
    for folder in (ROOT / "src", ROOT / "data" / "knowledge_base", ROOT / "data" / "processed" / "eval_v1"):
        files += [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and ".egg-info" not in str(p)]
    split_dirs = {"dev": "validation", "test": "test"}
    with (ROOT / "data" / "processed" / "eval_v1" / "cv_pool.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            files.append(ROOT / "data" / "interim" / "resume_parser" / split_dirs[row["eval_split"]]
                         / f"{row['resume_id']}.json")
    return files


def package() -> None:
    user = username()
    if BUILD.exists():
        shutil.rmtree(BUILD)
    dataset_dir = BUILD / "dataset"
    dataset_dir.mkdir(parents=True)
    files = project_files()
    # A unique id per package, written into the dataset and into every job: a job started on an
    # older dataset version (Kaggle still processing the upload) notices it and stops at once
    package_id = time.strftime("%Y%m%d-%H%M%S")
    with tarfile.open(dataset_dir / "project.tar.gz", "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=f"project/{path.relative_to(ROOT).as_posix()}")
        info = tarfile.TarInfo("project/PACKAGE_ID")
        info.size, info.mtime = len(package_id), int(time.time())
        tar.addfile(info, io.BytesIO(package_id.encode()))
    (BUILD / "PACKAGE_ID").write_text(package_id, encoding="utf-8")
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}],
    }, indent=1), encoding="utf-8")
    for job, spec in JOBS.items():
        kernel_dir = BUILD / f"kernel_{job}"
        kernel_dir.mkdir(parents=True)
        values = {"PACKAGE_ID": package_id, **spec.get("set", {})}
        if spec["code"].suffix == ".ipynb":  # notebook: the constants live in its code cells
            notebook = json.loads(spec["code"].read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    cell["source"] = set_constants("".join(cell["source"]), values).splitlines(keepends=True)
            code_file, kernel_type = "notebook.ipynb", "notebook"
            (kernel_dir / code_file).write_text(json.dumps(notebook, ensure_ascii=False), encoding="utf-8")
        else:  # one script can serve several jobs (e.g. one extraction engine each)
            code_file, kernel_type = "run.py", "script"
            (kernel_dir / code_file).write_text(set_constants(spec["code"].read_text(encoding="utf-8"), values),
                                                encoding="utf-8")
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{spec['slug']}", "title": spec["slug"], "code_file": code_file,
            "language": "python", "kernel_type": kernel_type, "is_private": True,
            # free GPU session: "GPU T4 x2" (2 x 16 GB). The P100 is avoided: recent PyTorch
            # builds no longer support it
            "enable_gpu": True, "machine_shape": "NvidiaTeslaT4", "enable_internet": True,
            "dataset_sources": [f"{user}/{DATASET_SLUG}"], "competition_sources": [],
            "kernel_sources": [f"{user}/{JOBS[other]['slug']}" for other in spec.get("inputs", [])],
        }, indent=1), encoding="utf-8")
    size = (dataset_dir / "project.tar.gz").stat().st_size / 1e6
    print(f"  packaged {len(files)} files ({size:.1f} MB); jobs: {', '.join(JOBS)}")


def wait_for_dataset_version(user: str, package_id: str, max_minutes: int = 15) -> None:
    """Wait until Kaggle serves the version just uploaded.

    Kaggle keeps serving the previous version while it processes an upload, and "datasets status"
    already says "ready" during that time: a job started too early silently ran old code. So we
    download the package id of the latest version until it is ours.
    """
    target = BUILD / "version_check"
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        shutil.rmtree(target, ignore_errors=True)
        kaggle("datasets", "download", f"{user}/{DATASET_SLUG}", "-f", "project/PACKAGE_ID", "-p", str(target),
               check=False)
        served = target / "PACKAGE_ID"
        if served.exists() and served.read_text(encoding="utf-8").strip() == package_id:
            print(f"  dataset version {package_id} is live")
            return
        time.sleep(15)
    sys.exit(f"Kaggle still serves an older dataset version after {max_minutes} min; run the push again later.")


def upload_dataset(note: str) -> None:
    """Send the package as a new version of the private dataset and wait until Kaggle serves it."""
    user = username()
    exists = DATASET_SLUG in kaggle("datasets", "list", "--mine", "--csv").stdout
    if exists:
        kaggle("datasets", "version", "-p", str(BUILD / "dataset"), "-m", note)
    else:
        kaggle("datasets", "create", "-p", str(BUILD / "dataset"))
    print(f"  dataset {'new version' if exists else 'created'}: {user}/{DATASET_SLUG} (private)")
    wait_for_dataset_version(user, (BUILD / "PACKAGE_ID").read_text(encoding="utf-8").strip())


def push(jobs: list[str]) -> None:
    upload_dataset(f"update for {', '.join(jobs)}")
    for job in jobs:
        out = kaggle("kernels", "push", "-p", str(BUILD / f"kernel_{job}")).stdout
        print("  " + out.strip().replace("\n", "\n  "))


def status(job: str) -> str:
    out = kaggle("kernels", "status", f"{username()}/{JOBS[job]['slug']}", check=False).stdout.strip()
    print(f"  {job}: {out}")
    return out.lower()


def wait(job: str, poll_seconds: int = 120) -> str:
    while True:
        state = status(job)
        if any(s in state for s in ("complete", "error", "cancel")):
            return state
        time.sleep(poll_seconds)


def fetch(job: str) -> None:
    slug, name = JOBS[job]["slug"], JOBS[job]["output"]
    target = BUILD / f"output_{job}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    kaggle("kernels", "output", f"{username()}/{slug}", "-p", str(target))
    produced = target / name
    if not produced.exists():
        print(f"  no results found; see the job log in {target}")
        return
    shutil.copytree(produced, ROOT / "data" / "evaluation" / name, dirs_exist_ok=True)
    print(f"  results -> data/evaluation/{name}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("all", "package", "upload", "push", "status", "fetch"))
    parser.add_argument("jobs", nargs="*", default=["extraction"], help=", ".join(JOBS))
    args = parser.parse_args()
    if unknown := [job for job in args.jobs if job not in JOBS]:
        parser.error(f"unknown job(s) {', '.join(unknown)}; choose from {', '.join(JOBS)}")
    if args.command in ("all", "package", "upload"):
        package()
    if args.command == "upload":  # code only, no job: e.g. for notebooks/kaggle/model_server.ipynb
        upload_dataset("code update")
        return
    if args.command in ("all", "push"):
        push(args.jobs)
    for job in args.jobs:
        if args.command == "status":
            status(job)
        if args.command == "all":
            print(f"  {job} final state: {wait(job)}")
        if args.command in ("all", "fetch"):
            fetch(job)


if __name__ == "__main__":
    main()
