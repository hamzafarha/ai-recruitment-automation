"""Kaggle GPU job (2x T4): evaluation of the application's AI layers.

Pushed and launched by scripts/kaggle_run.py; do not run it locally. TASK is set per job:
- "matching" (DSO3): scripts/eval_matching.py — fast semantic matching (thresholds calibrated on
  dev) and the full LLM analysis (RAG + Qwen2.5-7B) on the 36 graded CV-job pairs;
- "screening" (DSO4): scripts/eval_matching.py --methods semantic,llm,agent — the application
  pipeline's screening agent on the same pairs, next to the fixed analysis (whose saved answers are
  shipped with the code, so only the agent runs);
- "agent": scripts/eval_agent.py — the LangGraph assistant on 20 recruiter questions,
  in a fresh workspace (SQLite, embedded Qdrant, demo data).
On the Kaggle machine it unpacks the project, installs the dependencies, downloads what the task
needs from the public CV dataset, runs the evaluation and copies the reports to /kaggle/working.
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PACKAGE_ID = "local"  # set by scripts/kaggle_run.py: must match the attached dataset version
TASK = "matching"     # set per job by scripts/kaggle_run.py: matching, screening or agent
BUDGET_MINUTES = 150  # safety net for the weekly GPU quota: no new LLM analysis after this time
PROJECT = Path("/tmp/project")
OUTPUT = Path("/kaggle/working")
REPO = "md-zain/resume-parser-dataset"
os.environ["PYTHONUNBUFFERED"] = "1"


def sh(cmd: str, check: bool = True) -> int:
    print(f"\n+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check).returncode


def unpack_project() -> None:
    archive = next(Path("/kaggle/input").rglob("project.tar.gz"), None)
    if archive:
        with tarfile.open(archive) as tar:
            tar.extractall(PROJECT.parent)
    else:  # Kaggle may have extracted the archive already
        shutil.copytree(next(p for p in Path("/kaggle/input").rglob("project") if p.is_dir()), PROJECT)
    os.chdir(PROJECT)
    attached = (PROJECT / "PACKAGE_ID").read_text().strip() if (PROJECT / "PACKAGE_ID").exists() else "none"
    if attached != PACKAGE_ID:  # stop before using any GPU time on old code
        sys.exit(f"Wrong dataset version: the job expects package {PACKAGE_ID} but got {attached} "
                 "(Kaggle was still processing the upload). Push the job again.")


def main() -> None:
    start = time.time()
    unpack_project()
    sh("pip uninstall -y -q torchao", check=False)  # old torchao on Kaggle's image breaks PEFT; unused
    sh("pip install -q -U 'transformers>=4.56' 'accelerate>=1.0' && pip install -q 'qdrant-client>=1.19,<1.20' "
       "'sentence-transformers>=3' 'langgraph>=0.2' pymupdf python-docx pydantic-settings email-validator "
       "huggingface_hub httpx pyyaml sqlalchemy")
    os.environ.update({
        "PYTHONPATH": str(PROJECT / "src"),
        "LLM_PROVIDER": "huggingface", "LLM_DEVICE": "cuda", "LLM_DTYPE": "float16",  # T4: no bfloat16
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "QDRANT_LOCATION": ":memory:",  # embedded Qdrant (no Docker on Kaggle)
        "EMBEDDING_PROVIDER": "local", "EMBEDDING_DEVICE": "cuda", "EMBEDDING_MODEL": "BAAI/bge-m3",
        "RERANKING_MODEL": "BAAI/bge-reranker-v2-m3", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "N8N_EVENTS_ENABLED": "false",
    })
    sh("python --version && nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    if TASK in ("matching", "screening"):
        from huggingface_hub import hf_hub_download

        for split in ("validation", "test"):  # the real CVs' own text (public dataset)
            hf_hub_download(REPO, f"curated/{split}.jsonl", repo_type="dataset",
                            local_dir="data/raw/resumes/resume_parser/hf")
        methods = " --methods semantic,llm,agent" if TASK == "screening" else ""
        code = sh(f"python scripts/eval_matching.py{methods} --max-minutes {BUDGET_MINUTES}", check=False)
        produced = Path("data/evaluation/matching")
    else:
        os.environ["DATABASE_URL"] = "sqlite:////tmp/agent_eval.db"
        code = sh("python scripts/eval_agent.py", check=False)
        produced = Path("data/evaluation/agent")
    if produced.exists():
        shutil.copytree(produced, OUTPUT / produced.name, dirs_exist_ok=True)
    print(f"\nDone in {(time.time() - start) / 60:.0f} min", flush=True)
    sys.exit(code)  # a failed evaluation shows as an error on Kaggle


if __name__ == "__main__":
    main()
