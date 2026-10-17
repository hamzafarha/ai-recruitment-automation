"""Kaggle GPU job: Step 3 retrieval evaluation (BGE-M3 + Qdrant + BGE reranker).

Pushed and launched by scripts/kaggle_run.py (job "rag"); do not run it locally. On Kaggle it:
1. unpacks the project (code, knowledge base, frozen evaluation set) from the private dataset;
2. installs qdrant-client (Qdrant runs embedded, in memory: Kaggle has no Docker);
3. runs scripts/eval_rag.py: builds the knowledge index and evaluates the 40 test questions in
   4 configurations (dense / + reranker / + role filter / both);
4. copies the report to /kaggle/working/rag (the job's downloadable output).
"""

import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PACKAGE_ID = "local"  # set by scripts/kaggle_run.py: must match the attached dataset version
PROJECT = Path("/tmp/project")
# Print every log line immediately: Kaggle shows the log live, but Python buffers
# output that does not go to a terminal (lines would appear in late bursts)
os.environ["PYTHONUNBUFFERED"] = "1"
OUTPUT = Path("/kaggle/working")


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
    sh("pip install -q 'qdrant-client>=1.19,<1.20' pydantic-settings email-validator pyyaml 'sentence-transformers>=3'")
    os.environ.update({
        "PYTHONPATH": str(PROJECT / "src"),
        "QDRANT_LOCATION": ":memory:",     # embedded Qdrant
        "EMBEDDING_DEVICE": "cuda",        # Kaggle T4 GPU (embeddings + reranker in float16)
        "EMBEDDING_MODEL": "BAAI/bge-m3",
        "RERANKING_MODEL": "BAAI/bge-reranker-v2-m3",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })
    sh("python --version && nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    code = sh("python scripts/eval_rag.py", check=False)
    if Path("data/evaluation/rag").exists():
        shutil.copytree("data/evaluation/rag", OUTPUT / "rag", dirs_exist_ok=True)
    print(f"\nDone in {(time.time() - start) / 60:.0f} min", flush=True)
    sys.exit(code)  # a failed evaluation shows as an error on Kaggle


if __name__ == "__main__":
    main()
