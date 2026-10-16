"""Kaggle GPU job: run the Step 2 extraction evaluation with 7B models (Qwen2.5, Qwen2.5-VL).

Pushed and launched by scripts/kaggle_run.py; do not run it locally. On the Kaggle machine it:
1. unpacks the project (code + frozen evaluation set) from the attached private dataset;
2. installs the Python dependencies and Ollama, and pulls the models;
3. downloads the real CV PDFs from the public Hugging Face dataset (nothing private is uploaded);
4. runs scripts/eval_extraction.py for each (engine, model) in RUNS — the script first checks
   every file against the frozen manifest, so Kaggle evaluates exactly our eval_v1 files;
5. copies the reports to /kaggle/working/ (the job's downloadable output).
"""

import csv
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request
from pathlib import Path

RUNS = [  # (engine, Ollama model)
    ("ocr_llm", "qwen2.5:7b"),     # real pipeline, architecture model
    ("text_llm", "qwen2.5:7b"),    # same model on clean text: cost of OCR
    ("vlm", "qwen2.5vl:7b"),       # multimodal: page images, no OCR
    ("ocr_llm", "qwen2.5:3b"),     # the model that fits the local 4 GB GPU, for comparison
]
SPLIT = "dev"
PROJECT = Path("/tmp/project")
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


def start_ollama(models: list[str]) -> subprocess.Popen:
    sh("apt-get -qq update && apt-get -qq install -y zstd pciutils lshw > /dev/null", check=False)
    sh("curl -fsSL https://ollama.com/install.sh | sh")
    server = subprocess.Popen(["ollama", "serve"], stdout=open(OUTPUT / "ollama.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            urllib.request.urlopen("http://localhost:11434/api/version", timeout=2)
            break
        except OSError:
            time.sleep(2)
    for model in models:
        sh(f"ollama pull {model}")
    return server


def download_pdfs() -> None:
    from huggingface_hub import hf_hub_download

    rows = list(csv.DictReader(open("data/processed/eval_v1/cv_pool.csv", encoding="utf-8")))
    for i, row in enumerate(rows, 1):
        hf_hub_download("md-zain/resume-parser-dataset", f"raw/{row['category']}/{row['resume_id']}.pdf",
                        repo_type="dataset", local_dir="data/raw/resumes/resume_parser/hf")
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} PDFs", flush=True)


def main() -> None:
    start = time.time()
    unpack_project()
    sh("pip install -q pymupdf python-docx 'rapidocr>=3.0' onnxruntime pydantic-settings email-validator "
       "huggingface_hub httpx pyyaml")
    # Use the code from src/ directly instead of "pip install -e .": pyproject pins Python 3.12,
    # which Kaggle's image may not match
    os.environ["PYTHONPATH"] = str(PROJECT / "src")
    sh("python --version && nvidia-smi --query-gpu=name,memory.total --format=csv", check=False)
    server = start_ollama(sorted({model for _, model in RUNS}))
    download_pdfs()

    for engine, model in RUNS:
        sh(f"python scripts/eval_extraction.py --engine {engine} --model {model} --split {SPLIT}", check=False)

    shutil.copytree("data/evaluation/extraction", OUTPUT / "extraction", dirs_exist_ok=True)
    server.terminate()
    print(f"\nDone in {(time.time() - start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
