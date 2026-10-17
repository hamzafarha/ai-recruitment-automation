"""Kaggle GPU job (2x T4): Step 2 extraction evaluation with Qwen2.5 run by Hugging Face Transformers.

Pushed and launched by scripts/kaggle_run.py, one Kaggle job per engine; do not run it locally.
On the Kaggle machine it:
1. unpacks the project (code + frozen evaluation set) from the attached private dataset;
2. installs the Python dependencies (the model is downloaded from the Hugging Face Hub on first use);
3. downloads the real CVs (PDFs and the dataset's text) from the public Hugging Face dataset
   (nothing private is uploaded);
4. runs scripts/eval_extraction.py for ENGINE — the script first checks every file against the
   frozen manifest, so Kaggle evaluates exactly our eval_v1 files;
5. copies the reports to /kaggle/working/ (the job's downloadable output).
"""

import csv
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

PACKAGE_ID = "local"  # set by scripts/kaggle_run.py: must match the attached dataset version
ENGINE = "ocr_llm"  # set per job by scripts/kaggle_run.py: ocr_llm, text_llm or vlm
ADAPTER = ""  # set by scripts/kaggle_run.py: "auto" = the LoRA adapter of the finetune job (attached input)
MODELS = {  # the architecture's 7B models, float16, split across the two T4 GPUs (~15-17 GB)
    "ocr_llm": "Qwen/Qwen2.5-7B-Instruct",      # real pipeline: document -> OCR/text layer -> Qwen
    "text_llm": "Qwen/Qwen2.5-7B-Instruct",     # same model on the dataset's clean text: cost of OCR
    "vlm": "Qwen/Qwen2.5-VL-7B-Instruct",       # page images read directly, no OCR
}
SPLIT = "dev"
# Safety net for the weekly GPU quota (30 h): a normal run takes 1-3 h
BUDGET_MINUTES = 240
PROJECT = Path("/tmp/project")
# Print every log line immediately: Kaggle shows the log live, but Python buffers
# output that does not go to a terminal (lines would appear in late bursts)
os.environ["PYTHONUNBUFFERED"] = "1"
OUTPUT = Path("/kaggle/working")
REPO = "md-zain/resume-parser-dataset"


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


def download_cvs() -> None:
    """PDFs of the evaluation CVs (checked by the frozen manifest) and the dataset's text."""
    from huggingface_hub import hf_hub_download

    target = "data/raw/resumes/resume_parser/hf"
    hf_hub_download(REPO, f"curated/{'validation' if SPLIT == 'dev' else 'test'}.jsonl", repo_type="dataset",
                    local_dir=target)
    rows = list(csv.DictReader(open("data/processed/eval_v1/cv_pool.csv", encoding="utf-8")))
    for i, row in enumerate(rows, 1):
        hf_hub_download(REPO, f"raw/{row['category']}/{row['resume_id']}.pdf", repo_type="dataset", local_dir=target)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)} PDFs", flush=True)


def find_adapter() -> Path | None:
    """The fine-tuned LoRA adapter in the attached output of the finetune job."""
    return next((p.parent for p in Path("/kaggle/input").rglob("adapter/adapter_config.json")), None)


def main() -> None:
    start = time.time()
    unpack_project()
    adapter = find_adapter() if ADAPTER else None
    if ADAPTER and adapter is None:  # stop before using GPU time
        sys.exit("This job evaluates the fine-tuned model, but no adapter is attached: run the finetune job "
                 "until it completes, then push this job again.")
    # torch (with CUDA) and torchvision come with Kaggle's image; transformers is upgraded for Qwen2.5-VL
    sh("pip install -q -U 'transformers>=4.56' 'accelerate>=1.0' 'peft>=0.13' && pip install -q pymupdf "
       "python-docx 'rapidocr>=3.0' onnxruntime pydantic-settings email-validator huggingface_hub httpx pyyaml pillow")
    # Use the code from src/ directly instead of "pip install -e .": pyproject pins Python 3.12,
    # which Kaggle's image may not match
    os.environ["PYTHONPATH"] = str(PROJECT / "src")
    os.environ.update({"LLM_DEVICE": "cuda", "LLM_DTYPE": "float16",  # T4: no bfloat16
                       # less fragmented GPU memory: fewer out-of-memory errors on long CVs
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    sh("python --version && nvidia-smi --query-gpu=name,memory.total --format=csv && python -c "
       "'import torch, transformers; print(torch.__version__, transformers.__version__, "
       "torch.cuda.device_count(), \"GPUs\")'", check=False)
    download_cvs()

    code = sh(f"python scripts/eval_extraction.py --engine {ENGINE} --model {MODELS[ENGINE]} --split {SPLIT} "
              f"--max-minutes {BUDGET_MINUTES}" + (f" --adapter {adapter}" if adapter else ""), check=False)

    if Path("data/evaluation/extraction").exists():
        shutil.copytree("data/evaluation/extraction", OUTPUT / "extraction", dirs_exist_ok=True)
    print(f"\nDone in {(time.time() - start) / 60:.0f} min", flush=True)
    sys.exit(code)  # a failed evaluation shows as an error on Kaggle


if __name__ == "__main__":
    main()
