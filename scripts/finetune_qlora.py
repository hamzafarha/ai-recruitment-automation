"""Step 9 — QLoRA fine-tuning of Qwen2.5 for CV extraction.

Run on Kaggle by notebooks/kaggle/finetune_qwen_qlora.ipynb, one process per GPU:
  torchrun --standalone --nproc_per_node 2 scripts/finetune_qlora.py --config train_config.json

The config (see recruitment_ai.training.qlora.QLoRAConfig) is written by the notebook. Running
it again with the same output_dir resumes from the latest checkpoint.
"""

import argparse
from pathlib import Path

from recruitment_ai.training.qlora import QLoRAConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="JSON file with the QLoRAConfig fields")
    args = parser.parse_args()
    train(QLoRAConfig.model_validate_json(args.config.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
