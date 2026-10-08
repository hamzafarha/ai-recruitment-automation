"""Fine-tuning training package (LoRA / QLoRA with PEFT, TRL, bitsandbytes)."""

from recruitment_ai.training.dataset import RecruitmentDataset
from recruitment_ai.training.qlora import QLoRATrainingConfig, build_qlora_trainer

__all__ = ["RecruitmentDataset", "QLoRATrainingConfig", "build_qlora_trainer"]
