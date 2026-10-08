"""Dataset loading and formatting for recruitment fine-tuning."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TrainingSample(BaseModel):
    """Single sample for supervised fine-tuning of Qwen2.5-7B-Instruct."""

    raw_cv: str = Field(..., description="Raw or parsed candidate CV text")
    job_description: str = Field(..., description="Target job description")
    expected_output: dict[str, Any] = Field(
        ..., description="Ground truth structured extraction / match"
    )

    def to_chat_format(self) -> list[dict[str, str]]:
        """Format sample into Qwen chat template."""
        import json

        system_prompt = (
            "You are an expert AI Recruitment Assistant. Analyze the candidate CV "
            "against the target job description and return an accurate, evidence-backed evaluation."
        )
        user_prompt = (
            f"### CANDIDATE CV:\n{self.raw_cv}\n\n"
            f"### JOB DESCRIPTION:\n{self.job_description}"
        )
        assistant_reply = json.dumps(self.expected_output, indent=2)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_reply},
        ]


class RecruitmentDataset:
    """Manages training and evaluation sets for LoRA fine-tuning."""

    def __init__(self, data_path: Path | None = None) -> None:
        self.data_path = data_path or Path("./data/evaluation")

    def load_samples(self) -> list[TrainingSample]:
        """Load labeled CV-job training pairs."""
        # Returns empty or placeholder set if directory is empty
        return []
