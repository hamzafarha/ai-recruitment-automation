"""QLoRA / LoRA fine-tuning pipeline configuration for Qwen2.5-7B-Instruct."""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QLoRATrainingConfig(BaseModel):
    """Configuration for QLoRA fine-tuning with PEFT and bitsandbytes."""

    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    adapter_output_dir: str = "./models/recruitment-specialized-adapter"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048


def build_qlora_trainer(config: QLoRATrainingConfig, train_dataset: Any = None) -> Any:
    """Instantiate SFTTrainer with PEFT LoraConfig and 4-bit bitsandbytes quantization."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        _ = AutoTokenizer.from_pretrained(
            config.base_model_name, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        logger.info(f"Initialized PEFT adapter for {config.base_model_name}")
        return model
    except ImportError as exc:
        logger.info(
            f"Training dependencies (peft, trl, bitsandbytes) not present in lightweight env: {exc}"
        )
        return None
