"""QLoRA fine-tuning of Qwen2.5 for CV extraction (Step 9), made for Kaggle's free GPUs.

Launched by scripts/finetune_qlora.py under torchrun, one process per GPU: the two T4s train
in parallel (data parallel), which halves the GPU hours. Built to survive Kaggle's limits:
- the base model is loaded in 4-bit (NF4, ~5.5 GB per GPU) and only the LoRA weights train;
- float16 everywhere (the T4 has no bfloat16);
- a checkpoint every few minutes, and the run stops by itself at ``stop_at`` (before the
  session ends), so the outputs are always saved;
- a new run resumes from the latest checkpoint (same data order, same optimizer state);
- optional copy of each checkpoint to a private Hugging Face repository, in case the whole
  Kaggle session is lost.
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from recruitment_ai.training.data import IGNORE_INDEX, read_jsonl, tokenize_example

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class QLoRAConfig(BaseModel):
    """Everything a training run needs; written to JSON by the Kaggle notebook."""

    model_path: str = Field(description="Hugging Face id or local folder of the base model")
    train_file: str = Field(description="JSONL from training.data.build_training_set")
    output_dir: str = Field(description="checkpoints (a new run resumes from the latest)")
    adapter_dir: str = Field(description="final LoRA adapter, written when training completes")
    state_file: str = Field(description="JSON summary: complete / paused, steps, losses")
    max_seq_len: int = 4096
    epochs: float = 1.0
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 1
    grad_accum: int = 4
    warmup_fraction: float = 0.03
    max_grad_norm: float = 0.3  # QLoRA paper value
    seed: int = 42
    logging_steps: int = 5
    stop_at: float | None = Field(default=None, description="time.time() after which the run saves and stops")
    save_every_minutes: float = 15.0
    hub_backup_repo: str | None = Field(default=None, description="private HF repo for checkpoint copies")


def _all_agree(flag: bool) -> bool:
    """True on every GPU process if it is True on any: all processes must save and stop at the
    same step, otherwise the next synchronisation would wait forever."""
    import torch
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return flag
    value = torch.tensor([int(flag)], device=f"cuda:{torch.cuda.current_device()}")
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def _callbacks(cfg: QLoRAConfig, is_main: bool) -> list[Any]:
    from transformers import TrainerCallback

    class TimeBudget(TrainerCallback):
        """Checkpoint every ``save_every_minutes``; save and stop at ``stop_at``."""

        def __init__(self) -> None:
            self.last_save = time.time()
            self.stopped_early = False

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            now = time.time()
            stop = _all_agree(cfg.stop_at is not None and now >= cfg.stop_at)
            if stop or _all_agree(now - self.last_save >= cfg.save_every_minutes * 60):
                control.should_save = True
            if stop:
                control.should_training_stop = True
                self.stopped_early = True
            return control

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            self.last_save = time.time()
            return control

    class Progress(TrainerCallback):
        """One readable line per log: step, loss, elapsed time and estimated time left."""

        def __init__(self) -> None:
            self.start = time.time()
            self.first_step: int | None = None

        def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> Any:
            if not is_main or not logs or "loss" not in logs:
                return control
            if self.first_step is None:
                self.first_step = state.global_step
            done = max(state.global_step - self.first_step, 1)
            minutes = (time.time() - self.start) / 60
            eta = minutes / done * (state.max_steps - state.global_step)
            print(f"  step {state.global_step}/{state.max_steps} | loss {logs['loss']:.4f} | "
                  f"{minutes:.0f} min | about {eta:.0f} min left", flush=True)
            return control

    class HubBackup(TrainerCallback):
        """Copy each checkpoint to a private Hugging Face repo (in the background)."""

        def __init__(self, repo: str) -> None:
            from huggingface_hub import HfApi

            self.api, self.repo, self.uploads = HfApi(), repo, []
            self.api.create_repo(repo, private=True, exist_ok=True)

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            folder = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if folder.exists():
                self.uploads.append(self.api.upload_folder(
                    folder_path=str(folder), path_in_repo="last-checkpoint", repo_id=self.repo,
                    delete_patterns="*", commit_message=f"checkpoint {state.global_step}", run_as_future=True))
            return control

    callbacks: list[Any] = [TimeBudget(), Progress()]
    if is_main and cfg.hub_backup_repo:
        callbacks.append(HubBackup(cfg.hub_backup_repo))
    return callbacks


def collate_examples(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    """Pad a batch; with one example, ask the model for the answer's logits only.

    The loss only covers the answer (the prompt's labels are IGNORE_INDEX), but by default the
    model computes the logits of every position: 4,096 positions x Qwen's 152k-token vocabulary
    is several GB and ran a T4 out of memory. ``logits_to_keep`` keeps the last positions only
    (the answer plus the token before it, which predicts its first token); the labels are cut to
    the same positions, so the loss is unchanged.
    """
    import torch

    width = max(len(b["input_ids"]) for b in batch)

    def pad(key: str, value: int) -> Any:
        return torch.tensor([b[key] + [value] * (width - len(b[key])) for b in batch])

    out: dict[str, Any] = {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0),
                           "labels": pad("labels", IGNORE_INDEX)}
    if len(batch) == 1:  # with padding, answers would end at different positions
        labels = batch[0]["labels"]
        first_answer = next((i for i, label in enumerate(labels) if label != IGNORE_INDEX), 1)
        keep = len(labels) - first_answer + 1
        out["logits_to_keep"] = keep
        out["labels"] = out["labels"][:, -keep:]
    return out


def train(cfg: QLoRAConfig) -> dict[str, Any]:
    """Train (or resume) and write ``state_file``; returns the same summary."""
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint

    start = time.time()
    local_rank, world = int(os.environ.get("LOCAL_RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    is_main = int(os.environ.get("RANK", "0")) == 0
    torch.cuda.set_device(local_rank)

    def log(message: str) -> None:
        if is_main:
            print(message, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    examples = read_jsonl(cfg.train_file)
    dataset = [t for t in (tokenize_example(tokenizer, e["messages"], cfg.max_seq_len) for e in examples) if t]
    skipped = len(examples) - len(dataset)
    log(f"{len(dataset)} training examples ({skipped} longer than {cfg.max_seq_len} tokens skipped), "
        f"{world} GPU process(es)")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, dtype=torch.float16, device_map={"": local_rank}, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16),
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing
    # Unlike peft's prepare_model_for_kbit_training, nothing is cast to float32: a float32 norm
    # weight turns Qwen's activations into float32 (weight * fp16 -> fp32), doubling the memory of
    # every layer; it ran a T4 out of memory. The trained LoRA weights are still float32.
    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                             target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM"))
    if is_main:
        model.print_trainable_parameters()

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        return collate_examples(batch, tokenizer.pad_token_id)

    steps = math.ceil(len(dataset) / (cfg.batch_size * cfg.grad_accum * world) * cfg.epochs)
    args = TrainingArguments(
        output_dir=cfg.output_dir, per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum, num_train_epochs=cfg.epochs, learning_rate=cfg.learning_rate,
        lr_scheduler_type="cosine", warmup_steps=max(1, round(steps * cfg.warmup_fraction)),
        max_grad_norm=cfg.max_grad_norm, fp16=True, bf16=False, optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.logging_steps, save_strategy="steps", save_steps=max(10, steps // 4),
        save_total_limit=2, report_to="none", seed=cfg.seed, data_seed=cfg.seed, remove_unused_columns=False,
        ddp_find_unused_parameters=False, dataloader_num_workers=2, disable_tqdm=True,
    )
    callbacks = _callbacks(cfg, is_main)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=collate, callbacks=callbacks)

    last = get_last_checkpoint(cfg.output_dir) if Path(cfg.output_dir).is_dir() else None
    log(f"resuming from {last}" if last else f"starting a new run: {steps} optimizer steps")
    trainer.train(resume_from_checkpoint=last)

    complete = trainer.state.global_step >= trainer.state.max_steps
    summary = {
        "status": "complete" if complete else "paused", "global_step": trainer.state.global_step,
        "max_steps": trainer.state.max_steps, "examples": len(dataset), "skipped_too_long": skipped,
        "gpu_processes": world, "resumed_from": last, "minutes_this_session": round((time.time() - start) / 60, 1),
        "losses": [{"step": h["step"], "loss": h["loss"]} for h in trainer.state.log_history if "loss" in h],
        "config": cfg.model_dump(),
    }
    if is_main:
        backup = next((c for c in callbacks if hasattr(c, "uploads")), None)
        if complete:
            trainer.model.save_pretrained(cfg.adapter_dir)
            tokenizer.save_pretrained(cfg.adapter_dir)
            if backup:
                backup.uploads.append(backup.api.upload_folder(
                    folder_path=cfg.adapter_dir, path_in_repo="adapter", repo_id=backup.repo, run_as_future=True))
        for upload in backup.uploads if backup else []:
            upload.result()  # wait for the Hugging Face copies before the process ends
        Path(cfg.state_file).write_text(json.dumps(summary, indent=1), encoding="utf-8")
        log(f"training {summary['status']}: step {summary['global_step']}/{summary['max_steps']}")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()  # no barrier: the other processes simply end, torchrun waits for all
    return summary
