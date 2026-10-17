"""Fine-tuning (Step 9): QLoRA of Qwen2.5 for CV extraction, run on Kaggle's free GPUs.

``data`` builds the training examples (light, testable without a GPU); ``qlora`` trains
(imports torch, peft and bitsandbytes only when called).
"""
