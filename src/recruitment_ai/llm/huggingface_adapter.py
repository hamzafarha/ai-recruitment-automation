"""Hugging Face Transformers adapter: runs Qwen2.5 (text) and Qwen2.5-VL (vision) inside Python.

No server to install: the model is downloaded from the Hugging Face Hub on first use (cached in
~/.cache/huggingface). The project runs it on Kaggle's free GPUs (LLM_DEVICE=cuda): a 7B model in
float16 (~15 GB) is split across the two 16 GB T4 cards by ``device_map="auto"``.

Transformers cannot force the answer to follow a JSON schema, so ``generate_structured``:
1. asks for JSON: the schema is appended to the prompt and decoding is greedy (deterministic);
2. takes the JSON object out of the answer, and closes it if the token limit cut it off;
3. validates it with Pydantic (the extraction schema is lenient, see ``extractor.CVExtraction``).
"""

import io
import json
import logging
import math
import time
from typing import Any, TypeVar

from pydantic import BaseModel

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

JSON_INSTRUCTION = "\n\nAnswer with one JSON object only, following this JSON schema:\n{schema}"


def structured_prompt(prompt: str, schema: type[BaseModel]) -> str:
    """The user message of a structured request: the task, then the JSON schema to follow.

    Fine-tuning builds its training examples with this same function, so the model is trained
    on exactly the prompt it receives at inference time.
    """
    schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    return prompt + JSON_INSTRUCTION.format(schema=schema_json)


# --------------------------------------------------------------------------- JSON answers


def parse_json_object(text: str) -> Any:
    """The first JSON object in a model answer (code fences and text around it are ignored).

    An answer cut off by the token limit is repaired: the unfinished last item is dropped and
    the open brackets are closed, so everything generated before the cut is kept.
    """
    start = text.find("{")
    if start == -1:
        raise ValueError("The model answer contains no JSON object.")
    text = text[start:]
    try:
        return json.JSONDecoder().raw_decode(text)[0]
    except json.JSONDecodeError:
        return _close_truncated_json(text)


def _close_truncated_json(text: str) -> Any:
    # Places where the text can be cut, latest first: before a comma, after a closed bracket,
    # or right after the opening brace. Each comes with the brackets still open there.
    cuts: list[tuple[int, str]] = [(1, "{")]
    stack: list[str] = []
    in_string = escaped = False
    for i, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()
            cuts.append((i + 1, "".join(stack)))
        elif char == ",":
            cuts.append((i, "".join(stack)))
    for end, still_open in reversed(cuts):
        closing = "".join("}" if b == "{" else "]" for b in reversed(still_open))
        try:
            return json.loads(text[:end] + closing)
        except json.JSONDecodeError:
            continue
    raise ValueError("The model answer is not valid JSON.")


# --------------------------------------------------------------------------- adapter


class HuggingFaceAdapter(LLMProvider):
    """Chat / structured generation with a local Transformers model (text or vision-language)."""

    def __init__(self, settings: Settings | None = None, model: str | None = None,
                 vision: bool | None = None, adapter: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.huggingface_model
        # Optional LoRA adapter from our fine-tuning (folder with adapter_config.json)
        self.adapter = adapter or self.settings.llm_adapter
        # Qwen2.5-VL and similar models read images and need a processor instead of a tokenizer
        self.vision = vision if vision is not None else "-vl" in self.model.lower()
        self.last_stats: dict[str, Any] = {}
        self._model: Any = None
        self._processor: Any = None  # AutoTokenizer (text model) or AutoProcessor (vision model)

    @property
    def device(self) -> str:
        device = self.settings.llm_device
        if device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _ensure_loaded(self) -> None:
        """Load the model on first use (downloads it from the Hugging Face Hub the first time)."""
        if self._model is not None:
            return
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        dtype_name = self.settings.llm_dtype
        if dtype_name == "auto":  # float16 is unreliable on CPU; Kaggle's T4 GPUs lack bfloat16
            dtype_name = "float32" if self.device == "cpu" else "float16"
        start = time.perf_counter()
        loader = AutoProcessor if self.vision else AutoTokenizer
        model_class = AutoModelForImageTextToText if self.vision else AutoModelForCausalLM
        processor = loader.from_pretrained(self.model)
        if self.device == "cpu":
            model = model_class.from_pretrained(self.model, dtype=getattr(torch, dtype_name)).to("cpu")
        else:  # spread the layers over every visible GPU (needs the accelerate package)
            model = model_class.from_pretrained(self.model, dtype=getattr(torch, dtype_name), device_map="auto")
        if self.adapter:  # fine-tuned model = base model + our LoRA adapter
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter)
        # Kept only once complete: if the adapter fails to load, the next call must not silently
        # use the base model (it happened on Kaggle: the "fine-tuned" check ran the base model)
        self._processor, self._model = processor, model.eval()
        logger.info("Loaded %s%s on %s (%s) in %.0f s", self.model, f" + {self.adapter}" if self.adapter else "",
                    self.device, dtype_name, time.perf_counter() - start)

    def free_memory(self) -> None:
        """Release cached GPU memory, e.g. after an out-of-memory error on a very long CV."""
        import gc

        gc.collect()
        if self._model is not None and self.device != "cpu":
            import torch

            torch.cuda.empty_cache()

    def _images(self, images: list[bytes]) -> list[Any]:
        """PNG/JPEG bytes -> RGB images no larger than VLM_MAX_PIXELS (bounds the CPU cost)."""
        from PIL import Image

        out = []
        for data in images:
            image = Image.open(io.BytesIO(data)).convert("RGB")
            scale = math.sqrt(self.settings.vlm_max_pixels / (image.width * image.height))
            if scale < 1:
                image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
            out.append(image)
        return out

    def _messages(self, prompt: str, system_prompt: str | None, n_images: int = 0) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if not self.vision:  # text chat template: plain strings
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            return messages
        if system_prompt:  # vision chat template: lists of typed parts, images before the text
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        parts: list[dict[str, Any]] = [{"type": "image"} for _ in range(n_images)]
        messages.append({"role": "user", "content": [*parts, {"type": "text", "text": prompt}]})
        return messages

    def _generate(self, messages: list[dict[str, Any]], images: list[bytes] | None = None,
                  max_tokens: int = 1024, temperature: float = 0.0) -> str:
        import torch

        self._ensure_loaded()
        start = time.perf_counter()
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.vision:
            inputs = self._processor(text=[text], images=self._images(images) if images else None,
                                     return_tensors="pt")
        else:
            inputs = self._processor(text, return_tensors="pt")
        inputs = inputs.to(self.device)
        sampling: dict[str, Any] = {"do_sample": True, "temperature": temperature} if temperature > 0 else {
            "do_sample": False}
        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=max_tokens, **sampling)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        new_tokens = output[0, prompt_tokens:]
        self.last_stats = {
            "model": self.model,
            "adapter": self.adapter,
            "device": self.device,
            "seconds": round(time.perf_counter() - start, 2),
            "prompt_tokens": prompt_tokens,
            "output_tokens": int(new_tokens.shape[0]),
            # the answer reached the token limit: it was cut (or the model was looping)
            "truncated_output": int(new_tokens.shape[0]) >= max_tokens,
        }
        return self._processor.decode(new_tokens, skip_special_tokens=True)

    def _response(self, content: str) -> LLMResponse:
        return LLMResponse(content=content, model=self.model,
                           tokens_used=self.last_stats["prompt_tokens"] + self.last_stats["output_tokens"],
                           raw_response=self.last_stats)

    def generate(self, prompt: str, system_prompt: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> LLMResponse:
        return self._response(self._generate(self._messages(prompt, system_prompt), max_tokens=max_tokens,
                                             temperature=temperature))

    def chat(self, messages: list[LLMMessage], temperature: float = 0.0, max_tokens: int = 1024) -> LLMResponse:
        if self.vision:
            turns = [{"role": m.role, "content": [{"type": "text", "text": m.content}]} for m in messages]
        else:
            turns = [{"role": m.role, "content": m.content} for m in messages]
        return self._response(self._generate(turns, max_tokens=max_tokens, temperature=temperature))

    def generate_structured(self, prompt: str, schema: type[T], system_prompt: str | None = None,
                            images: list[bytes] | None = None, max_tokens: int = 2048) -> T:
        """Generate JSON for ``schema`` (optionally from page images) and validate it.

        ``max_tokens`` bounds the answer (a normal CV needs about 1,000 tokens): a model stuck
        repeating itself stops there instead of running for an hour on CPU.
        """
        messages = self._messages(structured_prompt(prompt, schema), system_prompt, len(images or []))
        answer = self._generate(messages, images, max_tokens=max_tokens)
        self.last_stats["raw_answer"] = answer  # kept for error analysis (what the model really wrote)
        if not answer.strip():
            raise ValueError("The model returned an empty answer.")
        return schema.model_validate(parse_json_object(answer))
