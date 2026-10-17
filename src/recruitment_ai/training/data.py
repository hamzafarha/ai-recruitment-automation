"""Fine-tuning data (Step 9): Resume Parser train split -> chat examples for CV extraction.

Each example is exactly what the extractor sends at inference time (same system prompt, same
user prompt with the JSON schema), followed by the expected answer built from the dataset's
labels. The labels were written by an LLM (dataset authors): they are "silver" references.

Leakage control: the evaluation set comes from the dataset's validation and test splits, and
some CVs appear in several splits under different ids. Every training CV whose normalized text
matches a validation/test CV is removed, so the model never trains on an evaluation CV.
"""

import random
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from recruitment_ai.data.normalization import PRESENT, text_hash
from recruitment_ai.data.resume_parser import convert_label, groundedness
from recruitment_ai.extraction.extractor import (
    MAX_INPUT_CHARS,
    SYSTEM_PROMPT,
    CVExtraction,
    ExtractedEducation,
    ExtractedJob,
    ExtractedLanguage,
    text_prompt,
)
from recruitment_ai.extraction.schemas import CandidateProfile
from recruitment_ai.llm.huggingface_adapter import structured_prompt

IGNORE_INDEX = -100  # label of the prompt tokens: the loss is computed on the answer only
_CORRUPTED = re.compile(r"[a-z]Claire[a-z]")  # template name glued into words (data inspection)


def _short(text: str | None, max_chars: int) -> str | None:
    """First line of a text, cut at a sentence end within ``max_chars`` (the prompt asks for
    short descriptions; the dataset's are full bullet lists)."""
    if not text:
        return None
    first = next((line.strip(" -•*\t") for line in text.splitlines() if line.strip(" -•*\t")), "")
    if len(first) > max_chars:
        cut = first[:max_chars]
        end = max(cut.rfind(". "), cut.rfind("; "))
        first = cut[: end + 1] if end > max_chars // 3 else cut.rsplit(" ", 1)[0]
    return first or None


def profile_to_extraction(profile: CandidateProfile) -> CVExtraction:
    """The answer the model should give for a CV, from its reference profile."""
    jobs = [
        ExtractedJob(
            title=e.title, company=e.company, location=e.location, start_date=e.start_date,
            end_date="Present" if e.end_date == PRESENT or (e.is_current and not e.end_date) else e.end_date,
            description=_short(e.description, 200),
        )
        for e in profile.experience
    ]
    education = [
        ExtractedEducation(degree=e.degree, field_of_study=e.field_of_study, institution=e.institution,
                           start_date=e.start_date, end_date=e.end_date)
        for e in profile.education
    ]
    return CVExtraction(  # lists longer than the schema's limits are trimmed by CVExtraction itself
        full_name=profile.full_name, email=str(profile.email) if profile.email else None, phone=profile.phone,
        location=profile.location, summary=_short(profile.summary, 300),
        experience=jobs, education=education, skills=[s.name for s in profile.skills],
        certifications=[c.name for c in profile.certifications],
        languages=[ExtractedLanguage(language=x.language, proficiency=x.proficiency) for x in profile.languages],
    )


def chat_example(cv_text: str, answer: CVExtraction) -> list[dict[str, str]]:
    """System + user messages exactly as sent at inference, then the expected JSON answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": structured_prompt(text_prompt(cv_text), CVExtraction)},
        {"role": "assistant", "content": answer.model_dump_json()},
    ]


def build_training_set(
    train_rows: Iterable[dict[str, Any]], exclude_texts: Iterable[str], n_train: int, n_holdout: int,
    seed: int = 42, min_groundedness: float = 0.8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Pick training and hold-out examples from the train split (rows from ``iter_split``).

    Removed first: CVs also present in the validation/test splits (any id), repeated texts,
    texts longer than the inference prompt keeps, corrupted texts, and CVs whose labels are
    poorly grounded in the text (values not found in the CV are probably invented by the
    labelling LLM: training on them would teach the model to invent). The choice is random
    but reproducible (``seed``).
    """
    excluded = {text_hash(t) for t in exclude_texts}
    stats: Counter[str] = Counter()
    seen: set[str] = set()
    candidates = []
    for row in train_rows:
        stats["train_rows"] += 1
        text, digest = row["resume_text"], text_hash(row["resume_text"])
        if digest in excluded:
            stats["removed_in_eval_splits"] += 1
        elif digest in seen:
            stats["removed_repeated_text"] += 1
        elif len(text.strip()) > MAX_INPUT_CHARS:
            stats["removed_too_long"] += 1
        elif _CORRUPTED.search(text):
            stats["removed_corrupted"] += 1
        elif groundedness(text, row["target"]) < min_groundedness:
            stats["removed_poorly_grounded"] += 1
        else:
            candidates.append(row)
        seen.add(digest)

    random.Random(seed).shuffle(candidates)
    picked = candidates[: n_holdout + n_train]

    def example(row: dict[str, Any], keep_reference: bool) -> dict[str, Any]:
        profile = convert_label(row["target"], resume_id=row["resume_id"])
        out = {"id": row["resume_id"], "category": row.get("category"),
               "messages": chat_example(row["resume_text"], profile_to_extraction(profile))}
        if keep_reference:  # the hold-out check extracts the CV again and scores it against this
            out.update(text=row["resume_text"], reference=profile.model_dump(mode="json", exclude={"created_at"}))
        return out

    holdout = [example(r, keep_reference=True) for r in picked[:n_holdout]]
    train = [example(r, keep_reference=False) for r in picked[n_holdout:]]
    report = {
        **dict(stats), "candidates": len(candidates), "train": len(train), "holdout": len(holdout),
        "train_categories": dict(Counter(e["category"] for e in train).most_common()),
        "labels": "silver (written by an LLM by the dataset authors)",
    }
    return train, holdout, report


def tokenize_example(tokenizer: Any, messages: list[dict[str, str]], max_len: int) -> dict[str, list[int]] | None:
    """Token ids of a chat example, with the loss restricted to the answer.

    Returns None when the example is longer than ``max_len``: it is skipped, not cut (a cut
    answer would teach the model to stop in the middle of the JSON).
    """
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    # the answer ends with the end-of-turn token, so the model learns where to stop
    answer_ids = tokenizer(messages[-1]["content"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + answer_ids
    if len(ids) > max_len:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [IGNORE_INDEX] * len(prompt_ids) + answer_ids}
