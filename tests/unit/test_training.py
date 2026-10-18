"""Tests for Step 9 (fine-tuning): training data, prompt parity, tokenization, Kaggle notebook.

No GPU, model or download needed.
"""

import ast
import importlib.util
import json
from pathlib import Path

from recruitment_ai.extraction.extractor import SYSTEM_PROMPT, CVExtraction, CVExtractor
from recruitment_ai.extraction.schemas import CandidateProfile, Experience, Skill
from recruitment_ai.training.data import (
    IGNORE_INDEX,
    _short,
    build_training_set,
    chat_example,
    profile_to_extraction,
    tokenize_example,
)

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "kaggle" / "finetune_qwen_qlora.ipynb"


def _row(resume_id: str, text: str, name: str = "Ann Lee", title: str = "Data Engineer") -> dict:
    """A train-split row as iter_split yields it."""
    target = {"personal_information": {"full_name": name},
              "work_experience": [{"job_title": title, "company": "Acme"}]}
    return {"resume_id": resume_id, "category": "Data Science", "resume_text": text, "target": target}


# --------------------------------------------------------------------------- answers


def test_profile_to_extraction_keeps_the_answer_short_and_marks_current_jobs():
    profile = CandidateProfile(
        full_name="Ann Lee",
        experience=[Experience(title="Data Engineer", company="Acme", start_date="2021-01", end_date="present",
                               description="Builds pipelines.\n- Spark on AWS\n- Airflow")],
        skills=[Skill(name=f"S{i}") for i in range(70)],
    )
    answer = profile_to_extraction(profile)
    job = answer.experience[0]
    assert job.end_date == "Present" and job.start_date == "2021-01"
    assert job.description == "Builds pipelines."  # first line only, not the bullet list
    assert len(answer.skills) == 60  # the schema's limit


def test_short_cuts_long_text_at_a_sentence_end():
    text = "First sentence is here. " + "word " * 100
    assert _short(text, 60) == "First sentence is here."
    assert _short("- bullet only", 50) == "bullet only" and _short(None, 10) is None


def test_training_prompt_is_exactly_the_inference_prompt(monkeypatch):
    """The model must be trained on the very messages the extractor sends at inference."""
    from recruitment_ai.llm.huggingface_adapter import HuggingFaceAdapter

    sent = {}

    def fake_generate(self, messages, images=None, max_tokens=1024, temperature=0.0, use_adapter=True):
        sent["messages"] = messages
        self.last_stats = {"prompt_tokens": 1, "output_tokens": 1}
        return '{"full_name": "Ann Lee"}'

    monkeypatch.setattr(HuggingFaceAdapter, "_generate", fake_generate)
    cv = "Ann Lee\nData Engineer at Acme (2021 - Present)"
    CVExtractor(llm=HuggingFaceAdapter(model="Qwen/Qwen2.5-7B-Instruct"), normalizer=object()).extract_text(cv)
    example = chat_example(cv, CVExtraction(full_name="Ann Lee"))
    assert sent["messages"] == example[:2]
    assert example[0]["content"] == SYSTEM_PROMPT and example[2]["role"] == "assistant"
    assert CVExtraction.model_validate_json(example[2]["content"]).full_name == "Ann Lee"


# --------------------------------------------------------------------------- selection


def test_build_training_set_removes_eval_cvs_and_bad_labels():
    good = [_row(f"r{i}", f"Ann Lee {i} Data Engineer at Acme since 2021") for i in range(6)]
    rows = good + [
        _row("leak", "ANN LEE 0 — data engineer at Acme since 2021!"),   # same text as good[0], other id
        _row("copy", good[1]["resume_text"]),                             # repeated inside train
        _row("bad", "Bob Smith, accountant", name="Ann Lee"),             # labels not in the text
        _row("glued", "Ann LeexClairey Data Engineer at Acme"),           # corrupted template text
        _row("long", "Ann Lee Data Engineer at Acme " + "x" * 17000),     # longer than the prompt keeps
    ]
    eval_texts = ["Ann Lee 5 Data Engineer at Acme since 2021"]  # good[5] is in the evaluation splits
    train, holdout, report = build_training_set(rows, eval_texts, n_train=3, n_holdout=1, seed=1)
    ids = {e["id"] for e in train + holdout}
    assert len(train) == 3 and len(holdout) == 1 and not ids & {"r5", "copy", "bad", "glued", "long"}
    assert report["removed_in_eval_splits"] == 1 and report["removed_poorly_grounded"] == 1
    assert report["removed_corrupted"] == 1 and report["removed_too_long"] == 1
    assert report["removed_repeated_text"] == 2  # "leak" (same text as r0) and "copy"
    assert "reference" in holdout[0] and "text" in holdout[0] and "reference" not in train[0]
    again, _, _ = build_training_set(rows, eval_texts, n_train=3, n_holdout=1, seed=1)
    assert [e["id"] for e in again] == [e["id"] for e in train]  # reproducible


def test_read_jsonl_keeps_unicode_line_separators_inside_strings(tmp_path):
    from recruitment_ai.training.data import read_jsonl

    rows = [{"text": "first part second part\x85end"}, {"text": "next CV"}]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert read_jsonl(path) == rows  # str.splitlines() would cut the first row in three


# --------------------------------------------------------------------------- tokenization


class CharTokenizer:
    """One token per character; the chat template just joins the contents."""

    eos_token = "|"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in messages) + ">"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_tokenize_example_trains_on_the_answer_only():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                {"role": "assistant", "content": "{}"}]
    tokens = tokenize_example(CharTokenizer(), messages, max_len=100)
    assert tokens["input_ids"] == [ord(c) for c in "SU>{}|"]
    assert tokens["labels"] == [IGNORE_INDEX] * 3 + [ord(c) for c in "{}|"]  # prompt masked, answer + end
    assert tokenize_example(CharTokenizer(), messages, max_len=5) is None  # too long: skipped, not cut


def test_collate_keeps_only_the_answer_logits():
    from recruitment_ai.training.qlora import collate_examples

    example = {"input_ids": [5, 6, 7, 8, 9], "attention_mask": [1] * 5,
               "labels": [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 8, 9]}
    out = collate_examples([example], pad_id=0)
    # the last 3 positions: the prompt's last token (predicts 8) and the 2 answer tokens
    assert out["logits_to_keep"] == 3
    assert out["labels"].tolist() == [[IGNORE_INDEX, 8, 9]]
    assert out["input_ids"].tolist() == [[5, 6, 7, 8, 9]]  # the model still reads the whole text
    two = collate_examples([example, {**example, "input_ids": [5, 6], "attention_mask": [1, 1],
                                      "labels": [IGNORE_INDEX, 6]}], pad_id=0)
    assert "logits_to_keep" not in two and two["labels"].shape == (2, 5)


# --------------------------------------------------------------------------- Kaggle notebook and jobs


def test_every_notebook_cell_parses_and_ends_saved_and_exported():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert "parameters" in notebook["cells"][1]["metadata"]["tags"]
    for i, source in enumerate(code_cells, 1):
        ast.parse(source)
        last_line = source.rstrip().splitlines()[-1]
        assert "saved and exported" in last_line or last_line.startswith("exported(") or (
            "exported(" in source.rstrip().splitlines()[-3]), f"cell {i} does not end with its export line"


def _kaggle_run():
    spec = importlib.util.spec_from_file_location("kaggle_run", ROOT / "scripts" / "kaggle_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kaggle_jobs_get_their_constants():
    kaggle_run = _kaggle_run()
    code = 'PACKAGE_ID = "local"  # set by kaggle_run\nENGINE = "ocr_llm"\nADAPTER = ""\n'
    out = kaggle_run.set_constants(code, {"PACKAGE_ID": "20261016-1", "ADAPTER": "auto"})
    assert 'PACKAGE_ID = "20261016-1"' in out and 'ADAPTER = "auto"' in out and 'ENGINE = "ocr_llm"' in out
    notebook_code = "".join(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][1]["source"])
    assert 'PACKAGE_ID = "x"' in kaggle_run.set_constants(notebook_code, {"PACKAGE_ID": "x"})
    assert kaggle_run.JOBS["extraction-ft"]["inputs"] == ["finetune"]
    assert all(spec["code"].exists() for spec in kaggle_run.JOBS.values())
