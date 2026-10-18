"""Under the hood, parts 3-4: a CV through document processing, step by step, and the fine-tuned
adapter compared with the base model on that CV."""

import json

import streamlit as st

from app_pages import common


def _raw(answer: str | None) -> str:
    if not answer:
        return "(not available)"
    try:
        return json.dumps(json.loads(answer[answer.find("{"):]), indent=2, ensure_ascii=False)
    except ValueError:  # includes JSONDecodeError: show the answer as written
        return answer


def _run_caption(run: dict) -> str:
    tokens = (f", {run['prompt_tokens']} prompt tokens → {run['output_tokens']} answer tokens"
              if run.get("prompt_tokens") else "")
    return f"{run.get('model') or 'model'} · adapter {'ON' if run['adapter_used'] else 'OFF'} · {run['seconds']} s{tokens}"


def _comparison(result: dict) -> None:
    ft, base, comparison = result["finetuned"], result["base"], result["comparison"]
    st.subheader("Base model vs fine-tuned adapter, on this CV")
    st.caption("Same text, same prompt: only the LoRA adapter is switched on or off on the Kaggle GPU.")
    st.dataframe([{"Section": s, "Base Qwen2.5-7B": c["base"], "Fine-tuned": c["finetuned"],
                   "Difference": c["finetuned"] - c["base"]} for s, c in comparison["counts"].items()],
                 hide_index=True)
    st.dataframe([{"Field": f, "Base": v["base"], "Fine-tuned": v["finetuned"],
                   "Same": "yes" if v["base"] == v["finetuned"] else "no"} for f, v in comparison["identity"].items()],
                 hide_index=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Skills found only by the fine-tuned model**")
    c1.write(", ".join(comparison["skills_only_finetuned"]) or "—")
    c2.markdown("**Skills found only by the base model**")
    c2.write(", ".join(comparison["skills_only_base"]) or "—")
    c1, c2 = st.columns(2)
    with c1:
        st.caption(_run_caption(ft))
        with st.expander("Fine-tuned: raw answer"):
            st.code(_raw(ft["raw_answer"]), language="json")
    with c2:
        st.caption(_run_caption(base))
        with st.expander("Base: raw answer"):
            st.code(_raw(base["raw_answer"]), language="json")
    with st.expander("Base model: the profile it produced"):
        common.show_profile(base["profile"])


def render() -> None:
    st.title("📄 Document processing, step by step")
    st.caption("Parts 3 and 4 of the architecture: how a CV becomes a structured profile. The file is read on this "
               "PC (text layer, Word, or OCR for scans); the text goes to Qwen2.5-7B with our fine-tuned QLoRA adapter "
               "on the Kaggle GPU; the answer is checked and normalised. Nothing is stored on this page.")
    if not common.need_api():
        return
    upload = st.file_uploader("A CV (PDF, Word or image)", type=["pdf", "docx", "png", "jpg", "jpeg"],
                              key="observe_cv")
    compare = st.checkbox("Also run the base model without the fine-tuned adapter, to compare (about twice as long)",
                          value=True)
    if upload and st.button("▶ Process this CV step by step", type="primary"):
        with st.spinner("Reading the document, then Qwen extracts the profile (1–3 minutes)…"):
            result = common.safe("POST", "/observe/extraction", timeout=common.LONG,
                                 files={"file": (upload.name, upload.getvalue(), upload.type)},
                                 data={"compare_base": "true" if compare else "false"})
        if result:
            st.session_state["observe_extraction"] = result
    result = st.session_state.get("observe_extraction")
    if not result:
        st.info("Upload a CV (your own or an invented one) and press the button.")
        return
    parsing = result["parsing"]
    st.subheader("1. Reading the document")
    c = st.columns(4)
    c[0].metric("File type", parsing["file_type"])
    c[1].metric("Reader", parsing.get("engine") or "—")
    c[2].metric("Characters", parsing["characters"])
    c[3].metric("Seconds", parsing["seconds"])
    if parsing["pages"]:
        st.dataframe([{"Page": p["page"], "How it was read": "OCR (scan)" if p["method"] == "ocr" else p["method"],
                       "Characters": p["characters"], "OCR confidence": p["ocr_confidence"]}
                      for p in parsing["pages"]], hide_index=True)
    st.caption("A page with almost no text layer is treated as a scan and read by OCR (PaddleOCR models).")
    with st.expander("The extracted text (what Qwen reads)"):
        st.text(result["text"])

    st.subheader("2. The prompt sent to Qwen")
    st.caption(f"{result['prompt']['user_chars']} characters"
               + (" — the CV was cut to fit the model's input limit" if result["prompt"]["cv_truncated"] else "")
               + ". The same prompt was used to fine-tune the model.")
    with st.expander("System prompt (the extraction rules)"):
        st.code(result["prompt"]["system"], language="text", wrap_lines=True)
    with st.expander("User message (the CV + the JSON schema to fill)"):
        st.code(result["prompt"]["user"], language="text", wrap_lines=True)

    ft = result["finetuned"]
    st.subheader("3. Qwen's answer (fine-tuned adapter)")
    st.caption(_run_caption(ft))
    with st.expander("Raw answer, exactly as the model wrote it"):
        st.code(_raw(ft["raw_answer"]), language="json")

    st.subheader("4. The checked, normalised profile")
    st.caption("Dates normalised, duplicates removed, skills mapped to the EN/FR skill lexicon, project technologies "
               "added to the skills, group headings removed.")
    common.show_profile(ft["profile"])
    if result.get("comparison"):
        st.divider()
        _comparison(result)
