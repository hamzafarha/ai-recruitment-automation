"""Candidates (diagram parts 1, 3, 8): a CV goes into the application pipeline; the stored profiles."""

import base64

import streamlit as st

from app_pages import common

TYPES = ["pdf", "docx", "png", "jpg", "jpeg"]


def _detail(candidate_id: str) -> None:
    data = common.safe("GET", f"/candidates/{candidate_id}")
    if not data:
        return
    common.show_profile(data["profile"])
    extraction = data.get("extraction") or {}
    st.caption(f"Source: {data['source']} · {common.extraction_caption(extraction, data.get('cv_filename'))}"
               + ("" if data.get("indexed", True) else " · ⚠ not indexed for search: Dashboard → Re-index"))
    if data.get("analyses"):
        st.markdown("#### Analyses")
        st.dataframe([{"Job": a["job_title"], "Grade": a["final_grade"] if a["final_grade"] is not None else a["grade"],
                       "Score": a["score"], "Method": common.method_short(a["method"]),
                       "Status": common.STATUS_LABELS.get(a["status"], a["status"]), "Date": a["created_at"][:10]}
                      for a in data["analyses"]], hide_index=True)
    with st.expander("CV text (what the AI analysis reads and quotes)"):
        text = common.safe("GET", f"/candidates/{candidate_id}/cv-text")
        st.text((text or {}).get("cv_text") or "No CV text stored.")
    with st.expander("Raw profile (JSON)"):
        st.json(data["profile"])
    with st.expander("Delete this candidate (right to erasure, KB-HR-004)"):
        sure = st.checkbox("I confirm: delete the candidate, their analyses and their search entries",
                           key=f"confirm_{candidate_id}")
        if st.button("Delete", disabled=not sure, key=f"delete_{candidate_id}"):
            if common.safe("DELETE", f"/candidates/{candidate_id}"):
                common.refresh()
                st.session_state.pop("candidate_pick", None)
                st.success("Deleted.")
                st.rerun()


def render() -> None:
    st.title("👤 Candidates")
    if not common.need_api():
        return
    add, browse, test = st.tabs(["➕ Add a candidate", "📋 All candidates", "🧪 Test the extraction only"])

    with add:
        st.caption("The CV goes into the application pipeline, the same workflow as a candidate's application: "
                   "document processing (PDF, Word, OCR for scans), extraction by Qwen2.5-7B with our fine-tuned "
                   "QLoRA adapter (Kaggle GPU), storage in PostgreSQL and the search index (BGE-M3 → Qdrant), then "
                   "the screening agent analyses it for the job (or finds the open jobs that fit), and the analysis "
                   "waits for your decision in Human validation. The file itself is not kept. Only use CVs you are "
                   "allowed to process.")
        jobs = [j for j in common.jobs() if j["status"] == "open"]
        labels = {"": "No specific job (the agent finds the open jobs that fit)"} | {j["id"]: common.job_label(j)
                                                                                     for j in jobs}
        c1, c2 = st.columns([2, 1])
        job_id = c1.selectbox("For the job", list(labels), format_func=labels.get, key="cv_upload_job")
        route = c2.radio("Run by", ["n8n", "api"], key="cv_upload_route",
                         format_func={"n8n": "n8n (one execution)", "api": "the API (if n8n is stopped)"}.get)
        upload = st.file_uploader("CV file", type=TYPES, key="cv_upload")
        if upload and st.button("Submit the CV to the pipeline", type="primary"):
            payload = {"filename": upload.name, "content_base64": base64.b64encode(upload.getvalue()).decode(),
                       "source": "upload", "consent": True, "job_id": job_id or None}
            with st.spinner("Intake, document processing, extraction by the fine-tuned Qwen, storage (about a "
                            "minute)…" if route == "n8n" else "Registering the application…"):
                result = common.submit_application(payload, route)
            if result:
                common.refresh()
                st.session_state["pipeline_pick"] = result["application_id"]
                if result.get("candidate_id"):
                    st.session_state["last_created"] = result["candidate_id"]
                st.success((f"{result['full_name']}: {result.get('skills_count', '?')} skills extracted. "
                            if result.get("full_name") else "Received. ")
                           + "The screening agent analyses the application now; the workflow then waits for your "
                             "decision in Human validation.")
                st.page_link(common.PAGES["pipeline"], label="Follow it on the Application pipeline page", icon="🔄")
        if st.session_state.get("last_created"):
            _detail(st.session_state["last_created"])

    with browse:
        candidates = common.candidates()
        if not candidates:
            st.info("No candidate yet: add a CV, or load the demo data from the Dashboard.")
        else:
            needle = st.text_input("Filter (name, title, location)", key="candidate_filter").strip().lower()
            shown = [c for c in candidates if not needle or needle in common.candidate_label(c).lower()
                     or needle in (c.get("location") or "").lower()]
            st.dataframe([{"Name": c["full_name"], "Latest position": c["headline"], "Location": c["location"],
                           "Positions": c["experience_count"], "Skills": c["skills_count"], "Source": c["source"],
                           "Indexed": "yes" if c["indexed"] else "no", "Added": c["created_at"][:10]} for c in shown],
                         hide_index=True)
            labels = {c["id"]: common.candidate_label(c) for c in shown}
            if labels:
                pick = st.selectbox("Open a candidate", list(labels), format_func=labels.get, key="candidate_pick")
                st.divider()
                _detail(pick)

    with test:
        st.caption("Extraction only (DSO1): nothing is stored. Shows exactly what the model extracted.")
        upload = st.file_uploader("CV file", type=TYPES, key="cv_test")
        if upload and st.button("Extract the profile", key="extract_only"):
            with st.spinner("Extracting…"):
                result = common.safe("POST", "/candidates/extract", timeout=common.LONG,
                                     files={"file": (upload.name, upload.getvalue(), upload.type)})
            if result:
                st.session_state["extraction_test"] = result
        result = st.session_state.get("extraction_test")
        if result:
            if result["llm_provider"] == "mock":
                st.warning("LLM_PROVIDER=mock: fake answers. Start the Kaggle AI server and set LLM_PROVIDER=remote.")
            common.show_profile(result["profile"])
            st.caption(common.extraction_caption(result["stats"], result["filename"]))
            with st.expander("Raw JSON"):
                st.json(result)
