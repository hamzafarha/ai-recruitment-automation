"""Candidate portal (diagram parts 1 -> 2 -> ... -> 9): a candidate applies; the application pipeline
(n8n workflow "Application pipeline") takes it from the CV to the final decision."""

import base64

import streamlit as st

from app_pages import common

CONSENT = ("I agree that my CV is processed to assess my application, as described in the candidate data policy "
           "(KB-HR-004): kept at most 24 months if not hired, deletion on request, a human reviews every decision.")


def render() -> None:
    st.title("📬 Candidate portal")
    st.caption("What a candidate sees. The application goes to **n8n**, whose *Application pipeline* workflow runs "
               "every step: intake checks, document processing, extraction by the fine-tuned Qwen, storage and "
               "search index, an answer to the candidate, then the screening agent, the report, the recruiters' "
               "notification, and it waits for the recruiter's decision before answering the candidate. Follow it on "
               "the Application pipeline page.")
    jobs = [j for j in common.jobs() if j["status"] == "open"]
    labels = {"": "No specific job (spontaneous application)"} | {j["id"]: common.job_label(j) for j in jobs}
    with st.form("application"):
        job_id = st.selectbox("Position", list(labels), format_func=labels.get)
        upload = st.file_uploader("Your CV (PDF, Word or image)", type=["pdf", "docx", "png", "jpg", "jpeg"])
        consent = st.checkbox(CONSENT)
        route = st.radio("Route (demo setting)", ["n8n", "api"], horizontal=True,
                         format_func={"n8n": "Through n8n (the workflow runs every step)",
                                      "api": "Directly to the API (it runs the same steps, if n8n is not running)"}.get)
        submitted = st.form_submit_button("Apply", type="primary")
    if not submitted:
        return
    if upload is None or not consent:
        st.error("Attach your CV and accept the data policy.")
        return
    payload = {"filename": upload.name, "content_base64": base64.b64encode(upload.getvalue()).decode(),
               "source": "portal", "consent": True, "job_id": job_id or None}
    with st.spinner("Sending your application (the CV is read by the AI: about a minute)…" if route == "n8n"
                    else "Sending your application…"):
        result = common.submit_application(payload, route)
    if result:
        common.refresh()
        st.success(result.get("message") or "Your application was received.")
        st.session_state["pipeline_pick"] = result["application_id"]
        st.caption(f"Recruiter view: application {common.short_id(result['application_id'])}"
                   + (f", candidate {result['full_name']} with {result.get('skills_count', '?')} skills extracted"
                      if result.get("full_name") else "")
                   + ". The screening agent and the next steps run now; the workflow then waits for a recruiter.")
        st.page_link(common.PAGES["pipeline"], label="Follow the application, step by step", icon="🔄")
