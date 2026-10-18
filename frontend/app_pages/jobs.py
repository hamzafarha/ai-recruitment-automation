"""Jobs (diagram parts 1, 6): job descriptions and the structured requirements candidates are scored on."""

import math

import pandas as pd
import streamlit as st

from app_pages import common

EMPTY_REQUIREMENTS = [{"requirement_text": "", "requirement_type": "required", "importance": "high",
                       "min_years_experience": None}]


def _requirements_table(requirements: list[dict]) -> None:
    st.dataframe([{"Id": r.get("id"), "Requirement": r.get("requirement_text"),
                   "Type": "must-have" if r.get("requirement_type") == "required" else "nice-to-have",
                   "Importance": r.get("importance"), "Min. years": r.get("min_years_experience")}
                  for r in requirements], hide_index=True,
                 column_config={"Requirement": st.column_config.TextColumn(width="large")})


def _editor(draft: dict, key: str) -> None:
    """Editable job (title, company... and the requirements table), then saved through the API."""
    c1, c2 = st.columns(2)
    title = c1.text_input("Job title", draft.get("title") or "", key=f"{key}_title")
    company = c2.text_input("Company", draft.get("company") or "", key=f"{key}_company")
    c1, c2, c3 = st.columns(3)
    location = c1.text_input("Location", draft.get("location") or "", key=f"{key}_location")
    employment = c2.text_input("Employment type", draft.get("employment_type") or "", key=f"{key}_type")
    role = c3.selectbox("Knowledge-base job profile (optional)",
                        ["", "ml_engineer", "nlp_llm_engineer", "backend_developer", "devops_cloud",
                         "data_analyst_scientist"], key=f"{key}_role",
                        help="Links the job to its profile in the HR knowledge base (accepted equivalences, "
                             "interview topics) used by the AI analysis.")
    st.markdown("**Requirements** — must-haves count for 85 % of the score; importance weights high 3, medium 2, "
                "low 1 (rubric KB-RR-002). Add or remove rows as needed.")
    frame = pd.DataFrame([{k: r.get(k) for k in ("requirement_text", "requirement_type", "importance",
                                                  "min_years_experience")}
                          for r in draft.get("requirements") or EMPTY_REQUIREMENTS])
    edited = st.data_editor(frame, num_rows="dynamic", hide_index=True, key=f"{key}_requirements",
                            column_config={
                                "requirement_text": st.column_config.TextColumn("Requirement", width="large",
                                                                                required=True),
                                "requirement_type": st.column_config.SelectboxColumn(
                                    "Type", options=["required", "preferred"], required=True, default="required"),
                                "importance": st.column_config.SelectboxColumn(
                                    "Importance", options=["high", "medium", "low"], required=True, default="medium"),
                                "min_years_experience": st.column_config.NumberColumn("Min. years", min_value=0,
                                                                                      max_value=40, step=0.5),
                            })
    description = st.text_area("Description", draft.get("description") or "", height=150, key=f"{key}_description")
    if st.button("💾 Save the job", type="primary", key=f"{key}_save"):
        requirements = []
        for row in edited.to_dict("records"):
            text = (row.get("requirement_text") or "").strip() if isinstance(row.get("requirement_text"), str) else ""
            if not text:
                continue
            years = row.get("min_years_experience")
            kind, importance = row.get("requirement_type"), row.get("importance")  # empty cells come back as NaN
            requirements.append({
                "requirement_text": text,
                "requirement_type": kind if kind in ("required", "preferred") else "required",
                "importance": importance if importance in ("high", "medium", "low") else "medium",
                "min_years_experience": None if years is None or (isinstance(years, float) and math.isnan(years))
                else float(years)})
        body = {"title": title.strip(), "company": company or None, "location": location or None,
                "employment_type": employment or None, "role": role or None, "description": description,
                "requirements": requirements}
        created = common.safe("POST", "/jobs", json=body, timeout=120)
        if created:
            common.refresh()
            st.session_state.pop(f"{key}_draft", None)
            st.success(f"Saved: {created['title']} ({created['must_haves']} must-haves, {created['nice_to_haves']} "
                       "nice-to-haves), indexed for search. Next: Matching → rank the candidates.")


def render() -> None:
    st.title("💼 Jobs")
    if not common.need_api():
        return
    browse, from_ad, manual = st.tabs(["📋 All jobs", "✨ New job from an advertisement (AI)", "✍️ New job (manual)"])

    with browse:
        jobs = common.jobs()
        if not jobs:
            st.info("No job yet: create one, or load the demo data from the Dashboard.")
        else:
            st.dataframe([{"Title": j["title"], "Company": j["company"], "Location": j["location"],
                           "Must-haves": j["must_haves"], "Nice-to-haves": j["nice_to_haves"], "Status": j["status"],
                           "Source": j["source"], "Added": j["created_at"][:10]} for j in jobs], hide_index=True)
            labels = {j["id"]: common.job_label(j) for j in jobs}
            pick = st.selectbox("Open a job", list(labels), format_func=labels.get, key="job_pick")
            job = common.safe("GET", f"/jobs/{pick}")
            if job:
                st.subheader(job["title"])
                st.caption(" · ".join(x for x in (job.get("company"), job.get("location"), job.get("employment_type"),
                                                  f"profile {job['role']}" if job.get("role") else None) if x))
                _requirements_table(job["requirements"])
                with st.expander("Description"):
                    st.write(job.get("description") or "—")
                c1, c2 = st.columns(2)
                new_status = "closed" if job["status"] == "open" else "open"
                if c1.button(f"Mark as {new_status}", key=f"status_{pick}"):
                    if common.safe("PATCH", f"/jobs/{pick}", json={"status": new_status}):
                        common.refresh()
                        st.rerun()
                with c2.popover("Delete this job"):
                    st.write("Deletes the job, its analyses and its search entries.")
                    if st.button("Delete", key=f"delete_job_{pick}"):
                        if common.safe("DELETE", f"/jobs/{pick}"):
                            common.refresh()
                            st.rerun()

    with from_ad:
        st.caption("Paste a job advertisement (English or French): Qwen2.5-7B turns it into checkable requirements "
                   "(must-have / nice-to-have, importance, minimum years). Review and edit them, then save.")
        text = st.text_area("Job advertisement", height=260, key="job_ad")
        if st.button("✨ Extract the requirements", disabled=len(text.strip()) < 80):
            with st.spinner("Reading the advertisement (about 30 s on the Kaggle GPU)…"):
                draft = common.safe("POST", "/jobs/extract", json={"text": text}, timeout=600)
            if draft:
                st.session_state["ad_draft"] = draft
                for suffix in ("title", "company", "location", "type", "role", "requirements", "description"):
                    st.session_state.pop(f"ad_{suffix}", None)  # show the new draft, not the previous edits
        if st.session_state.get("ad_draft"):
            _editor(st.session_state["ad_draft"], "ad")

    with manual:
        _editor({"requirements": EMPTY_REQUIREMENTS}, "manual")
