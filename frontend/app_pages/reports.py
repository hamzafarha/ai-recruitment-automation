"""Reports (diagram part 6): the evidence-backed structured report of every analysis, with downloads."""

import streamlit as st

from app_pages import common


def render() -> None:
    st.title("📄 Reports")
    if not common.need_api():
        return
    jobs = common.jobs()
    job_labels = {"": "All jobs"} | {j["id"]: common.job_label(j) for j in jobs}
    c1, c2 = st.columns(2)
    job_id = c1.selectbox("Job", list(job_labels), format_func=job_labels.get, key="reports_job")
    status = c2.selectbox("Status", ["", *common.STATUS_LABELS], key="reports_status",
                          format_func=lambda s: common.STATUS_LABELS.get(s, "All statuses"))
    params = {k: v for k, v in (("job_id", job_id), ("status", status)) if v}
    result = common.safe("GET", "/analyses", params=params)
    analyses = (result or {}).get("data") or []
    if not analyses:
        st.info("No analysis yet: Matching → choose a job → analyse candidates.")
        return
    st.dataframe([{"Candidate": a["candidate_name"], "Job": a["job_title"],
                   "Grade": a["final_grade"] if a["final_grade"] is not None else a["grade"], "Score": a["score"],
                   "Method": common.method_short(a["method"]),
                   "Status": common.STATUS_LABELS.get(a["status"], a["status"]),
                   "Date": a["created_at"][:16].replace("T", " ")} for a in analyses], hide_index=True)
    labels = {a["id"]: f"{a['candidate_name']} — {a['job_title']} (grade {a['grade']}, "
                       f"{common.method_short(a['method'])}, {a['created_at'][:10]})" for a in analyses}
    ids = list(labels)
    preferred = st.session_state.get("report_pick")
    pick = st.selectbox("Open a report", ids, format_func=labels.get, index=ids.index(preferred)
                        if preferred in ids else 0, key="report_select")
    st.session_state["report_pick"] = pick
    analysis = common.safe("GET", f"/analyses/{pick}")
    if not analysis:
        return
    st.divider()
    st.subheader(f"{analysis['candidate_name']} — {analysis['job_title']}")
    common.download_buttons(pick)
    common.show_report(analysis)
    if analysis.get("reviews"):
        st.markdown("#### Recruiter decisions")
        st.dataframe([{"Date": r["created_at"][:16].replace("T", " "), "Reviewer": r["reviewer"],
                       "Decision": r["decision"], "Adjusted grade": r["adjusted_grade"], "Interview": r["interview_at"],
                       "Notes": r["notes"]} for r in analysis["reviews"]], hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.page_link(common.PAGES["review"], label="Decide on this analysis (human validation)", icon="✅")
    with c2.popover("Delete this analysis"):
        if st.button("Delete", key=f"delete_analysis_{pick}"):
            if common.safe("DELETE", f"/analyses/{pick}"):
                common.refresh()
                st.session_state.pop("report_pick", None)
                st.rerun()
