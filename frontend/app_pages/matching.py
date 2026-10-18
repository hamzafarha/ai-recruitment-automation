"""Matching (DSO3, diagram parts 5-6): rank every candidate for a job, then analyse the best in depth."""

import streamlit as st

from app_pages import common


def render() -> None:
    st.title("🎯 Matching")
    if not common.need_api():
        return
    jobs = [j for j in common.jobs() if j["status"] == "open"]
    if not jobs:
        st.info("No open job: create one in Jobs, or load the demo data from the Dashboard.")
        return
    st.caption("**1. Fast ranking** — every candidate graded in seconds with the fit scoring rubric (BGE-M3 "
               "retrieval of CV evidence, BGE reranker, skill lexicon; no LLM).  \n**2. Full AI analysis** — for "
               "the candidates you choose: the HR rules are retrieved (RAG), Qwen2.5-7B judges every requirement "
               "with a CV excerpt, the code checks each excerpt and computes the grade (about a minute each).")
    labels = {j["id"]: common.job_label(j) for j in jobs}
    job_id = st.selectbox("Job", list(labels), format_func=labels.get, key="matching_job")
    rankings = st.session_state.setdefault("rankings", {})
    if st.button("🏁 Rank all candidates", type="primary") or job_id not in rankings:
        with st.spinner("Scoring every candidate against every requirement…"):
            result = common.safe("GET", f"/jobs/{job_id}/ranking", timeout=600)
        if result is None:
            return
        rankings[job_id] = result["data"]
    rows = rankings.get(job_id) or []
    if not rows:
        st.info("No candidate yet: add CVs in Candidates, or load the demo data from the Dashboard.")
        return

    table = [{"#": i, "Candidate": r["name"] or "Unnamed", "Grade": f"{r['grade']} · {r['grade_label']}",
              "Score": r["score"], "Must-haves met": f"{r['met']}/{r['must_total']}", "Partial": r["partial"],
              "Missing": r["missing"], "Experience (y)": r["total_years"],
              "Main gaps": " · ".join(r["gaps"][:2]),
              "AI analysis": (f"grade {r['analysis_grade']} · {common.STATUS_LABELS.get(r['analysis_status'])}"
                              if r["analysis_id"] else "—")} for i, r in enumerate(rows, 1)]
    st.dataframe(table, hide_index=True, column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "Main gaps": st.column_config.TextColumn(width="large")})
    st.caption("Fast estimate: use it to choose whom to analyse, never to reject (a recruiter decides, KB-RR-001).")

    names = {r["candidate_id"]: f"{r['name'] or 'Unnamed'} (grade {r['grade']}, {r['score']:g})" for r in rows}
    default = [r["candidate_id"] for r in rows if r["grade"] >= 2][:3] or [rows[0]["candidate_id"]]
    chosen = st.multiselect("Candidates to analyse in depth", list(names), default=default, format_func=names.get,
                            key=f"chosen_{job_id}")
    c1, c2 = st.columns([2, 1])
    method = c2.radio("Method", ["llm", "semantic"], horizontal=True, key="analysis_method",
                      format_func={"llm": "Full AI analysis", "semantic": "Fast (stored as is)"}.get)
    if c1.button(f"🔬 Analyse {len(chosen)} candidate(s)", type="primary", disabled=not chosen):
        progress = st.progress(0.0, text="Starting…")
        done = []
        for i, candidate_id in enumerate(chosen):
            progress.progress(i / len(chosen), text=f"Analysing {names[candidate_id]} ({i + 1}/{len(chosen)})…")
            analysis = common.safe("POST", "/analyses", json={"candidate_id": candidate_id, "job_id": job_id,
                                                              "method": method}, timeout=common.LONG)
            if analysis:
                done.append(analysis)
        progress.progress(1.0, text="Done")
        common.refresh()
        rankings.pop(job_id, None)
        st.session_state["analysed"] = [a["id"] for a in done]
        if done:
            st.session_state["report_pick"] = done[0]["id"]
    for analysis_id in st.session_state.get("analysed") or []:
        analysis = common.safe("GET", f"/analyses/{analysis_id}")
        if analysis and analysis["job_id"] == job_id:
            report = analysis["report"]
            with st.container(border=True):
                st.markdown(f"**{analysis['candidate_name']}** — {common.grade_badge(report['grade'], report['grade_label'])}"
                            f" · score {report['score']:g}/100 · {report.get('action_label', '')}")
                st.write(report.get("summary") or "")
                for warning in report.get("warnings") or []:
                    st.warning(warning)
    if st.session_state.get("analysed"):
        st.page_link(common.PAGES["reports"], label="Open the full reports and download them", icon="📄")
        st.page_link(common.PAGES["review"], label="Validate them (human decision)", icon="✅")
