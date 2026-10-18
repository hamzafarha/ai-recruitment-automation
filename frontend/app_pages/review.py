"""Human validation (diagram part 9): the recruiter shortlists, schedules, holds or rejects."""

import datetime as dt

import streamlit as st

from app_pages import common

DECISIONS = {"shortlist": "✅ Shortlist", "interview": "📅 Invite to an interview", "hold": "⏸ Put on hold",
             "reject": "✖ Reject (a written reason is required)"}


def _decision_form(analysis: dict) -> None:
    report = analysis["report"]
    with st.form(f"review_{analysis['id']}"):
        st.markdown("#### Your decision")
        c1, c2 = st.columns(2)
        reviewer = c1.text_input("Reviewer", st.session_state.get("reviewer", "recruiter"))
        decision = c2.radio("Decision", list(DECISIONS), format_func=DECISIONS.get, index=None)
        c1, c2, c3 = st.columns(3)
        grade_options = ["keep", 0, 1, 2, 3]
        adjusted = c1.selectbox("Grade", grade_options, format_func=lambda g: f"keep {report['grade']} (AI)"
                                if g == "keep" else f"set to {g} · {common.GRADE_LABELS[g]}")
        day = c2.date_input("Interview date (for an interview)", dt.date.today() + dt.timedelta(days=3))
        hour = c3.time_input("Interview time", dt.time(10, 0))
        notes = st.text_area("Notes / reason (required to reject; shared with the n8n notification)")
        if st.form_submit_button("Save the decision", type="primary"):
            if decision is None:
                st.error("Choose a decision first.")
                return
            st.session_state["reviewer"] = reviewer
            body = {"decision": decision, "reviewer": reviewer or "recruiter", "notes": notes or None,
                    "adjusted_grade": None if adjusted == "keep" else adjusted,
                    "interview_at": f"{day:%Y-%m-%d} {hour:%H:%M}" if decision == "interview" else None}
            saved = common.safe("POST", f"/analyses/{analysis['id']}/review", json=body, timeout=120)
            if saved:
                common.refresh()
                application = saved.get("application")
                if application:  # the decision resumed the application's workflow
                    after = {"closed": "it is closed: the candidate's answer is prepared (Notifications)",
                             "waiting_review": "it still waits (on hold, or another analysis of this application "
                                               "has no decision yet)",
                             "running": "n8n runs its last steps: final decision → message to the candidate → "
                                        "closed"}.get(application["status"], application["status"])
                    message = f"{DECISIONS[decision]} saved for {analysis['candidate_name']}. Application workflow: {after}."
                    st.session_state["pipeline_pick"] = application["id"]
                else:
                    message = (f"{DECISIONS[decision]} saved for {analysis['candidate_name']}. A notification was "
                               "created (and sent to n8n when enabled).")
                st.session_state["review_done"] = message
                st.rerun()


def render() -> None:
    st.title("✅ Human validation")
    st.caption("The AI proposes, a recruiter decides (KB-RR-001): every analysis waits here until a recruiter "
               "shortlists, invites, holds or rejects the candidate. A rejection needs a written reason; the grade "
               "can be adjusted. Each decision is stored and becomes a notification (n8n prepares the message).")
    if not common.need_api():
        return
    if message := st.session_state.pop("review_done", None):
        st.success(message)
        if "Application workflow" in message:
            st.page_link(common.PAGES["pipeline"], label="See the application's workflow", icon="🔄")
    waiting, decided = st.tabs(["⏳ Waiting for a decision", "📚 Decisions made"])
    with waiting:
        pending = ((common.safe("GET", "/analyses", params={"status": "pending_review"}) or {}).get("data") or [])
        if not pending:
            st.info("Nothing waits for a decision. Analyse candidates in Matching first.")
        else:
            labels = {a["id"]: f"{a['candidate_name']} — {a['job_title']} · grade {a['grade']} · {a['score']:g}/100 · "
                               f"{common.method_short(a['method'])}" for a in pending}
            ids = list(labels)
            preferred = st.session_state.get("report_pick")
            pick = st.selectbox(f"{len(pending)} analyses waiting", ids, format_func=labels.get,
                                index=ids.index(preferred) if preferred in ids else 0)
            analysis = common.safe("GET", f"/analyses/{pick}")
            if analysis:
                report = analysis["report"]
                st.markdown(f"### {analysis['candidate_name']} — {analysis['job_title']}")
                if analysis.get("application"):
                    st.caption("From the application pipeline: your decision resumes this application's workflow "
                               "(final decision → message to the candidate → closed).")
                st.markdown(f"{common.grade_badge(report['grade'], report['grade_label'])} · score "
                            f"**{report['score']:g}/100** · {report.get('action_label', '')}")
                st.info(report.get("summary") or "—")
                must = [r for r in report.get("requirements") or [] if r.get("kind") == "must"]
                st.markdown(" · ".join(f"{common.REQ_STATUS[s]}: {sum(r['status'] == s for r in must)}"
                                       for s in ("met", "partial", "missing")) + " (must-haves)")
                c1, c2 = st.columns(2)
                c1.markdown("**Gaps to explore**\n" + "\n".join(f"- {g}" for g in report.get("gaps") or ["—"]))
                c2.markdown("**Interview questions**\n" + "\n".join(f"- {q}" for q in
                                                                     report.get("interview_questions") or ["—"]))
                with st.expander("Full report"):
                    common.show_report(analysis)
                _decision_form(analysis)
    with decided:
        result = common.safe("GET", "/analyses") or {}
        done = [a for a in result.get("data") or [] if a["status"] != "pending_review"]
        if not done:
            st.caption("No decision yet.")
        else:
            st.dataframe([{"Candidate": a["candidate_name"], "Job": a["job_title"],
                           "Decision": common.STATUS_LABELS.get(a["status"], a["status"]),
                           "AI grade": a["grade"], "Final grade": a["final_grade"],
                           "Reviewer": (a.get("last_review") or {}).get("reviewer"),
                           "Interview": (a.get("last_review") or {}).get("interview_at"),
                           "Notes": (a.get("last_review") or {}).get("notes"),
                           "When": ((a.get("last_review") or {}).get("created_at") or "")[:16].replace("T", " ")}
                          for a in done], hide_index=True)
