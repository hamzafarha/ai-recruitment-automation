"""Recruiter Web UI built with Streamlit."""

import json
import os
from pathlib import Path

import requests
import streamlit as st

st.set_page_config(
    page_title="AI Recruitment Automation",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")


@st.cache_data(ttl=60, show_spinner=False)
def check_backend_status() -> tuple[bool, dict]:
    """Query FastAPI backend health endpoint (cached one minute: Streamlit reruns on every click).

    The health check also tests PostgreSQL and Qdrant: when Docker is stopped each of those waits
    for its connection timeout, so the API needs several seconds to answer.
    """
    try:
        res = requests.get(f"{BACKEND_API_URL}/api/v1/health", timeout=15)
        if res.status_code == 200:
            return True, res.json()
        return False, {"error": f"Status code {res.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "evaluation"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def _pts(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.1f} pts"


def show_evaluation() -> None:
    """Before / after fine-tuning: two extraction evaluation runs side by side (files on this PC)."""
    import pandas as pd

    from recruitment_ai.evaluation.compare import (
        compare_documents,
        compare_groups,
        compare_metrics,
        default_pair,
        list_runs,
        load_run,
    )

    st.header("Extraction: before vs after fine-tuning")
    st.caption(
        "Same frozen evaluation CVs, same code and same scoring: only the model differs. The runs are "
        "made on Kaggle and downloaded with `python scripts/kaggle_run.py fetch <job>`."
    )
    root = EVAL_DIR / "extraction"
    names = list_runs(root)
    if not names:
        st.info("No evaluation report on this computer yet: `python scripts/kaggle_run.py fetch extraction`.")
        return
    default_before, default_after = default_pair(names)
    col1, col2 = st.columns(2)
    before_name = col1.selectbox("Before", names, index=names.index(default_before))
    after_options = [n for n in names if n != before_name]
    if not after_options:
        st.info("Only one run so far. The fine-tuned run appears here after "
                "`python scripts/kaggle_run.py all extraction-ft`.")
        return
    after_name = col2.selectbox("After", after_options, index=after_options.index(default_after)
                                if default_after in after_options else 0)
    before, after = load_run(root, before_name), load_run(root, after_name)
    mb, ma = before["metrics"], after["metrics"]

    rows = compare_metrics(before, after)
    cards = st.columns(4)
    for i, row in enumerate(rows):
        cards[i % 4].metric(row["metric"], _pct(row["after"]),
                            delta=None if row["delta"] is None else _pts(row["delta"]))
    halluc_b = (mb["overall"].get("identity") or {}).get("hallucinations")
    halluc_a = (ma["overall"].get("identity") or {}).get("hallucinations")
    sec_b, sec_a = mb.get("mean_seconds_per_document"), ma.get("mean_seconds_per_document")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", ma.get("documents"))
    c2.metric("Errors", ma.get("errors"), delta=(ma.get("errors") or 0) - (mb.get("errors") or 0),
              delta_color="inverse")
    c3.metric("Identity hallucinations", halluc_a, delta_color="inverse",
              delta=halluc_a - halluc_b if halluc_a is not None and halluc_b is not None else None)
    c4.metric("Seconds per CV", sec_a, delta_color="inverse",
              delta=round(sec_a - sec_b, 1) if sec_a and sec_b else None)

    import altair as alt

    # Axis from 0 to 100% and metrics in the cards' order: a truncated axis exaggerates small gaps
    bars = pd.DataFrame([{"metric": r["metric"], "run": run, "score": r[key] * 100}
                         for r in rows for run, key in (("Before", "before"), ("After", "after"))
                         if r[key] is not None])
    st.altair_chart(
        alt.Chart(bars).mark_bar().encode(
            y=alt.Y("metric:N", sort=[r["metric"] for r in rows], title=None),
            yOffset=alt.YOffset("run:N", sort=["Before", "After"]),
            x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title="score (%)"),
            color=alt.Color("run:N", sort=["Before", "After"], title=None,
                            scale=alt.Scale(range=["#9ecae9", "#1f77b4"])),
            tooltip=["metric", "run", alt.Tooltip("score:Q", format=".1f")],
        ),
        use_container_width=True,
    )

    tab_groups, tab_docs, tab_training = st.tabs(["By document type", "Per CV", "Fine-tuning run"])
    with tab_groups:
        for group, label in (("input_type", "Input type"), ("source", "Source"), ("language", "Language")):
            st.markdown(f"**{label}** — overall F1")
            st.dataframe(
                [{label: r[group], "Documents": r["documents"], "Before": _pct(r["before"]),
                  "After": _pct(r["after"]), "Change": _pts(r["delta"])} for r in compare_groups(before, after, group)],
                hide_index=True, use_container_width=True)
    with tab_docs:
        docs = compare_documents(before, after)
        better = sum(1 for d in docs if (d["delta"] or 0) > 0.005)
        worse = sum(1 for d in docs if (d["delta"] or 0) < -0.005)
        st.write(f"**{better}** CVs improved, **{worse}** got worse, **{len(docs) - better - worse}** unchanged "
                 "(overall F1 per CV).")
        st.dataframe(
            [{"CV": d["doc_id"], "Type": d["input_type"], "Language": d["language"], "Before": _pct(d["before"]),
              "After": _pct(d["after"]), "Change": _pts(d["delta"]),
              "Error": "before" if d["error_before"] else ("after" if d["error_after"] else "")} for d in docs],
            hide_index=True, use_container_width=True)
    with tab_training:
        report = EVAL_DIR / "finetune" / "report.md"
        state = EVAL_DIR / "finetune" / "training_state.json"
        if report.exists():
            st.markdown(report.read_text(encoding="utf-8"))
        if state.exists():
            losses = json.loads(state.read_text(encoding="utf-8")).get("losses", [])
            if losses:
                st.markdown("**Training loss** (lower = the model fits the training answers better)")
                st.line_chart(pd.DataFrame(losses).set_index("step")["loss"])
        else:
            st.info("Training details not downloaded yet: `python scripts/kaggle_run.py fetch finetune`.")


# Sidebar Navigation
st.sidebar.title("💼 AI Recruitment")
backend_alive, backend_info = check_backend_status()

if backend_alive:
    st.sidebar.success("Backend API Connected")
    st.sidebar.caption(f"Host: {BACKEND_API_URL}")
else:
    st.sidebar.error("Backend Disconnected")
    st.sidebar.caption(f"Attempting {BACKEND_API_URL}")

nav_selection = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Candidates",
        "Evaluation",
        "Candidate Details",
        "Jobs",
        "Candidate Matching",
        "AI Analysis",
        "Reports",
        "Recruiter Feedback",
    ],
)

st.title("AI Recruitment Automation System")

if nav_selection == "Dashboard":
    st.header("Executive Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Active Jobs", value="12", delta="+2")
    with col2:
        st.metric(label="CVs Processed", value="148", delta="+15")
    with col3:
        st.metric(label="Qualified Matches", value="42", delta="+6")
    with col4:
        st.metric(label="Avg Match Score", value="84.2%", delta="+1.8%")

    st.divider()
    st.subheader("System Health & Connectivity")
    if backend_alive:
        st.json(backend_info)
    else:
        st.warning(
            "FastAPI backend is offline. Start the backend with: `uvicorn api.main:app --reload`"
        )

elif nav_selection == "Candidates":
    st.header("CV Extraction")
    st.caption(
        "Upload a CV (PDF, Word or image). It is read on this computer (OCR for scans); "
        "Qwen then extracts the candidate profile (about a minute with the Kaggle model server)."
    )
    uploaded_file = st.file_uploader(
        "Candidate CV", type=["pdf", "docx", "png", "jpg", "jpeg"]
    )
    if uploaded_file and st.button("Extract profile", type="primary"):
        with st.spinner("Reading the CV and extracting the profile..."):
            try:
                res = requests.post(
                    f"{BACKEND_API_URL}/api/v1/candidates/extract",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=900,
                )
                if res.ok:
                    st.session_state["extraction"] = res.json()
                else:
                    detail = res.json().get("detail", res.text) if "json" in res.headers.get("content-type", "") else res.text
                    st.error(f"Extraction failed ({res.status_code}): {detail}")
            except requests.RequestException as e:
                st.error(f"Backend not reachable: {e}")

    result = st.session_state.get("extraction")
    if result:
        profile, stats = result["profile"], result["stats"]
        if result["llm_provider"] == "mock":
            st.warning(
                "LLM_PROVIDER=mock: these are fake answers. To use Qwen, start the Kaggle model "
                "server and set LLM_PROVIDER=remote in .env (README, section 8)."
            )
        st.subheader(profile.get("full_name") or "Name not found")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"**Email**  \n{profile.get('email') or '—'}")
        col2.markdown(f"**Phone**  \n{profile.get('phone') or '—'}")
        col3.markdown(f"**Location**  \n{profile.get('location') or '—'}")
        if profile.get("summary"):
            st.write(profile["summary"])

        st.markdown("#### Experience")
        jobs = [
            {"Title": j.get("title"), "Company": j.get("company"), "Start": j.get("start_date"),
             "End": j.get("end_date"), "Location": j.get("location")}
            for j in profile.get("experience", [])
        ]
        if jobs:
            st.dataframe(jobs, use_container_width=True, hide_index=True)
        else:
            st.write("—")

        st.markdown("#### Education")
        schools = [
            {"Degree": e.get("degree"), "Field": e.get("field_of_study"),
             "Institution": e.get("institution"), "End": e.get("end_date")}
            for e in profile.get("education", [])
        ]
        if schools:
            st.dataframe(schools, use_container_width=True, hide_index=True)
        else:
            st.write("—")

        st.markdown("#### Projects")
        for project in profile.get("projects", []):
            tech = f" · *{', '.join(project['technologies'])}*" if project.get("technologies") else ""
            st.markdown(f"- **{project['title']}**: {project.get('description') or ''}{tech}")
        if not profile.get("projects"):
            st.write("—")

        st.markdown("#### Skills")
        st.write(", ".join(s["name"] for s in profile.get("skills", [])) or "—")
        col1, col2 = st.columns(2)
        col1.markdown("#### Languages")
        col1.write(", ".join(
            x["language"] + (f" ({x['proficiency']})" if x.get("proficiency") else "")
            for x in profile.get("languages", [])
        ) or "—")
        col2.markdown("#### Certifications")
        col2.write(", ".join(c["name"] for c in profile.get("certifications", [])) or "—")

        st.caption(
            f"{result['filename']} · input: {stats.get('input_type', '?')} · "
            f"model: {stats.get('model', '?')}{' + fine-tuned adapter' if stats.get('adapter') else ''} · "
            f"{stats.get('seconds', '?')} s generation, {stats.get('parse_seconds', '?')} s reading"
            + (" · CV shortened (too long for the GPU)" if stats.get("retried_after_out_of_memory") else "")
        )
        with st.expander("Raw JSON"):
            st.json(result)

elif nav_selection == "Evaluation":
    show_evaluation()

elif nav_selection == "Candidate Details":
    st.header("Candidate Profile Details")
    st.write(
        "Candidate detailed view, career timeline, extracted skills, and education history."
    )

elif nav_selection == "Jobs":
    st.header("Job Vacancies & Requirements")
    st.write("Configured positions, target requirements, and required competencies.")

elif nav_selection == "Candidate Matching":
    st.header("Candidate ↔ Job Matching")
    st.write(
        "Multidimensional evaluation across semantics, required skills, and experience."
    )

elif nav_selection == "AI Analysis":
    st.header("Multi-Agent AI Analysis")
    st.write(
        "LangGraph agent traces, qualitative candidate assessment, and grounded evidence retrieval."
    )

elif nav_selection == "Reports":
    st.header("Candidate Assessment Reports")
    st.write("Exportable executive summary reports and structured profiles.")

elif nav_selection == "Recruiter Feedback":
    st.header("Human-in-the-Loop Validation")
    st.write(
        "Recruiter review interface: Approve, Reject, or Modify algorithmic recommendations."
    )
    st.selectbox(
        "Select Candidate Match",
        ["Match #101 - Senior ML Engineer", "Match #102 - Data Architect"],
    )
    decision = st.radio("Review Decision", ["Approved", "Rejected", "Modified"])
    rating = st.slider("Quality Rating", 1, 5, 4)
    comments = st.text_area("Recruiter Notes / Modifications")
    if st.button("Submit Feedback"):
        st.success("Recruiter validation recorded.")
