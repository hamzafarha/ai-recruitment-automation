"""Recruiter Web UI built with Streamlit."""

import os

import requests
import streamlit as st

st.set_page_config(
    page_title="AI Recruitment Automation",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")


def check_backend_status() -> tuple[bool, dict]:
    """Query FastAPI backend health endpoint."""
    try:
        res = requests.get(f"{BACKEND_API_URL}/api/v1/health", timeout=2)
        if res.status_code == 200:
            return True, res.json()
        return False, {"error": f"Status code {res.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}


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
