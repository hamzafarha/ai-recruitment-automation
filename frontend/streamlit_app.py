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
    st.header("Candidate Directory")
    st.info(
        "Upload new candidate CVs (PDF, DOCX) to trigger ingestion and structured extraction."
    )
    uploaded_file = st.file_uploader(
        "Upload Candidate Resume", type=["pdf", "docx", "png", "jpg"]
    )
    if uploaded_file and st.button("Upload to Processing Pipeline"):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            res = requests.post(
                f"{BACKEND_API_URL}/api/v1/documents/upload", files=files, timeout=5
            )
            if res.status_code == 200:
                st.success(f"File uploaded successfully: {res.json().get('filename')}")
            else:
                st.error(f"Upload failed: {res.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

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
