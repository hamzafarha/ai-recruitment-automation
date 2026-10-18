"""Recruiter interface (Streamlit): run with ``python -m streamlit run frontend/streamlit_app.py``.

Every page talks to the FastAPI backend (BACKEND_API_URL, default http://localhost:8000); the
Evaluation page reads the reports in data/evaluation/.
"""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", Path(__file__).resolve().parent):  # works without "pip install -e ." too
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

st.set_page_config(page_title="AI Recruitment Automation", page_icon="💼", layout="wide",
                   initial_sidebar_state="expanded")

from app_pages import (  # noqa: E402
    architecture,
    assistant,
    candidates,
    common,
    dashboard,
    evaluation,
    jobs,
    matching,
    notifications,
    observe_agent,
    observe_database,
    observe_documents,
    observe_matching,
    observe_vectors,
    pipeline,
    portal,
    reports,
    review,
    search,
)

common.PAGES.update({
    "dashboard": st.Page(dashboard.render, title="Dashboard", icon="🏠", url_path="dashboard", default=True),
    "pipeline": st.Page(pipeline.render, title="Application pipeline", icon="🔄", url_path="pipeline"),
    "candidates": st.Page(candidates.render, title="Candidates", icon="👤", url_path="candidates"),
    "jobs": st.Page(jobs.render, title="Jobs", icon="💼", url_path="jobs"),
    "matching": st.Page(matching.render, title="Matching", icon="🎯", url_path="matching"),
    "reports": st.Page(reports.render, title="Reports", icon="📄", url_path="reports"),
    "review": st.Page(review.render, title="Human validation", icon="✅", url_path="validation"),
    "assistant": st.Page(assistant.render, title="AI assistant", icon="🤖", url_path="assistant"),
    "search": st.Page(search.render, title="Search", icon="🔎", url_path="search"),
    "portal": st.Page(portal.render, title="Candidate portal", icon="📬", url_path="portal"),
    "notifications": st.Page(notifications.render, title="Notifications & n8n", icon="🔔", url_path="notifications"),
    "evaluation": st.Page(evaluation.render, title="Evaluation", icon="📊", url_path="evaluation"),
    "architecture": st.Page(architecture.render, title="Architecture (live)", icon="🗺️", url_path="architecture"),
    "observe_documents": st.Page(observe_documents.render, title="Document processing", icon="📄",
                                 url_path="observe-documents"),
    "observe_vectors": st.Page(observe_vectors.render, title="Vectors & RAG", icon="🧭", url_path="observe-vectors"),
    "observe_matching": st.Page(observe_matching.render, title="Matching internals", icon="🧩",
                                url_path="observe-matching"),
    "observe_agent": st.Page(observe_agent.render, title="Assistant internals", icon="🤖", url_path="observe-agent"),
    "observe_database": st.Page(observe_database.render, title="Database", icon="🗄️", url_path="observe-database"),
})
p = common.PAGES
navigation = st.navigation({
    "Overview": [p["dashboard"], p["pipeline"]],
    "Recruiting": [p["candidates"], p["jobs"], p["matching"], p["reports"], p["review"]],
    "AI": [p["assistant"], p["search"]],
    "Automation": [p["portal"], p["notifications"]],
    "Under the hood": [p["architecture"], p["observe_documents"], p["observe_vectors"], p["observe_matching"],
                       p["observe_agent"], p["observe_database"]],
    "Results": [p["evaluation"]],
})
st.sidebar.markdown("### 💼 AI Recruitment")
common.sidebar()
navigation.run()
