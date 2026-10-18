"""Knowledge sources search (diagram parts 5-6): BGE-M3 in Qdrant, reordered by the BGE reranker."""

import streamlit as st

from app_pages import common

ROLES = ["", "ml_engineer", "nlp_llm_engineer", "backend_developer", "devops_cloud", "data_analyst_scientist"]


def render() -> None:
    st.title("🔎 Search")
    st.caption("Semantic search in the three knowledge sources: the HR knowledge base (policies, scoring rubric, job "
               "profiles, skills; English and French), the candidates' CVs and the job descriptions. The query is "
               "embedded with BGE-M3, Qdrant returns the closest chunks, and the BGE cross-encoder reorders them. "
               "These are the searches the AI assistant and the AI analysis use.")
    if not common.need_api():
        return
    knowledge, cvs, jobs = st.tabs(["📘 HR knowledge base", "👤 Candidates", "💼 Jobs"])
    with knowledge:
        c1, c2, c3 = st.columns([4, 2, 1])
        query = c1.text_input("Question", "How do we count years of experience for internships?", key="kb_query")
        role = c2.selectbox("Job profile filter", ROLES, format_func=lambda r: r or "all roles", key="kb_role")
        top_k = c3.number_input("Results", 1, 10, 4, key="kb_k")
        if query and len(query) >= 2:
            result = common.safe("GET", "/search/knowledge", params={"q": query, "role": role or None, "top_k": top_k},
                                 timeout=120)
            for hit in (result or {}).get("data") or []:
                with st.container(border=True):
                    st.markdown(f"**{hit['doc_id']}** — {hit['title']} / {hit['section']} · relevance "
                                f"`{hit['score']:.3f}` · {hit.get('language') or ''}")
                    st.markdown(hit["text"])
            if result is not None and not result.get("data"):
                st.info("No result: is the knowledge base indexed? Dashboard → Set up.")
    with cvs:
        c1, c2 = st.columns([5, 1])
        query = c1.text_input("What are you looking for?", "Kubernetes in production", key="cv_query")
        top_k = c2.number_input("Results", 1, 20, 5, key="cv_k")
        if query and len(query) >= 2:
            result = common.safe("GET", "/search/candidates", params={"q": query, "top_k": top_k}, timeout=120)
            rows = (result or {}).get("data") or []
            for hit in rows:
                with st.container(border=True):
                    st.markdown(f"**{hit['name'] or 'Unnamed'}** · relevance `{hit['score']:.3f}` · {hit['label']}")
                    st.caption(hit["evidence"])
            if result is not None and not rows:
                st.info("No candidate indexed yet.")
    with jobs:
        query = st.text_input("Skills or profile", "SQL, Power BI dashboards and ETL pipelines", key="job_query")
        if query and len(query) >= 2:
            result = common.safe("GET", "/search/jobs", params={"q": query, "top_k": 5}, timeout=120)
            for hit in (result or {}).get("data") or []:
                with st.container(border=True):
                    st.markdown(f"**{hit['name']}** · relevance `{hit['score']:.3f}` · {hit['section']}")
                    st.caption(hit["evidence"][:600])
