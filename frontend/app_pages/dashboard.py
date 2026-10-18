"""Dashboard: getting started, key figures, recent activity and system status."""

import pandas as pd
import streamlit as st

from app_pages import common


def _setup_box(status: dict | None, figures: dict) -> None:
    status = status or {}
    ai = status.get("ai_server", {})
    provider = status.get("llm_provider", "?")
    server_ok = not ai.get("used") or bool(ai.get("reachable"))
    models_ok = provider != "mock" and server_ok
    search_ok = status.get("embedding_provider") == "local" or server_ok  # embeddings are enough to index
    knowledge_ok = bool(status.get("knowledge_ready"))
    data_ok = figures.get("jobs", 0) > 0 and figures.get("candidates", 0) > 0
    if not models_ok:
        models_text = ("run `notebooks/kaggle/model_server.ipynb` on Kaggle, paste the 4 lines it prints into `.env`, "
                       "restart the API")
    elif ai.get("used"):
        models_text = f"Qwen on the Kaggle AI server ({ai.get('model')}" + (
            " + fine-tuned adapter" if ai.get("adapter") else "") + ")"
    else:
        models_text = f"provider {provider}"
    with st.container(border=True):
        st.markdown("#### Getting started")
        st.markdown(f"{'✅' if models_ok else '⬜'} **1. AI models connected** — {models_text}  \n"
                    f"{'✅' if knowledge_ok else '⬜'} **2. HR knowledge base indexed** (RAG: 15 documents in Qdrant)"
                    f"  \n{'✅' if data_ok else '⬜'} **3. Jobs and candidates** — load the demo (5 real job "
                    "postings, 12 invented candidates) or add your own")
        if ai.get("used") and not ai.get("reachable"):
            st.error(f"Kaggle AI server: {ai.get('error')}")
        c1, c2 = st.columns(2)
        if c1.button("⚙️ Set up: index the HR knowledge + load the demo data", type="primary", width="stretch",
                     disabled=not search_ok):
            with st.spinner("Embedding the knowledge base, the jobs and the candidates (BGE-M3)…"):
                result = common.safe("POST", "/admin/setup", json={"demo_data": True}, timeout=common.LONG)
            if result:
                common.refresh()
                st.success(f"Ready: {result['knowledge_chunks']} knowledge chunks, {result['jobs_indexed']} jobs, "
                           f"{result['candidates_indexed']} candidates indexed.")
                st.rerun()
        if c2.button("🔄 Re-index jobs and candidates (after a Qdrant reset)", width="stretch",
                     disabled=not search_ok):
            with st.spinner("Re-indexing…"):
                result = common.safe("POST", "/admin/reindex", timeout=common.LONG)
            if result:
                common.refresh()
                st.success(f"{result['jobs']} jobs and {result['candidates']} candidates re-indexed.")


def render() -> None:
    st.title("💼 AI Recruitment Automation")
    st.caption("One workflow from a CV to the final decision (n8n): document processing, extraction by the fine-tuned "
               "Qwen2.5, a LangGraph screening agent (BGE-M3, Qdrant, BGE reranker, HR rules, evidence-checked "
               "analysis), the recruiter's decision and the answer to the candidate. Plus an AI assistant. Every "
               "model runs on Kaggle's free GPUs.")
    if not common.need_api():
        return
    status = common.status()
    figures = common.safe("GET", "/dashboard") or {}
    _setup_box(status, figures)

    cols = st.columns(6)
    cols[0].metric("Candidates", figures.get("candidates", 0))
    cols[1].metric("Open jobs", figures.get("open_jobs", 0))
    cols[2].metric("Analyses", figures.get("analyses", 0))
    cols[3].metric("Pending reviews", figures.get("pending_reviews", 0))
    cols[4].metric("Shortlisted / interview", figures.get("shortlisted", 0))
    avg = figures.get("average_score")
    cols[5].metric("Average score", f"{avg:g}" if avg is not None else "—")
    apps = figures.get("applications") or {}
    if any(apps.values()):
        st.markdown(f"**Application pipeline:** {apps.get('running', 0)} running · {apps.get('waiting_review', 0)} "
                    f"waiting for a decision · {apps.get('closed', 0)} closed · {apps.get('failed', 0)} stopped at a "
                    "failed step")
        st.page_link(common.PAGES["pipeline"], label="Open the Application pipeline", icon="🔄")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Recent analyses")
        recent = figures.get("recent_analyses") or []
        if recent:
            st.dataframe([{"Candidate": a["candidate_name"], "Job": a["job_title"],
                           "Grade": a["final_grade"] if a["final_grade"] is not None else a["grade"],
                           "Score": a["score"], "Method": common.method_short(a["method"]),
                           "Status": common.STATUS_LABELS.get(a["status"], a["status"])} for a in recent],
                         hide_index=True)
        else:
            st.caption("No analysis yet: Matching → rank the candidates of a job → run the AI analysis.")
        grades = figures.get("grades") or {}
        if any(grades.values()):
            st.markdown("#### Grades of all analyses")
            st.bar_chart(pd.DataFrame({"analyses": [grades.get(g, 0) for g in ("0", "1", "2", "3")]},
                                      index=["0 No fit", "1 Weak", "2 Partial", "3 Strong"]))
    with right:
        st.markdown(f"#### Notifications ({figures.get('unread_notifications', 0)} unread)")
        for n in figures.get("recent_notifications") or []:
            icon = "🤖" if n["source"] == "n8n" else "🔔"
            st.markdown(f"{icon} **{n['title']}**  \n<small>{n['created_at'][:16].replace('T', ' ')} · "
                        f"{n['event']}</small>", unsafe_allow_html=True)
        if not figures.get("recent_notifications"):
            st.caption("No notification yet.")

    with st.expander("System status"):
        health = common.health() or {}
        for name, component in (health.get("components") or {}).items():
            ok = component.get("healthy")
            st.markdown(f"{'🟢' if ok else '🔴' if ok is False else '⚪'} **{name}** — "
                        f"{component.get('details') or component.get('configured') or ''}")
        if status:
            st.markdown(f"**Models:** LLM `{status['llm_model']}` ({status['llm_provider']}), embeddings "
                        f"`{status['embedding_model']}` ({status['embedding_provider']}), reranker "
                        f"`{status['reranking_model']}`")
            st.markdown("**Qdrant collections:** " + (", ".join(f"{k}: {v} points" for k, v in
                                                               status["collections"].items())
                                                     or status.get("qdrant_error") or "—"))
            st.markdown(f"**n8n:** {status['n8n']['url']} (events " +
                        ("enabled" if status["n8n"]["events_enabled"] else "disabled: N8N_EVENTS_ENABLED=false") + ")")
