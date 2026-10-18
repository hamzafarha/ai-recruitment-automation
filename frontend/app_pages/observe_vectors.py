"""Under the hood, parts 5-6: the Qdrant collections (the knowledge sources), their vectors, and one
retrieval step by step (BGE-M3 search, then BGE reranker)."""

import altair as alt
import pandas as pd
import streamlit as st

from app_pages import common

COLLECTIONS = {"hr_knowledge": "HR knowledge base: one point per section of the 15 HR documents",
               "candidates": "Candidates: one point per position, project, degree, skills group… of each CV",
               "jobs": "Jobs: one point for the requirements of each job, one per part of its description"}
ROLES = ["", "ml_engineer", "nlp_llm_engineer", "backend_developer", "devops_cloud", "data_analyst_scientist"]


def _collections() -> None:
    result = common.safe("GET", "/observe/collections")
    if not result:
        return
    st.dataframe([{"Collection": c["name"], "Points": c["points"], "Vector size": c["dimension"],
                   "Distance": c["distance"], "Status": c["status"], "What a point is": COLLECTIONS[c["name"]]}
                  for c in result["data"]], hide_index=True)
    st.caption("Every text is turned into 1,024 numbers by BGE-M3 (multilingual: French and English texts with the "
               "same meaning get close vectors). Qdrant finds the closest vectors by cosine similarity. Also visible "
               "in Qdrant's own dashboard: http://localhost:6333/dashboard")


def _browse() -> None:
    name = st.selectbox("Collection", list(COLLECTIONS), key="browse_collection")
    result = common.safe("GET", f"/observe/collections/{name}/points", params={"limit": 100})
    if not result or not result["points"]:
        st.info("This collection is empty: Dashboard → Set up.")
        return
    pts = result["points"]
    st.dataframe([{"Point": p["label"], "Text": p["payload"].get("text", "")[:160]} for p in pts], hide_index=True)
    labels = {p["id"]: p["label"] for p in pts}
    pick = st.selectbox("Open a point", list(labels), format_func=labels.get, key="browse_point")
    detail = common.safe("GET", f"/observe/collections/{name}/points/{pick}")
    if detail:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("**Payload** (stored with the vector)")
            st.json(detail["payload"], expanded=True)
        with c2:
            st.markdown(f"**Vector**: {detail['dimension']} numbers, length {detail['norm']}")
            st.line_chart(pd.DataFrame({"value": detail["vector"]}), height=220)
            st.caption("First values: " + ", ".join(f"{v:+.3f}" for v in detail["vector"][:10]) + " …")


def _map() -> None:
    c1, c2 = st.columns(2)
    name = c1.selectbox("Collection", list(COLLECTIONS), key="map_collection")
    color_by = c2.selectbox("Color by", {"hr_knowledge": ["document_id", "category", "language", "section"],
                                         "candidates": ["name", "section"],
                                         "jobs": ["title", "section"]}[name], key=f"map_color_{name}")
    result = common.safe("GET", f"/observe/collections/{name}/map", params={"color_by": color_by}, timeout=60)
    if not result or not result["points"]:
        st.info("Not enough points to draw.")
        return
    frame = pd.DataFrame(result["points"])
    st.altair_chart(alt.Chart(frame).mark_circle(size=90, opacity=0.8).encode(
        x=alt.X("x:Q", title="component 1"), y=alt.Y("y:Q", title="component 2"),
        color=alt.Color("group:N", title=color_by, legend=alt.Legend(columns=1, symbolLimit=40)),
        tooltip=["label", "text"]).interactive(), width="stretch")
    variance = result.get("explained_variance") or [0, 0]
    st.caption(f"The 1,024-dimension vectors flattened to 2 dimensions (PCA, {sum(variance):.0%} of the variation "
               "kept). Points close together are texts with a similar meaning: the snippets of one candidate, or "
               "the sections of one HR document, form groups. Hover a point to read it.")


def _retrieval() -> None:
    c1, c2, c3 = st.columns([4, 2, 1])
    query = c1.text_input("Query", "How are internships counted in years of experience?", key="explain_query")
    collection = c2.selectbox("Collection", list(COLLECTIONS), key="explain_collection")
    top_k = c3.number_input("Keep", 1, 10, 5, key="explain_k")
    role = ""
    if collection == "hr_knowledge":
        role = st.selectbox("Job-profile filter (optional)", ROLES, format_func=lambda r: r or "none",
                            key="explain_role")
    if st.button("▶ Search step by step", type="primary") and len(query.strip()) >= 2:
        result = common.safe("GET", "/observe/retrieval", timeout=120,
                             params={"q": query, "collection": collection, "role": role or None, "top_k": top_k})
        if result:
            st.session_state["explain_result"] = result
    result = st.session_state.get("explain_result")
    if not result:
        return
    e, s = result["embedding"], result["seconds"]
    st.markdown(f"**1. The query becomes a vector** — {e['model']}: {e['dimension']} numbers "
                f"({s['embedding']} s). First values: `{', '.join(f'{v:+.3f}' for v in e['first_values'][:6])} …`")
    st.markdown(f"**2. Qdrant returns the closest points** by cosine similarity ({len(result['dense'])} points, "
                f"{s['qdrant']} s)" + (f", filtered on the job profile {result['filter']}" if result["filter"] else ""))
    st.markdown(f"**3. The BGE cross-encoder rescores each (query, text) pair** and reorders them ({s['reranking']} s); "
                f"the top {result['top_k']} are used." if result["reranker"] else
                "**3. No reranker configured**: the cosine order is kept.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before reranking** (cosine)")
        st.dataframe([{"#": d["dense_rank"], "Cosine": d["cosine"], "Point": d["label"]} for d in result["dense"]],
                     hide_index=True)
    with c2:
        st.markdown("**After reranking** (what the app uses)")
        st.dataframe([{"#": r["rerank_rank"], "Reranker score": r["rerank_score"],
                       "Moved": (f"↑ {r['moved']}" if r["moved"] > 0 else f"↓ {-r['moved']}" if r["moved"] < 0
                                 else "="), "Point": r["label"], "Kept": "✅" if r["rerank_rank"] <= result["top_k"]
                       else ""} for r in result["reranked"]], hide_index=True)
    with st.expander("Texts of the kept points"):
        for r in result["reranked"][: result["top_k"]]:
            st.markdown(f"**{r['rerank_rank']}. {r['label']}** (score {r['rerank_score']})")
            st.caption(r["text"])


def render() -> None:
    st.title("🧭 Vectors & retrieval (RAG)")
    st.caption("Parts 5 and 6 of the architecture: the knowledge sources stored as vectors in Qdrant, and how a "
               "search finds the right ones — the same search used by the AI analysis (company rules) and by the "
               "assistant's tools (CV search, job search, HR knowledge search).")
    if not common.need_api():
        return
    tabs = st.tabs(["Collections", "Browse points", "2D map of the vectors", "One search, step by step"])
    for tab, show in zip(tabs, (_collections, _browse, _map, _retrieval), strict=True):
        with tab:
            show()
