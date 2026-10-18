"""Under the hood: the 9 parts of the architecture, live, with a link to observe each one."""

import streamlit as st

from app_pages import common

GREEN, GREY, RED = "#d1fadf", "#eef1f5", "#fde2e1"


def _dot_label(*lines: str) -> str:
    return "\\n".join(line.replace('"', "'") for line in lines if line)


def diagram(o: dict) -> str:
    """The architecture as a graph, each part colored by its live state."""
    t, v, ai, n8n = o["tables"], o["vectors"], o["ai_server"], o["n8n"]
    live = sum(bool(w.get("live")) for w in n8n.get("webhooks") or [])
    vectors_ok = isinstance(v.get("hr_knowledge"), int) and v.get("hr_knowledge", 0) > 0
    ai_ok = not ai.get("used") or ai.get("reachable")
    parts = {
        "users": (_dot_label("1. Users & inputs", f"{t['candidates']} candidates · {t['jobs']} jobs"),
                  GREEN if t["candidates"] else GREY),
        "n8n": (_dot_label("2. n8n orchestrator", f"{live}/4 workflows live"),
                GREEN if live == 4 else RED if n8n.get("reachable") is False else GREY),
        "docs": (_dot_label("3. Document processing", "PyMuPDF · DOCX · OCR → Qwen"), GREEN if ai_ok else RED),
        "ft": (_dot_label("4. Fine-tuned Qwen2.5-7B", "QLoRA adapter " + ("loaded" if ai.get("adapter") else "—")),
               GREEN if ai.get("adapter") else GREY),
        "rag": (_dot_label("5. Agentic RAG", f"{o['analyses_by_method'].get('agent', 0)} screenings by the agent",
                           f"{t['agent_runs']} assistant runs"),
                GREEN if vectors_ok and ai_ok else GREY),
        "qdrant": (_dot_label("Qdrant (BGE-M3 vectors)",
                              " · ".join(f"{k} {n}" for k, n in v.items() if isinstance(n, int))),
                   GREEN if vectors_ok else RED if "error" in v else GREY),
        "outputs": (_dot_label("6. Evidence-backed reports", f"{t['analyses']} analyses"),
                    GREEN if t["analyses"] else GREY),
        "notify": (_dot_label("7. Notifications", f"{t['notifications']} events"),
                   GREEN if t["notifications"] else GREY),
        "pg": (_dot_label("8. PostgreSQL", f"{sum(t.values())} rows in {len(t)} tables"), GREEN),
        "human": (_dot_label("9. Human validation", f"{t['reviews']} decisions"), GREEN if t["reviews"] else GREY),
    }
    nodes = "\n".join(f'  {key} [label="{label}", fillcolor="{color}"];' for key, (label, color) in parts.items())
    return ("digraph {\n  rankdir=LR; bgcolor=transparent;\n"
            '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#9aa5b1"];\n'
            '  edge [color="#7b8794", fontname="Helvetica", fontsize=9];\n' + nodes + "\n"
            '  users -> n8n [label="CV, job ad"]; users -> docs [label="upload"]; n8n -> docs;\n'
            '  ft -> docs [label="extraction"]; ft -> rag [label="analysis (base model)"];\n'
            '  docs -> pg [label="profile"]; docs -> qdrant [label="snippets"];\n'
            '  qdrant -> rag [label="retrieval + reranking"]; pg -> rag [label="tools"];\n'
            '  rag -> outputs; outputs -> human; human -> pg [label="decision"];\n'
            '  human -> notify; outputs -> notify; notify -> n8n [label="events"];\n}')


def _card(title: str, text: str, lines: list[str], page: str | None, label: str = "Observe") -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(text)
        for line in lines:
            st.markdown(f"- {line}")
        if page:
            st.page_link(common.PAGES[page], label=label, icon="🔬")


def render() -> None:
    st.title("🗺️ Architecture, live")
    st.caption("Each part of the architecture as it is right now, with a link to the page that shows it working. "
               "Green = working with data; grey = not used yet; red = a problem.")
    if not common.need_api():
        return
    o = common.safe("GET", "/observe/overview", timeout=60)
    if not o:
        return
    st.graphviz_chart(diagram(o), width="stretch")
    t, v, ai, n8n = o["tables"], o["vectors"], o["ai_server"], o["n8n"]
    webhooks = n8n.get("webhooks") or []
    last = o.get("last_extraction") or {}
    rows = [st.columns(3) for _ in range(3)]
    cells = [c for row in rows for c in row]
    with cells[0]:
        _card("1. Users & inputs", "Recruiters upload CVs and job ads; candidates apply in the portal. Every CV "
              "goes into the application pipeline.",
              [f"candidates by source: {o['candidates_by_source'] or '—'}", f"{t['jobs']} jobs",
               f"applications: {o.get('applications_by_status') or '—'}"], "pipeline", "Follow the applications")
    with cells[1]:
        _card("2. n8n orchestrator", "Workflows: the application pipeline (one execution per application, CV → "
              "final decision), new job, event messages, daily digest, error alert.",
              [f"n8n {'reachable' if n8n.get('reachable') else 'not reachable'}, events "
               f"{'enabled' if n8n.get('events_enabled') else 'disabled'}",
               f"{sum(bool(w.get('live')) for w in webhooks)}/4 workflows live",
               f"{o['notifications_by_source'].get('n8n', 0)} messages written by n8n"], "notifications")
    with cells[2]:
        _card("3. Document processing", "PDF text layer, Word, OCR for scans; then Qwen fills the profile.",
              [f"last upload: {last.get('candidate') or '—'} ({last.get('input_type') or '—'}, "
               f"{last.get('seconds') or '—'} s)" if last else "no CV uploaded yet"], "observe_documents")
    with cells[3]:
        _card("4. Fine-tuning (QLoRA)", "Qwen2.5-7B taught our CV format; used for every CV extraction.",
              [f"adapter on the AI server: {'yes' if ai.get('adapter') else 'no'}",
               "compare base vs fine-tuned on your CV"], "observe_documents", "Compare on a CV")
    with cells[4]:
        _card("5. Agentic RAG layer", "The pipeline's screening agent (LangGraph): understands the task, chooses "
              "its tools, retrieves (BGE-M3 + Qdrant) and reranks (BGE reranker) CV evidence and HR rules, iterates, "
              "then Qwen analyses and code checks every claim. Plus the assistant (8 tools).",
              [f"analyses: {o['analyses_by_method'] or '—'}", f"{t['agent_runs']} assistant runs"], "pipeline",
              "Observe the screening agent")
        st.page_link(common.PAGES["observe_agent"], label="Observe the assistant", icon="🔬")
        st.page_link(common.PAGES["observe_vectors"], label="Observe retrieval and reranking", icon="🔬")
        st.page_link(common.PAGES["observe_matching"], label="Observe contextualization", icon="🔬")
    with cells[5]:
        _card("6. Outputs & knowledge sources", "Evidence-backed reports; HR knowledge base, candidates and jobs "
              "as searchable vectors.",
              [f"{t['analyses']} reports", "vectors: " + (", ".join(f"{k} {n}" for k, n in v.items()))],
              "reports")
    with cells[6]:
        _card("7. External services", "The Kaggle AI server (all models) and notifications.",
              [f"AI server: {'reachable' if ai.get('reachable') else 'not used' if not ai.get('used') else 'down'}"
               + (f", {ai.get('requests')} requests served" if ai.get("requests") is not None else ""),
               f"notifications: {o['notifications_by_source'] or '—'}"], "notifications")
    with cells[7]:
        _card("8. Data storage", "PostgreSQL: the system's records.",
              [", ".join(f"{k} {n}" for k, n in t.items())], "observe_database")
    with cells[8]:
        _card("9. Human validation", "A recruiter decides every case; the AI only proposes.",
              [f"{t['reviews']} decisions", f"statuses: {o['analyses_by_status'] or '—'}"], "review")
    if ai.get("used") and ai.get("reachable"):
        with st.expander("Kaggle AI server details"):
            st.json(ai)
