"""AI assistant (DSO4, diagram part 5): questions answered by the LangGraph agent with its tools."""

import json

import streamlit as st

from app_pages import common

EXAMPLES = [
    "Who are the best candidates for the GenAI Engineer job?",
    "Find candidates who have used Kubernetes.",
    "Is Mehdi Trabelsi a good fit for the AWS Cloud Engineer job? What are the gaps?",
    "How are internships counted in years of experience?",
    "Combien de temps conserve-t-on les données d'un candidat non retenu ?",
    "Among Priya Raman, Daniel Okoro and Hannah Weiss, who has the most hands-on experience with LLM applications?",
]
SOURCE_ICONS = {"candidate": "👤", "job": "💼", "knowledge": "📘"}


def show_run(run: dict) -> None:
    st.markdown(run.get("answer") or "_(no answer)_")
    for warning in run.get("warnings") or []:
        st.caption(f"⚠ {warning}")
    sources = run.get("sources") or []
    if sources:
        st.caption("Sources: " + " · ".join(f"{SOURCE_ICONS.get(s['type'], '•')} {s['title']}" for s in sources[:12]))
    steps = run.get("steps") or []
    with st.expander(f"How it answered: {len(steps)} tool call(s), {run.get('llm_calls') or 0} model calls, "
                     f"{run.get('seconds')} s"):
        st.caption(f"LangGraph: decide → act → … → respond ({run.get('graph') or 'graph'}), model "
                   f"{run.get('model') or '?'}")
        for i, step in enumerate(steps, 1):
            arguments = json.dumps(step.get("arguments") or {}, ensure_ascii=False)
            st.markdown(f"**{i}. `{step['tool']}`** `{arguments}` · {step.get('seconds', 0)} s")
            if step.get("thought"):
                st.caption(f"Why: {step['thought']}")
            if step.get("ok"):
                st.json(step.get("data"), expanded=False)
            else:
                st.error(step.get("error") or "failed")
        if run.get("final_thought"):
            st.caption(f"Then: {run['final_thought']}")
        if not steps:
            st.caption("No tool was needed.")


def render() -> None:
    st.title("🤖 AI assistant")
    st.caption("Ask about candidates, jobs and the company's HR rules, in English or French. A LangGraph agent "
               "(Qwen2.5-7B) picks its tools — CV search, job search, HR knowledge search (RAG), candidate database, "
               "ranking, matching, reranking — runs them, and answers from their results only. Each question is "
               "answered on its own, in about 30 s to 2 min; every run is stored with its trace.")
    if not common.need_api():
        return
    with st.expander("Its tools"):
        tools = (common.safe("GET", "/agent/tools") or {}).get("data") or []
        for tool in tools:
            st.markdown(f"- `{tool['signature']}` — {tool['description']}")
    history = st.session_state.setdefault("chat", [])
    if not history:
        st.markdown("**Try:**")
        cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            if cols[i % 2].button(example, key=f"example_{i}", width="stretch"):
                st.session_state["pending_question"] = example
                st.rerun()
    for item in history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            show_run(item["run"])
    question = st.chat_input("Ask a question…") or st.session_state.pop("pending_question", None)
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Choosing tools, searching, reading the results…"):
                run = common.safe("POST", "/agent/ask", json={"question": question}, timeout=common.LONG)
            if run:
                show_run(run)
                history.append({"question": question, "run": run})
    if history and st.button("🧹 Clear the conversation"):
        st.session_state["chat"] = []
        st.rerun()
    with st.expander("Previous questions (stored traces)"):
        runs = (common.safe("GET", "/agent/runs", params={"limit": 20}) or {}).get("data") or []
        if runs:
            st.dataframe([{"When": r["created_at"][:16].replace("T", " "), "Question": r["question"],
                           "Tools": ", ".join(s["tool"] for s in r["steps"]), "Seconds": r["seconds"]} for r in runs],
                         hide_index=True)
        else:
            st.caption("None yet.")
