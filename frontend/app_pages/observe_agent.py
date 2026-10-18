"""Under the hood, part 5: how the LangGraph assistant works — its graph, its tools, its prompts, and
the exact path each question took through the graph. (The pipeline's screening agent is observed on the
Application pipeline page.)"""

import json

import streamlit as st

from app_pages import common

GRAPH = """digraph {
  rankdir=LR; bgcolor=transparent;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#9aa5b1", fillcolor="#eef1f5"];
  START [shape=circle, label="start", fillcolor="#d1fadf"];
  END [shape=doublecircle, label="end", fillcolor="#d1fadf"];
  decide [label="decide\\nQwen reads the question,\\nthe jobs and the results so far,\\nand picks ONE tool (or answers)", fillcolor="#e4ecf5"];
  act [label="act\\nthe tool runs: Qdrant search +\\nreranker, PostgreSQL, matching", fillcolor="#fff4d6"];
  respond [label="respond\\nQwen writes the answer\\nfrom the tool results only", fillcolor="#e4ecf5"];
  START -> decide;
  decide -> act [label="a tool is chosen"];
  act -> decide [label="observation"];
  decide -> respond [label="enough to answer,\\nor 4 tool calls done"];
  respond -> END;
}"""


def _escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace('"', "'")[:60]


def path_graph(timeline: list[dict]) -> str:
    """The path one question took: every node run, in order, with its time."""
    lines = ['digraph {', '  rankdir=LR; bgcolor=transparent;',
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#9aa5b1"];',
             '  s [shape=circle, label="start", fillcolor="#d1fadf"];']
    previous = "s"
    for i, event in enumerate(timeline, 1):
        node = event["node"]
        if node == "decide":
            action = (event.get("decision") or {}).get("action") or ("error" if event.get("error") else "—")
            label = f"{i}. decide\\n→ {_escape(action)}\\n{event.get('seconds', 0)} s"
            color = "#fde2e1" if event.get("error") else "#e4ecf5"
        elif node == "act":
            label = f"{i}. act: {_escape(event.get('tool'))}\\n{_escape(event.get('result', ''))}\\n{event.get('seconds', 0)} s"
            color = "#fff4d6" if event.get("ok") else "#fde2e1"
        else:
            label = f"{i}. respond\\n{event.get('seconds', 0)} s"
            color = "#fde2e1" if event.get("error") else "#e4ecf5"
        lines.append(f'  n{i} [label="{label}", fillcolor="{color}"];')
        lines.append(f"  {previous} -> n{i};")
        previous = f"n{i}"
    lines += ['  e [shape=doublecircle, label="end", fillcolor="#d1fadf"];', f"  {previous} -> e;", "}"]
    return "\n".join(lines)


def _what(event: dict) -> str:
    if event["node"] == "decide":
        if event.get("error"):
            return f"planning failed: {event['error']}"
        d = event.get("decision") or {}
        args = json.dumps(d.get("arguments") or {}, ensure_ascii=False)
        return f"“{d.get('thought') or ''}” → {d.get('action')} {args if args != '{}' else ''} ({event.get('why')})"
    if event["node"] == "act":
        return f"{event.get('tool')}({json.dumps(event.get('arguments') or {}, ensure_ascii=False)}) → {event.get('result')}"
    return ("answer written" + (f", {event.get('answer_chars')} characters" if event.get("answer_chars") else "")
            + (" (fallback: the tool results as they are)" if event.get("fallback") else ""))


def show_run(run: dict) -> None:
    timeline = run.get("timeline") or []
    st.markdown(f"**Question:** {run['question']}")
    if not timeline:
        st.info("This run was recorded before timelines existed: ask a new question to see its path.")
    else:
        st.graphviz_chart(path_graph(timeline), width="stretch")
        st.dataframe([{"#": i, "Node": e["node"], "What happened": _what(e), "Seconds": e.get("seconds"),
                       "Tokens in → out": (f"{e['prompt_tokens']} → {e['output_tokens']}"
                                           if e.get("prompt_tokens") else "")}
                      for i, e in enumerate(timeline, 1)], hide_index=True,
                     column_config={"What happened": st.column_config.TextColumn(width="large")})
    st.markdown(f"**Answer** ({run.get('seconds')} s in total, {run.get('llm_calls')} model calls, "
                f"graph: {run.get('graph')}):")
    with st.container(border=True):
        st.markdown(run.get("answer") or "")
    for step in run.get("steps") or []:
        with st.expander(f"Tool result: {step['tool']} ({step.get('seconds')} s)"):
            st.json(step.get("data") if step.get("ok") else {"error": step.get("error")}, expanded=False)
    if run.get("sources"):
        st.caption("Sources: " + " · ".join(s["title"] for s in run["sources"]))


def render() -> None:
    st.title("🤖 Assistant internals")
    st.caption("Part 5 of the architecture has two LangGraph agents. The screening agent works inside the "
               "application pipeline (see its path for each application on the Application pipeline page). The AI "
               "assistant, shown here, answers recruiters' questions: a graph of three nodes. Qwen2.5-7B decides "
               "which tool to call, the tool runs, and the loop goes on until Qwen can answer (at most 4 tool calls). "
               "Code guards it: arguments are validated, a repeated call is refused, a failed step degrades "
               "gracefully, and every run is stored with its timeline.")
    if not common.need_api():
        return
    st.graphviz_chart(GRAPH, width="stretch")
    tools = common.safe("GET", "/agent/tools") or {}
    with st.expander(f"Its {len(tools.get('data') or [])} tools"):
        st.dataframe([{"Tool": t["signature"], "What it does": t["description"]} for t in tools.get("data") or []],
                     hide_index=True, column_config={"What it does": st.column_config.TextColumn(width="large")})
    prompts = tools.get("system_prompts") or {}
    if prompts:
        with st.expander("What Qwen is told before each decision (system prompt of “decide”)"):
            st.code(prompts.get("decide", ""), language="text", wrap_lines=True)
        with st.expander("What Qwen is told before answering (system prompt of “respond”)"):
            st.code(prompts.get("respond", ""), language="text", wrap_lines=True)

    st.subheader("Watch a question go through the graph")
    question = st.text_input("Ask a question", "Who are the best candidates for the GenAI Engineer job?",
                             key="observe_question")
    if st.button("▶ Ask and trace", type="primary") and len(question.strip()) >= 3:
        with st.spinner("decide → act → … → respond (30 s to 2 min)…"):
            run = common.safe("POST", "/agent/ask", json={"question": question}, timeout=common.LONG)
        if run:
            st.session_state["observe_run"] = run["id"]
    runs = (common.safe("GET", "/agent/runs", params={"limit": 30}) or {}).get("data") or []
    if not runs:
        st.info("No question yet: ask one above.")
        return
    labels = {r["id"]: f"{r['created_at'][:16].replace('T', ' ')} — {r['question'][:80]}" for r in runs}
    ids = list(labels)
    preferred = st.session_state.get("observe_run")
    pick = st.selectbox("Run", ids, format_func=labels.get, index=ids.index(preferred) if preferred in ids else 0,
                        key="observe_run_pick")
    show_run(next(r for r in runs if r["id"] == pick))
