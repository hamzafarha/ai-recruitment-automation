"""The application pipeline (every part of the diagram): ONE workflow per application, from the CV
received to the final decision. Each application moves through the steps; every step shows what ran
(models, tools, timings) and the screening agent shows its path through its LangGraph graph."""

import base64

import streamlit as st

from app_pages import common

COLORS = {"done": "#d1fadf", "running": "#dbeafe", "waiting": "#fff4d6", "failed": "#fde2e1", "pending": "#eef1f5"}
ICONS = {"done": "✅", "running": "🔄", "waiting": "⏳", "failed": "❌", "pending": "▫️"}
AGENT_COLORS = {"understand": "#e4ecf5", "plan": "#e4ecf5", "act": "#fff4d6", "evaluate": "#f3e8ff",
                "analyse": "#e4ecf5", "verify": "#d1fadf", "next": "#eef1f5"}


def _dot(text: object, limit: int = 48) -> str:
    return str(text).replace("\\", "/").replace('"', "'")[:limit]


def flow_graph(steps: list[dict]) -> str:
    """The workflow, each step colored by its state for this application."""
    lines = ["digraph {", "  rankdir=LR; bgcolor=transparent; nodesep=0.2; ranksep=0.25;",
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#9aa5b1"];']
    for i, s in enumerate(steps):
        status = s.get("status", "pending")
        label = f"{_dot(s['label'])}\\npart {s['part']}"
        if status == "done" and s.get("seconds") is not None:
            label += f"\\n{s['seconds']:g} s"
        elif status in ("waiting", "running", "failed"):
            label += f"\\n{status}"
        lines.append(f'  s{i} [label="{label}", fillcolor="{COLORS.get(status, COLORS["pending"])}"];')
        if i:
            lines.append(f"  s{i - 1} -> s{i};")
    return "\n".join(lines + ["}"])


def _agent_detail(e: dict) -> str:
    node = e["node"]
    if node == "understand":
        return f"{e.get('task')}: {', '.join(e.get('jobs') or [])}"
    if node == "plan":
        return f"round {e.get('round')} by {e.get('by')}: {e.get('calls')} calls"
    if node == "act":
        return f"{e.get('cv_searches')} CV + {e.get('hr_searches')} HR searches"
    if node == "evaluate":
        return f"weak: {', '.join(e.get('weak') or []) or 'none'} → {e.get('next')}"
    if node == "analyse":
        return "failed → fast estimate" if e.get("error") else "Qwen judgement"
    if node == "verify":
        return f"{e.get('changes')} change(s) → grade {e.get('grade')}"
    return f"→ {e.get('next')}"


def agent_graph(timeline: list[dict]) -> str:
    """The screening agent's path: every node run, in order (diagram steps 1-7)."""
    lines = ["digraph {", "  rankdir=LR; bgcolor=transparent;",
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=9, color="#9aa5b1"];',
             '  s [shape=circle, label="start", fillcolor="#d1fadf"];']
    previous = "s"
    for i, e in enumerate(timeline, 1):
        node = e["node"]
        detail = _agent_detail(e)
        color = "#fde2e1" if e.get("error") else AGENT_COLORS.get(node, "#eef1f5")
        seconds = f"\\n{e['seconds']} s" if e.get("seconds") else ""
        lines.append(f'  n{i} [label="{i}. {node}\\n{_dot(detail)}{seconds}", fillcolor="{color}"];')
        lines.append(f"  {previous} -> n{i};")
        previous = f"n{i}"
    lines += ['  e [shape=doublecircle, label="end", fillcolor="#d1fadf"];', f"  {previous} -> e;", "}"]
    return "\n".join(lines)


def _show_screen(result: dict) -> None:
    st.markdown(f"**Task:** {result.get('task')} · {result.get('llm_calls')} Qwen calls · "
                f"{result.get('tool_calls')} tool calls · {result.get('passages')} CV passages "
                f"(from the {'CV text' if result.get('passages_from') == 'cv_text' else 'profile'}) · "
                f"{result.get('seconds')} s · graph: {result.get('graph')}")
    if result.get("jobs_considered"):
        st.markdown("**1. Understand the task** — open jobs graded by the fast semantic matching:")
        st.dataframe([{"Job": c["title"], "Fast grade": f"{c['grade']} · {c['grade_label']}", "Score": c["score"]}
                      for c in result["jobs_considered"]], hide_index=True)
    st.dataframe([{"Job analysed": j["title"], "Grade": f"{j['grade']} · {j['grade_label']}", "Score": j["score"],
                   "Method": common.method_short(j["method"]), "Planning rounds by Qwen": j.get("planning_rounds"),
                   "Claims changed by the checks": j.get("checks_changed")} for j in result.get("jobs_analysed") or []],
                 hide_index=True)
    timeline = result.get("timeline") or []
    if timeline:
        st.graphviz_chart(agent_graph(timeline), width="stretch")
        st.dataframe([{"#": i, "Diagram step": e.get("step"), "Node": e["node"], "Job": e.get("job") or "",
                       "What happened": e.get("why") or e.get("thought") or e.get("error") or "",
                       "Seconds": e.get("seconds"),
                       "Tokens in → out": f"{e['prompt_tokens']} → {e['output_tokens']}" if e.get("prompt_tokens") else ""}
                      for i, e in enumerate(timeline, 1)], hide_index=True,
                     column_config={"What happened": st.column_config.TextColumn(width="large")})
    if result.get("tools"):
        with st.expander(f"Every tool call ({len(result['tools'])})"):
            st.dataframe([{"Job": c.get("job") or "", "Round": c.get("round"), "Tool": c.get("tool"),
                           "Requirement": c.get("requirement") or "", "Query": c.get("query") or c.get("result"),
                           "Found": c.get("found"), "Best relevance": c.get("best")} for c in result["tools"]],
                         hide_index=True, column_config={"Query": st.column_config.TextColumn(width="large")})
    for warning in result.get("warnings") or []:
        st.warning(warning)


def _show_step(step: dict) -> None:
    result = step.get("result") or {}
    name = step["step"]
    if step.get("error"):
        st.error(step["error"])
    if step.get("note"):
        st.info(step["note"])
    if name == "screen" and result:
        _show_screen(result)
    elif name == "document" and result:
        st.markdown(f"**{result.get('engine')}** · {result.get('input_type')} · {result.get('pages')} page(s), "
                    f"{result.get('ocr_pages')} by OCR · {result.get('characters')} characters · {result.get('seconds')} s")
        st.text(result.get("preview") or "")
    elif name == "extract" and result:
        st.markdown(f"**{result.get('full_name') or 'Name not found'}** — {result.get('experience')} positions, "
                    f"{result.get('education')} degrees, {result.get('projects')} projects, {result.get('skills')} "
                    f"skills · model {result.get('model')}"
                    + (" + fine-tuned QLoRA adapter" if result.get("adapter") else " (no adapter)")
                    + f" · {result.get('seconds')} s · {result.get('prompt_tokens')} → {result.get('output_tokens')} "
                      "tokens")
    elif name in ("acknowledge", "notify_candidate") and result.get("text"):
        st.markdown(f"**To:** {result.get('to') or '(no e-mail in the CV)'} · **Subject:** {result.get('subject')}")
        st.text(result["text"])
    elif name == "report" and result:
        for a in result.get("analyses") or []:
            st.markdown(f"- **{a['job_title']}** — {common.grade_badge(a['grade'], a['grade_label'])} · "
                        f"{a['score']:g}/100 · {a.get('action_label') or ''}")
        st.page_link(common.PAGES["reports"], label="Open the reports", icon="📄")
    elif result:
        st.json(result, expanded=False)


def _details(application_id: str, was_running: bool) -> None:
    app = common.safe("GET", f"/applications/{application_id}")
    if not app:
        return
    if was_running and app["status"] != "running":
        st.rerun()  # finished: refresh the whole page (and stop refreshing)
    st.markdown(f"### {app.get('candidate_name') or app.get('filename')} — {app.get('applied_for') or ''}")
    st.markdown(f"{common.APPLICATION_STATUS.get(app['status'], app['status'])} · step "
                f"{app['steps_done']}/{app['steps_total']} done · run by **{app['orchestrator']}**"
                + (f" · decision: {common.DECISION_LABELS.get(app['decision'], app['decision'])}" if app.get("decision")
                   else "") + (f" · grade {app['grade']}, {app['score']:g}/100" if app.get("grade") is not None else ""))
    st.graphviz_chart(flow_graph(app["steps"]), width="stretch")
    if app["status"] == "failed":
        st.error(f"Stopped at «{app['stage_label']}»: {app.get('error')}")
        if st.button("▶ Continue the workflow from this step (with the API)", type="primary", key="continue_app"):
            with st.spinner("Running the next steps (a few minutes when the screening agent runs)…"):
                if common.safe("POST", f"/applications/{application_id}/continue", timeout=common.LONG):
                    st.rerun()
    elif app["status"] == "waiting_review":
        st.info("The workflow waits for a recruiter's decision on the analysis: decide in Human validation; the "
                "decision resumes this application's workflow (final decision → message to the candidate → closed).")
        st.page_link(common.PAGES["review"], label="Decide in Human validation", icon="✅")
    for step in app["steps"]:
        status = step.get("status", "pending")
        title = (f"{ICONS.get(status, '·')} {step['label']} — part {step['part']}"
                 + (f" · {step['seconds']:g} s" if step.get("seconds") is not None else ""))
        with st.expander(title, expanded=status in ("failed", "running") or (step["step"] == "screen"
                                                                           and status == "done")):
            st.caption(step.get("what") or "")
            _show_step(step)


def _submit_form() -> None:
    jobs = [j for j in common.jobs() if j["status"] == "open"]
    labels = {"": "No specific job (the agent finds the open jobs that fit)"} | {j["id"]: common.job_label(j)
                                                                                 for j in jobs}
    with st.form("pipeline_submit"):
        c1, c2 = st.columns([2, 1])
        job_id = c1.selectbox("Job", list(labels), format_func=labels.get)
        route = c2.radio("Run by", ["n8n", "api"], format_func={"n8n": "n8n (one execution)",
                                                                 "api": "the API (if n8n is stopped)"}.get)
        upload = st.file_uploader("CV (PDF, Word or image)", type=["pdf", "docx", "png", "jpg", "jpeg"])
        if st.form_submit_button("Submit the CV", type="primary") and upload is not None:
            payload = {"filename": upload.name, "content_base64": base64.b64encode(upload.getvalue()).decode(),
                       "source": "upload", "consent": True, "job_id": job_id or None}
            with st.spinner("Intake, document processing, extraction by the fine-tuned Qwen, storage (about a "
                            "minute)…" if route == "n8n" else "Registering the application…"):
                result = common.submit_application(payload, route)
            if result:
                common.refresh()
                st.session_state["pipeline_pick"] = result["application_id"]
                st.rerun()


def render() -> None:
    st.title("🔄 Application pipeline")
    st.caption("One workflow per application, from the CV to the final decision: intake → document processing → "
               "candidate JSON by the fine-tuned Qwen → storage and search index → acknowledgement → screening agent "
               "(LangGraph) → evidence-backed report → recruiters notified → human validation → final decision → "
               "message to the candidate → closed. n8n runs it (workflow “Application pipeline”, one execution per "
               "application) or, when n8n is stopped, the API runs the same steps.")
    if not common.need_api():
        return
    with st.expander("➕ Submit a CV to the pipeline"):
        _submit_form()
    data = common.safe("GET", "/applications") or {}
    applications = data.get("data") or []
    if not applications:
        st.info("No application yet: submit a CV above, in Candidates or in the Candidate portal.")
        return
    st.dataframe([{"Candidate": a.get("candidate_name") or a.get("filename"), "Applied for": a.get("applied_for"),
                   "Status": common.APPLICATION_STATUS.get(a["status"], a["status"]), "Step": a["stage_label"],
                   "Done": f"{a['steps_done']}/{a['steps_total']}",
                   "Grade": a.get("grade"), "Decision": common.DECISION_LABELS.get(a.get("decision"), a.get("decision")),
                   "Run by": a["orchestrator"], "Received": a["created_at"][:16].replace("T", " ")}
                  for a in applications], hide_index=True)
    labels = {a["id"]: f"{a.get('candidate_name') or a.get('filename')} — "
                       f"{common.APPLICATION_STATUS.get(a['status'], a['status'])} — {a['created_at'][:16].replace('T', ' ')}"
              for a in applications}
    ids = list(labels)
    preferred = st.session_state.get("pipeline_pick")
    pick = st.selectbox("Open an application", ids, format_func=labels.get,
                        index=ids.index(preferred) if preferred in ids else 0, key="pipeline_select")
    running = next(a for a in applications if a["id"] == pick)["status"] == "running"
    st.fragment(_details, run_every=5 if running else None)(pick, running)
    with st.expander("The steps and what each one runs"):
        steps = (common.safe("GET", "/applications/steps") or {}).get("data") or []
        st.dataframe([{"Step": s["label"], "Part": s["part"], "What runs": s["what"]} for s in steps],
                     hide_index=True, column_config={"What runs": st.column_config.TextColumn(width="large")})
        st.caption("Fine-tuning is not a step: it ran once on Kaggle (QLoRA, 873 CVs); the “Candidate JSON” step "
                   f"uses the adapter it produced. The workflow's n8n webhook: {common.N8N_URL}/webhook/cv-intake")
