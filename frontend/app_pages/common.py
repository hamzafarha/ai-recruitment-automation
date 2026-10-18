"""Shared helpers of the interface: API calls, status, profile and report display."""

import os
from typing import Any

import requests
import streamlit as st

from recruitment_ai.config.settings import get_settings

SETTINGS = get_settings()
API_URL = os.getenv("BACKEND_API_URL", SETTINGS.backend_api_url).rstrip("/")
V1 = f"{API_URL}{SETTINGS.api_v1_str}"
N8N_URL = os.getenv("N8N_URL", SETTINGS.n8n_url).rstrip("/")
LONG = 900  # seconds: extraction, AI analysis and the assistant take one to a few minutes on the Kaggle GPU
PAGES: dict[str, Any] = {}  # st.Page objects by name, filled by streamlit_app.py (for st.page_link)

GRADE_COLORS = {3: "green", 2: "blue", 1: "orange", 0: "red"}
GRADE_LABELS = {3: "Strong fit", 2: "Partial fit", 1: "Weak fit", 0: "No fit"}
STATUS_LABELS = {"pending_review": "⏳ Pending review", "shortlisted": "✅ Shortlisted", "interview": "📅 Interview",
                 "on_hold": "⏸ On hold", "rejected": "✖ Rejected"}
REQ_STATUS = {"met": "✅ Met", "partial": "🟡 Partial", "missing": "❌ Missing"}
LEVELS = {"demonstrated": "demonstrated", "equivalent": "equivalent", "listed": "listed only",
          "not_evidenced": "not evidenced"}
METHOD_SHORT = {"agent": "AI agent", "llm": "AI", "semantic": "semantic"}
METHOD_LONG = {"agent": "Screening agent (LangGraph: CV evidence and HR rules retrieved per requirement, Qwen2.5-7B "
                        "judgement, every claim checked by code)",
               "llm": "AI analysis (Qwen2.5-7B + HR knowledge, evidence-checked)",
               "semantic": "Fast semantic estimate (BGE-M3 + reranker)"}
APPLICATION_STATUS = {"running": "🔄 Running", "waiting_review": "⏳ Waiting for a decision", "closed": "✅ Closed",
                      "failed": "❌ Stopped at a failed step"}
DECISION_LABELS = {"interview": "📅 Interview", "shortlist": "✅ Shortlisted", "reject": "✖ Rejected"}


class APIError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def call(method: str, path: str, timeout: float = 60, **kwargs: Any) -> Any:
    """One API request; raises APIError with a readable message."""
    try:
        response = requests.request(method, f"{V1}{path}", timeout=timeout, **kwargs)
    except requests.Timeout as exc:
        raise APIError(0, f"No answer after {timeout:.0f} s. The Kaggle AI server may be busy or stopped.") from exc
    except requests.RequestException as exc:
        raise APIError(0, f"The API is not reachable at {API_URL}. Start it with: "
                          "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(f"{'.'.join(map(str, d.get('loc', [])))}: {d.get('msg')}" for d in detail)
        raise APIError(response.status_code, str(detail))
    if "json" in response.headers.get("content-type", ""):
        return response.json()
    return response.content


def safe(method: str, path: str, timeout: float = 60, **kwargs: Any) -> Any:
    """Like ``call``, but shows the error on the page and returns None."""
    try:
        return call(method, path, timeout=timeout, **kwargs)
    except APIError as exc:
        st.error(f"{'Error ' + str(exc.status) + ': ' if exc.status else ''}{exc.detail}")
        return None


def refresh() -> None:
    """After a change: forget cached lists and status."""
    st.cache_data.clear()


@st.cache_data(ttl=30, show_spinner=False)
def health() -> dict[str, Any] | None:
    try:
        return requests.get(f"{API_URL}/api/v1/health", timeout=20).json()
    except (requests.RequestException, ValueError):
        return None


@st.cache_data(ttl=30, show_spinner=False)
def status() -> dict[str, Any] | None:
    try:
        return call("GET", "/admin/status", timeout=25)
    except APIError:
        return None


@st.cache_data(ttl=15, show_spinner=False)
def jobs() -> list[dict[str, Any]]:
    try:
        return call("GET", "/jobs")["data"]
    except APIError:
        return []


@st.cache_data(ttl=15, show_spinner=False)
def candidates() -> list[dict[str, Any]]:
    try:
        return call("GET", "/candidates")["data"]
    except APIError:
        return []


def sidebar() -> None:
    h = health()
    with st.sidebar:
        if h is None:
            st.error("API not reachable")
            st.caption(f"Start it: `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000` ({API_URL})")
            return
        components = h.get("components", {})
        icons = {True: "🟢", False: "🔴"}
        lines = [f"{icons[components.get('postgresql', {}).get('healthy', False)]} PostgreSQL",
                 f"{icons[components.get('qdrant', {}).get('healthy', False)]} Qdrant"]
        if "ai_server" in components:
            lines.append(f"{icons[components['ai_server']['healthy']]} Kaggle AI server")
        if "n8n" in components:
            lines.append(f"{icons[components['n8n']['healthy']]} n8n")
        provider = components.get("llm_provider", {}).get("configured", "?")
        st.caption(" · ".join(lines))
        if provider == "mock":
            st.warning("LLM_PROVIDER=mock: AI answers are fake. Start the Kaggle AI server (Dashboard).", icon="⚠️")
        st.caption(f"[API docs]({API_URL}/docs) · [n8n]({N8N_URL})")


def method_short(method: str | None) -> str:
    return METHOD_SHORT.get(method or "", method or "—")


def submit_application(payload: dict[str, Any], route: str) -> dict[str, Any] | None:
    """A CV into the application pipeline: through n8n (its workflow runs every step, the answer comes after the
    extraction) or straight to the API (the same steps, run by the API in the background). ``payload``: filename,
    content_base64, source, consent, job_id. Returns {"application_id", "message", ...}, or None (error shown)."""
    if route == "n8n":
        try:
            response = requests.post(f"{N8N_URL}/webhook/cv-intake", json=payload, timeout=LONG)
        except requests.RequestException:
            st.error(f"n8n is not reachable at {N8N_URL}: start it (`docker compose up -d n8n`) and publish the "
                     "workflows (`python scripts/n8n_setup.py`), or send the CV straight to the API.")
            return None
        if response.status_code == 404:
            st.error("n8n answered 404: the 'Application pipeline' workflow is not imported or not active "
                     "(`python scripts/n8n_setup.py`).")
            return None
        try:
            data = response.json()
        except ValueError:
            data = None
        if not isinstance(data, dict):
            st.error(f"n8n answered {response.status_code}: {response.text[:300]}")
            return None
        if response.status_code >= 400 or data.get("detail") or not data.get("application_id"):
            st.error(f"The application was not received: {data.get('detail') or data.get('message') or data}")
            return None
        return data
    created = safe("POST", "/applications/submit", json=payload, timeout=120)
    if created is None:
        return None
    return {"application_id": created["id"], "full_name": None, "skills_count": None,
            "message": "Your application was received: the AI reads your CV (about a minute), then the screening "
                       "agent analyses it. A recruiter will answer within 10 working days."}


def need_api() -> bool:
    if health() is None:
        st.error(f"The API is not reachable at {API_URL}. Start it with: "
                 "`python -m uvicorn api.main:app --host 0.0.0.0 --port 8000`")
        return False
    return True


def grade_badge(grade: int | None, label: str | None = None) -> str:
    if grade is None:
        return "—"
    return f":{GRADE_COLORS.get(grade, 'gray')}-badge[Grade {grade} · {label or GRADE_LABELS.get(grade, '')}]"


def short_id(value: str | None) -> str:
    return (value or "")[:8]


def job_label(job: dict[str, Any]) -> str:
    return f"{job['title']}" + (f" — {job['company']}" if job.get("company") else "")


def candidate_label(c: dict[str, Any]) -> str:
    return f"{c.get('full_name') or 'Unnamed'}" + (f" — {c['headline']}" if c.get("headline") else "")


# --------------------------------------------------------------------------- profile


def show_profile(profile: dict[str, Any], show_contact: bool = True) -> None:
    st.subheader(profile.get("full_name") or "Name not found")
    if show_contact:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Email**  \n{profile.get('email') or '—'}")
        c2.markdown(f"**Phone**  \n{profile.get('phone') or '—'}")
        c3.markdown(f"**Location**  \n{profile.get('location') or '—'}")
    if profile.get("summary"):
        st.write(profile["summary"])
    st.markdown("#### Experience")
    rows = [{"Title": j.get("title"), "Company": j.get("company"), "Start": j.get("start_date"),
             "End": j.get("end_date") or ("present" if j.get("is_current") else None),
             "Description": j.get("description")} for j in profile.get("experience") or []]
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.write("—")
    st.markdown("#### Education")
    rows = [{"Degree": e.get("degree"), "Field": e.get("field_of_study"), "Institution": e.get("institution"),
             "End": e.get("end_date")} for e in profile.get("education") or []]
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.write("—")
    st.markdown("#### Projects")
    for project in profile.get("projects") or []:
        tech = f" · *{', '.join(project['technologies'])}*" if project.get("technologies") else ""
        st.markdown(f"- **{project.get('title') or 'Project'}**: {project.get('description') or ''}{tech}")
    if not profile.get("projects"):
        st.write("—")
    st.markdown("#### Skills")
    st.write(", ".join(s["name"] if isinstance(s, dict) else str(s) for s in profile.get("skills") or []) or "—")
    c1, c2 = st.columns(2)
    c1.markdown("#### Languages")
    c1.write(", ".join(x.get("language", "") + (f" ({x['proficiency']})" if x.get("proficiency") else "")
                       for x in profile.get("languages") or [] if isinstance(x, dict)) or "—")
    c2.markdown("#### Certifications")
    c2.write(", ".join(c["name"] if isinstance(c, dict) else str(c) for c in profile.get("certifications") or [])
             or "—")


def extraction_caption(stats: dict[str, Any], filename: str | None = None) -> str:
    parts = [filename] if filename else []
    if stats.get("origin"):
        return stats["origin"]
    parts += [f"input: {stats.get('input_type', '?')}",
              f"model: {stats.get('model', '?')}" + (" + fine-tuned adapter" if stats.get("adapter") else "")]
    if stats.get("seconds"):
        parts.append(f"{stats['seconds']} s generation, {stats.get('parse_seconds', '?')} s reading")
    if stats.get("retried_after_out_of_memory"):
        parts.append("CV shortened (too long for the GPU)")
    return " · ".join(str(p) for p in parts)


# --------------------------------------------------------------------------- report


def show_report(analysis: dict[str, Any]) -> None:
    """The evidence-backed report of an analysis (diagram part 6)."""
    report = analysis.get("report") or {}
    final = analysis.get("final_grade")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grade (0-3)", f"{report.get('grade')} · {report.get('grade_label')}",
              delta=f"adjusted to {final} by the recruiter" if final is not None and final != report.get("grade")
              else None, delta_color="off")
    c2.metric("Score", f"{report.get('score', 0):g} / 100")
    c3.metric("Must-have coverage", f"{(report.get('must_coverage') or 0):.0%}")
    years = report.get("total_years")
    c4.metric("Experience (KB-RR-003)", f"{years:g} years" if years is not None else "unknown")
    st.markdown(f"**Proposed action:** {report.get('action_label', '')}  \n"
                f"**Status:** {STATUS_LABELS.get(analysis.get('status'), analysis.get('status'))}  \n"
                f"**Method:** {METHOD_LONG.get(report.get('method'), report.get('method'))}"
                + (f" · {report['model']}" if report.get("model") else "")
                + (f" · {report['seconds']:g} s" if report.get("seconds") else ""))
    for warning in report.get("warnings") or []:
        st.warning(warning)
    st.info(report.get("summary") or "—")
    st.markdown("#### Requirements and evidence")
    rows = []
    for r in report.get("requirements") or []:
        level = LEVELS.get(r.get("level"), r.get("level"))
        if r.get("min_years") and r.get("years_found") is not None:
            level += f" ({r['years_found']:g} y / {r['min_years']:g} y)"
        evidence = r.get("evidence") or ""
        if evidence and not r.get("grounded", True):
            evidence += " ⚠ not found in the CV"
        rows.append({"Id": r.get("id"), "Requirement": r.get("text"),
                     "Type": ("must" if r.get("kind") == "must" else "nice") + f" · {r.get('importance')}",
                     "Status": REQ_STATUS.get(r.get("status"), r.get("status")), "Evidence level": level,
                     "CV evidence": evidence, "Where": r.get("evidence_source") or "", "Note": r.get("note") or ""})
    st.dataframe(rows, hide_index=True,
                 column_config={"Requirement": st.column_config.TextColumn(width="medium"),
                                "CV evidence": st.column_config.TextColumn(width="large")})
    c1, c2, c3 = st.columns(3)
    for col, title, items in ((c1, "Strengths", report.get("strengths")), (c2, "Gaps to explore", report.get("gaps")),
                              (c3, "Interview questions", report.get("interview_questions"))):
        col.markdown(f"#### {title}")
        col.markdown("\n".join(f"- {item}" for item in items or []) or "—")
    if report.get("knowledge_sources"):
        with st.expander(f"Company rules used ({len(report['knowledge_sources'])} HR knowledge excerpts, RAG)"):
            for k in report["knowledge_sources"]:
                st.markdown(f"**{k.get('doc_id')}** — {k.get('title')}" + (f" / {k['section']}" if k.get("section") else ""))
                st.caption(k.get("excerpt") or "")
    trace = report.get("trace")
    if trace:
        with st.expander("🔬 Under the hood: HR knowledge searches, Qwen's raw answer, checks by code"):
            if trace.get("error"):
                st.error(f"The AI analysis failed: {trace['error']} (the fast semantic estimate is shown)")
            st.caption(f"Qwen: {trace.get('llm_seconds', '—')} s, {trace.get('prompt_tokens') or '—'} prompt tokens → "
                       f"{trace.get('output_tokens') or '—'} answer tokens; prompt of {trace.get('prompt_chars')} "
                       "characters.")
            st.markdown("**HR knowledge searches (RAG):** "
                        + "; ".join(f"“{q['query']}”" for q in trace.get("knowledge_queries") or []))
            checks = trace.get("checks") or {}
            st.markdown("**Checks by code:** quotes not found in the CV → "
                        + (", ".join(checks.get("downgraded_ungrounded") or []) or "none")
                        + " · requirements skipped by the model → "
                        + (", ".join(checks.get("not_assessed") or []) or "none"))
            st.code(trace.get("raw_answer") or "(not available)", language="json", wrap_lines=True)
        agent = trace.get("agent")
        if agent:
            show_agent_trace(agent, (trace.get("checks") or {}).get("verification") or [])
        if "observe_matching" in PAGES:
            st.page_link(PAGES["observe_matching"], label="See every step of this analysis", icon="🧩")
    st.caption("Decision support only: a recruiter makes every decision (KB-RR-001). \"Not evidenced\" means the CV "
               "does not mention it, not that the candidate lacks it.")


def show_agent_trace(agent: dict[str, Any], changes: list[dict[str, Any]]) -> None:
    """What the screening agent did for one job: its searches, its planning rounds, the evidence it kept and
    what its checks changed in the model's judgement."""
    with st.expander(f"🤖 Screening agent: {len(agent.get('tool_calls') or [])} tool calls, "
                     f"{agent.get('planning_rounds', 0)} planning round(s) by Qwen, {len(changes)} check(s) applied"):
        if agent.get("considered"):
            st.markdown("**Open jobs graded for this spontaneous application** (fast semantic matching)")
            st.dataframe([{"Job": c["title"], "Fast grade": f"{c['grade']} · {c['grade_label']}", "Score": c["score"]}
                          for c in agent["considered"]], hide_index=True)
        for plan in agent.get("plans") or []:
            st.markdown(f"**Planning round {plan['round']}** ({plan['by']}): {plan.get('thought') or '—'}")
            if plan.get("calls"):
                st.caption(" · ".join(f"{c['tool']}({c.get('requirement') or ''}: “{c['query']}”)"
                                      for c in plan["calls"]))
            if plan.get("dropped"):
                st.caption("Refused by code: " + " · ".join(f"“{d['query']}” ({d['why']})" for d in plan["dropped"]))
        calls = agent.get("tool_calls") or []
        if calls:
            st.dataframe([{"Round": c.get("round"), "Tool": c.get("tool"), "Requirement": c.get("requirement") or "",
                           "Query": c.get("query"), "Found": c.get("found"), "Best relevance": c.get("best"),
                           "HR documents": ", ".join(c.get("documents") or [])} for c in calls], hide_index=True,
                         column_config={"Query": st.column_config.TextColumn(width="large")})
        evidence = agent.get("evidence") or {}
        if evidence:
            st.markdown("**CV passages kept for each requirement** (reranked, best first)")
            st.dataframe([{"Req.": rid, "Where": h.get("label"), "Relevance": h.get("score"), "Passage": h.get("text")}
                          for rid, hits in evidence.items() for h in hits], hide_index=True,
                         column_config={"Passage": st.column_config.TextColumn(width="large")})
        st.markdown("**Checks by code on the model's judgement**")
        if changes:
            st.dataframe([{"Req.": c["id"], "Model said": c["from"], "Kept": c["to"], "Why": c["why"]}
                          for c in changes], hide_index=True,
                         column_config={"Why": st.column_config.TextColumn(width="large")})
        else:
            st.caption("No change: every claim passed the checks.")


def download_buttons(analysis_id: str, key: str = "") -> None:
    c1, c2 = st.columns(2)
    try:
        md = call("GET", f"/analyses/{analysis_id}/report.md")
        c1.download_button("⬇ Report (Markdown)", md, file_name=f"report_{short_id(analysis_id)}.md",
                           mime="text/markdown", key=f"md_{key}{analysis_id}", width="stretch")
    except APIError as exc:
        c1.error(exc.detail)
    try:
        pdf = call("GET", f"/analyses/{analysis_id}/report.pdf", timeout=60)
        c2.download_button("⬇ Report (PDF)", pdf, file_name=f"report_{short_id(analysis_id)}.pdf",
                           mime="application/pdf", key=f"pdf_{key}{analysis_id}", width="stretch")
    except APIError as exc:
        c2.error(exc.detail)
