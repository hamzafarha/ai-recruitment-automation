"""Notifications (diagram part 7) and the n8n orchestrator (part 2)."""

import json

import requests
import streamlit as st

from app_pages import common

EVENT_ICONS = {"candidate.created": "👤", "job.created": "💼", "analysis.completed": "🔬", "review.decided": "✅",
               "setup.completed": "⚙️", "digest": "📰", "message.prepared": "✉️", "workflow.error": "⚠️"}


def _n8n_live() -> None:
    """Which workflows are live (their webhooks answer), and their latest runs (with an n8n API key)."""
    status = common.safe("GET", "/observe/n8n", timeout=30) or {}
    if not status.get("reachable"):
        st.warning(f"n8n is not reachable at {common.N8N_URL}: docker compose up -d n8n")
        return
    marks = {True: "🟢 published", False: "⚪ not published: open it in n8n and click Publish", None: "?"}
    st.dataframe([{"Workflow": w["workflow"], "Webhook": w["path"], "State": marks[w["live"]]}
                  for w in status.get("webhooks") or []], hide_index=True)
    if status.get("api_error"):
        st.error(f"n8n API: {status['api_error']}")
    if status.get("executions") is not None:
        st.markdown("**Latest runs** (newest first; open one to see the data of every step in n8n)")
        st.dataframe([{"Workflow": e["workflow"], "Status": e["status"], "Mode": e["mode"],
                       "Started": (e["started"] or "")[:19].replace("T", " "), "Seconds": e["seconds"],
                       "Open in n8n": e["link"]} for e in status["executions"]], hide_index=True,
                     column_config={"Open in n8n": st.column_config.LinkColumn(display_text="open")})
    else:
        st.caption("To list the runs here too: in n8n → Settings → n8n API → Create an API key, put it in .env as "
                   "N8N_API_KEY=… and restart the API. Without it, see them in n8n: open a workflow → Executions.")


def _n8n_panel() -> None:
    st.markdown("#### n8n workflows")
    st.markdown(
        "- **Application pipeline** (`/webhook/cv-intake`): one execution per application, from the CV to the final "
        "decision — intake checks → document processing → candidate JSON (fine-tuned Qwen) → store and index → "
        "answer to the candidate → screening agent → report → recruiters notified → **waits** for the recruiter's "
        "decision → final decision → message to the candidate → closed (see the Application pipeline page).\n"
        "- **New job** (`/webhook/new-job`): a job advertisement → API (structured requirements) → notification.\n"
        "- **Recruitment events** (`/webhook/recruitment-events`): the other app events (analyses and decisions "
        "from the Matching page) → the message to send (interview invitation, rejection letter after the "
        "recruiter's approval) → notification (+ optional e-mail node).\n"
        "- **Daily digest** (every day at 9:00, or `/webhook/daily-digest`): pending reviews and service levels → "
        "notification.\n- **Error alert**: any failed workflow → notification.")
    st.caption(f"n8n editor: {common.N8N_URL} · workflows in `n8n/workflows/` (README, n8n section).")
    _n8n_live()
    c1, c2 = st.columns(2)
    if c1.button("Send a test event to n8n", width="stretch"):
        body = {"event": "test", "title": "Test event from the app", "message": "Checking the n8n connection.",
                "payload": {}, "source": "app"}
        try:
            response = requests.post(f"{common.N8N_URL}/webhook/recruitment-events", json=body, timeout=20)
            if response.ok:
                st.success("n8n received the event: a notification from n8n appears below in a few seconds.")
            else:
                st.error(f"n8n answered {response.status_code}: is the workflow imported and active?")
        except requests.RequestException:
            st.error(f"n8n is not reachable at {common.N8N_URL} (docker compose up -d n8n).")
    if c2.button("Run the daily digest now", width="stretch"):
        try:
            response = requests.post(f"{common.N8N_URL}/webhook/daily-digest", json={}, timeout=60)
            if response.ok:
                st.success("Digest requested: it appears below in a few seconds.")
            else:
                st.error(f"n8n answered {response.status_code}: is the 'Daily digest' workflow active?")
        except requests.RequestException:
            st.error(f"n8n is not reachable at {common.N8N_URL}.")


def render() -> None:
    st.title("🔔 Notifications")
    st.caption("Events of the app (new candidate, analysis ready, recruiter decision…) and the messages prepared by "
               "the n8n workflows (🤖). With N8N_EVENTS_ENABLED=true every app event is also sent to n8n.")
    if not common.need_api():
        return
    c1, c2 = st.columns([3, 1])
    unread_only = c1.toggle("Unread only", value=False)
    if c2.button("Mark all as read", width="stretch"):
        common.safe("POST", "/notifications/read-all")
        common.refresh()
    result = common.safe("GET", "/notifications", params={"unread_only": unread_only, "limit": 100}) or {}
    st.caption(f"{result.get('unread', 0)} unread")
    for n in result.get("data") or []:
        with st.container(border=True):
            icon = "🤖" if n["source"] == "n8n" else EVENT_ICONS.get(n["event"], "🔔")
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"{icon} **{n['title']}**" + ("" if n["read"] else " · :blue-badge[new]"))
            c1.caption(f"{n['created_at'][:16].replace('T', ' ')} · {n['event']} · from {n['source']}")
            if n.get("message"):
                c1.markdown(n["message"])
            if n.get("payload"):
                with c1.expander("Details"):
                    st.code(json.dumps(n["payload"], indent=1, ensure_ascii=False), language="json")
            if not n["read"] and c2.button("Read", key=f"read_{n['id']}"):
                common.safe("POST", f"/notifications/{n['id']}/read")
                st.rerun()
    if not result.get("data"):
        st.caption("No notification yet.")
    st.divider()
    _n8n_panel()
