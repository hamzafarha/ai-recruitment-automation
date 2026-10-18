"""Under the hood, part 8: the PostgreSQL tables and their rows."""

import json

import streamlit as st

from app_pages import common

DESCRIPTIONS = {
    "candidates": "One row per candidate: the extracted profile (JSON), the CV text used as evidence, extraction details.",
    "jobs": "One row per job: its requirements (JSON: M1.. must-haves, N1.. nice-to-haves).",
    "analyses": "One row per candidate analysed for a job: grade, score, status and the full report (JSON, with the "
                "AI trace).",
    "reviews": "One row per recruiter decision (human validation).",
    "notifications": "One row per event of the app or message written by n8n.",
    "agent_runs": "One row per question to the AI assistant, with its tool steps and timeline (JSON).",
}


def _short(value: object) -> object:
    """A table cell: JSON shown as a short summary (open the row to read it whole)."""
    if isinstance(value, dict):
        return f"{{…}} {len(value)} keys"
    if isinstance(value, list):
        return f"[…] {len(value)} items"
    return value


def render() -> None:
    st.title("🗄️ Database (PostgreSQL)")
    st.caption("Part 8 of the architecture: the system's records. The same tables can be opened with Adminer "
               "(`docker compose --profile tools up -d adminer` → http://localhost:8080) or psql "
               "(docs/reports/observe_the_data.md).")
    if not common.need_api():
        return
    overview = common.safe("GET", "/observe/tables")
    if not overview:
        return
    st.dataframe([{"Table": t["table"], "Rows": t["rows"], "Holds": DESCRIPTIONS.get(t["table"], ""),
                   "Columns": ", ".join(t["columns"])} for t in overview["tables"]], hide_index=True,
                 column_config={"Holds": st.column_config.TextColumn(width="large")})
    if overview.get("migration"):
        st.caption(f"Database schema version (Alembic): `{overview['migration']}`")
    c1, c2, c3 = st.columns([2, 1, 1])
    name = c1.selectbox("Table", [t["table"] for t in overview["tables"]], key="db_table")
    limit = c2.number_input("Rows", 5, 200, 20, step=5, key="db_limit")
    page = c3.number_input("Page", 1, 1000, 1, key="db_page")
    result = common.safe("GET", f"/observe/tables/{name}", params={"limit": limit, "offset": (page - 1) * limit})
    if not result:
        return
    st.caption(f"{result['total']} rows in `{name}`, newest first. Types: "
               + ", ".join(f"{c['name']} ({c['type']})" for c in result["columns"]))
    if not result["rows"]:
        st.info("No row on this page.")
        return
    st.dataframe([{k: _short(v) for k, v in row.items()} for row in result["rows"]], hide_index=True)
    labels = {i: " · ".join(str(row.get(k)) for k in ("full_name", "title", "question", "event", "decision", "id")
                            if row.get(k))[:120] for i, row in enumerate(result["rows"])}
    pick = st.selectbox("Open a row", list(labels), format_func=labels.get, key=f"db_row_{name}")
    row = result["rows"][pick]
    for column, value in row.items():
        if isinstance(value, dict | list):
            with st.expander(f"{column} (JSON)"):
                st.json(value, expanded=1)
        elif isinstance(value, str) and len(value) > 120:
            with st.expander(f"{column} (text)"):
                st.text(value)
    with st.expander("The whole row (JSON)"):
        st.code(json.dumps(row, indent=1, ensure_ascii=False, default=str), language="json")
