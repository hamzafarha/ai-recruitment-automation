"""Evaluation results (DSO1-DSO4), read from data/evaluation/ (runs made on Kaggle, downloaded with
``python scripts/kaggle_run.py fetch <job>``)."""

import csv
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "evaluation"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def _pts(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.1f} pts"


def _missing(job: str, what: str) -> None:
    st.info(f"No {what} results on this computer yet. Run it on Kaggle's GPUs, then download it:\n\n"
            f"`python scripts/kaggle_run.py all {job}` (or `fetch {job}` if it already ran).")


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _bars(rows: list[dict], label: str, groups: list[str]) -> None:
    """Grouped horizontal bars on a 0-100 % axis (a truncated axis exaggerates small gaps)."""
    frame = pd.DataFrame(rows)
    st.altair_chart(alt.Chart(frame).mark_bar().encode(
        y=alt.Y("metric:N", sort=list(dict.fromkeys(frame["metric"])), title=None),
        yOffset=alt.YOffset(f"{label}:N", sort=groups),
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title="score (%)"),
        color=alt.Color(f"{label}:N", sort=groups, title=None),
        tooltip=["metric", label, alt.Tooltip("score:Q", format=".1f")]), width="stretch")


# --------------------------------------------------------------------------- DSO1 extraction


def extraction_tab() -> None:
    from recruitment_ai.evaluation.compare import (
        compare_documents,
        compare_groups,
        compare_metrics,
        default_pair,
        list_runs,
        load_run,
    )

    st.caption("Same frozen evaluation CVs (dev split), same code and scoring: only the model differs.")
    root = EVAL_DIR / "extraction"
    names = list_runs(root)
    if not names:
        _missing("extraction", "extraction")
        return
    default_before, default_after = default_pair(names)
    c1, c2 = st.columns(2)
    before_name = c1.selectbox("Before", names, index=names.index(default_before))
    after_options = [n for n in names if n != before_name]
    if not after_options:
        st.info("Only one run so far: the fine-tuned run appears after `python scripts/kaggle_run.py all extraction-ft`.")
        return
    after_name = c2.selectbox("After", after_options, index=after_options.index(default_after)
                              if default_after in after_options else 0)
    before, after = load_run(root, before_name), load_run(root, after_name)
    mb, ma = before["metrics"], after["metrics"]
    rows = compare_metrics(before, after)
    cards = st.columns(4)
    for i, row in enumerate(rows):
        cards[i % 4].metric(row["metric"], _pct(row["after"]), delta=None if row["delta"] is None else _pts(row["delta"]))
    halluc_b = (mb["overall"].get("identity") or {}).get("hallucinations")
    halluc_a = (ma["overall"].get("identity") or {}).get("hallucinations")
    sec_b, sec_a = mb.get("mean_seconds_per_document"), ma.get("mean_seconds_per_document")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documents", ma.get("documents"))
    c2.metric("Errors", ma.get("errors"), delta=(ma.get("errors") or 0) - (mb.get("errors") or 0),
              delta_color="inverse")
    c3.metric("Identity hallucinations", halluc_a, delta_color="inverse",
              delta=halluc_a - halluc_b if halluc_a is not None and halluc_b is not None else None)
    c4.metric("Seconds per CV", sec_a, delta_color="inverse", delta=round(sec_a - sec_b, 1) if sec_a and sec_b else None)
    _bars([{"metric": r["metric"], "run": run, "score": r[key] * 100}
           for r in rows for run, key in (("Before", "before"), ("After", "after")) if r[key] is not None],
          "run", ["Before", "After"])
    groups, docs = st.tabs(["By document type", "Per CV"])
    with groups:
        for group, label in (("input_type", "Input type"), ("source", "Source"), ("language", "Language")):
            st.markdown(f"**{label}** — overall F1")
            st.dataframe([{label: r[group], "Documents": r["documents"], "Before": _pct(r["before"]),
                           "After": _pct(r["after"]), "Change": _pts(r["delta"])}
                          for r in compare_groups(before, after, group)], hide_index=True)
    with docs:
        items = compare_documents(before, after)
        better = sum(1 for d in items if (d["delta"] or 0) > 0.005)
        worse = sum(1 for d in items if (d["delta"] or 0) < -0.005)
        st.write(f"**{better}** CVs improved, **{worse}** got worse, **{len(items) - better - worse}** unchanged.")
        st.dataframe([{"CV": d["doc_id"], "Type": d["input_type"], "Language": d["language"],
                       "Before": _pct(d["before"]), "After": _pct(d["after"]), "Change": _pts(d["delta"]),
                       "Error": "before" if d["error_before"] else ("after" if d["error_after"] else "")}
                      for d in items], hide_index=True)


def finetune_tab() -> None:
    report = EVAL_DIR / "finetune" / "report.md"
    state = _read_json(EVAL_DIR / "finetune" / "training_state.json")
    if not report.exists() and not state:
        _missing("finetune", "fine-tuning")
        return
    if state and state.get("losses"):
        st.markdown("**Training loss** (lower = the model reproduces the reference answers better)")
        st.line_chart(pd.DataFrame(state["losses"]).set_index("step")["loss"])
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- DSO2 retrieval


def rag_tab() -> None:
    metrics = _read_json(EVAL_DIR / "rag" / "metrics.json")
    if not metrics:
        _missing("rag", "retrieval (RAG)")
        return
    st.caption("40 test questions (English and French) on the HR knowledge base: is the right document retrieved? "
               "Hit@k = the expected document is in the top k; MRR = 1 / rank of the first expected document.")
    configs = list(metrics)
    rows = [{"Configuration": name, "Hit@1": _pct(s["all"]["hit@1"]), "Hit@3": _pct(s["all"]["hit@3"]),
             "Hit@5": _pct(s["all"].get("hit@5")), "MRR": f"{s['all']['mrr']:.2f}",
             "Cross-lingual Hit@3": _pct((s.get("cross_lingual") or {}).get("hit@3"))} for name, s in metrics.items()]
    st.dataframe(rows, hide_index=True)
    _bars([{"metric": m, "configuration": name, "score": (metrics[name]["all"][key] or 0) * 100}
           for name in configs for m, key in (("Hit@1", "hit@1"), ("Hit@3", "hit@3"), ("MRR", "mrr"))],
          "configuration", configs)
    report = EVAL_DIR / "rag" / "report.md"
    if report.exists():
        with st.expander("Full report"):
            st.markdown(report.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- DSO3 matching


def matching_tab() -> None:
    metrics = _read_json(EVAL_DIR / "matching" / "metrics.json")
    if not metrics:
        _missing("matching", "matching")
        return
    st.caption(f"{metrics.get('pairs')} graded CV-job pairs (silver labels from the rubric KB-RR-002). The fast "
               "matcher's thresholds are calibrated on dev; the test numbers are the final ones.")
    split = st.radio("Pairs", ["test", "dev", "all"], horizontal=True, key="matching_split")
    rows, bars = [], []
    for name, summary in metrics["systems"].items():
        g, r = summary[split]["grades"], summary[split]["requirements"]
        if not g.get("pairs"):
            continue
        rows.append({"System": name, "Pairs": g["pairs"], "Grade accuracy": _pct(g["accuracy"]),
                     "Within one grade": _pct(g["within_one"]), "MAE": g["mae"], "Quadratic kappa": g["qwk"],
                     "Spearman": g["spearman"], "Must-have status accuracy": _pct(r.get("accuracy")),
                     "Must-have macro-F1": r.get("macro_f1")})
        bars += [{"metric": "Grade accuracy", "system": name, "score": g["accuracy"] * 100},
                 {"metric": "Within one grade", "system": name, "score": g["within_one"] * 100},
                 {"metric": "Must-have status accuracy", "system": name, "score": (r.get("accuracy") or 0) * 100}]
    st.dataframe(rows, hide_index=True)
    if bars:
        _bars(bars, "system", list(dict.fromkeys(b["system"] for b in bars)))
    if metrics.get("thresholds"):
        t = metrics["thresholds"]
        st.caption(f"Calibrated thresholds (used by the app): demonstrated ≥ {t['demonstrated']}, equivalent ≥ "
                   f"{t['equivalent']} ({t.get('calibrated_on')}).")
    pairs_csv = EVAL_DIR / "matching" / "pairs.csv"
    if pairs_csv.exists():
        with pairs_csv.open(encoding="utf-8") as f:
            data = list(csv.DictReader(f))
        with st.expander("Per pair"):
            st.dataframe(data, hide_index=True)
    report = EVAL_DIR / "matching" / "report.md"
    if report.exists():
        with st.expander("Full report"):
            st.markdown(report.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- DSO4 screening agent


def screening_tab() -> None:
    metrics = _read_json(EVAL_DIR / "matching" / "metrics.json") or {}
    systems = metrics.get("systems") or {}
    agent = next((name for name in systems if name.startswith("agent")), None)
    if agent is None:
        _missing("screening", "screening agent")
        st.caption("The job runs the application pipeline's screening agent on the 36 graded pairs of the matching "
                   "evaluation, next to the fixed analysis (same model, same pairs): the difference is the agent.")
        return
    st.caption("The application pipeline's screening agent on the 36 graded CV-job pairs, next to the fixed analysis "
               "(same Qwen2.5-7B, same pairs). The agent retrieves CV evidence and HR rules per requirement, plans "
               "more searches when the evidence is weak, and checks every claim of the model.")
    split = st.radio("Pairs", ["test", "dev", "all"], horizontal=True, key="screening_split")
    rows = []
    for name in [n for n in systems if n.startswith(("llm", "agent", "semantic (calibrated"))]:
        g, r = systems[name][split]["grades"], systems[name][split]["requirements"]
        if g.get("pairs"):
            rows.append({"System": name, "Pairs": g["pairs"], "Grade accuracy": _pct(g["accuracy"]),
                         "Within one grade": _pct(g["within_one"]), "Quadratic kappa": g["qwk"],
                         "Must-have status accuracy": _pct(r.get("accuracy")), "Must-have macro-F1": r.get("macro_f1")})
    st.dataframe(rows, hide_index=True)
    stats = metrics.get("agent") or {}
    if stats:
        c = st.columns(4)
        c[0].metric("Qwen calls / pair", stats["mean_llm_calls"])
        c[1].metric("Tool calls / pair", stats["mean_tool_calls"])
        c[2].metric("Pairs with planning rounds", f"{stats['pairs_with_planning']}/{stats['pairs']}")
        c[3].metric("Claims changed by the checks", stats["checks_applied"])
        st.caption("Checks by kind: " + (", ".join(f"{k}: {v}" for k, v in stats["checks_by_kind"].items())
                                         or "none") + f" · fallbacks to the fast estimate: {stats['fallbacks']}.")


# --------------------------------------------------------------------------- DSO4 assistant


def agent_tab() -> None:
    metrics = _read_json(EVAL_DIR / "agent" / "metrics.json")
    if not metrics:
        _missing("agent", "AI assistant")
        return
    o = metrics["summary"]["overall"]
    st.caption(f"{o['scenarios']} recruiter questions on the demo data, answered by the LangGraph agent with "
               f"{metrics.get('model')}.")
    c = st.columns(5)
    c[0].metric("Task success", _pct(o["success"]))
    c[1].metric("Right first tool", _pct(o["tool_choice"]))
    c[2].metric("Expected answer", _pct(o["answer"]))
    c[3].metric("Grounded names", _pct(o["grounded"]))
    c[4].metric("Seconds / question", o["mean_seconds"])
    refusals = metrics["summary"]["refusals"]
    st.caption(f"Discriminatory requests refused: {_pct(refusals['success'])} ({refusals['scenarios']} scenarios) · "
               f"{o['mean_tool_calls']} tool calls and {o['mean_llm_calls']} model calls per question.")
    st.dataframe([{"Category": k, "Scenarios": v["scenarios"], "Success": _pct(v["success"]),
                   "Tool choice": _pct(v["tool_choice"]), "Answer": _pct(v["answer"])}
                  for k, v in metrics["summary"]["by_category"].items()], hide_index=True)
    report = EVAL_DIR / "agent" / "report.md"
    if report.exists():
        with st.expander("Full report (every scenario, answers of the failed ones)"):
            st.markdown(report.read_text(encoding="utf-8"))


def render() -> None:
    st.title("📊 Evaluation")
    st.caption("Every number comes from a run on Kaggle's GPUs on the frozen evaluation set eval_v1 (no model runs on "
               "our PCs). Dev is used for tuning, test for the final numbers.")
    tabs = st.tabs(["DSO1 · Extraction (before/after fine-tuning)", "DSO1 · Fine-tuning run", "DSO2 · Retrieval (RAG)",
                    "DSO3 · Matching", "DSO4 · Screening agent", "DSO4 · AI assistant"])
    for tab, show in zip(tabs, (extraction_tab, finetune_tab, rag_tab, matching_tab, screening_tab, agent_tab),
                         strict=True):
        with tab:
            show()
