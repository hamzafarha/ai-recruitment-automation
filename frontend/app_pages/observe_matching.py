"""Under the hood, part 5: what feeds an evidence-backed analysis — semantic evidence per requirement,
the HR rules retrieved (RAG), the contextualized prompt — and what Qwen answered."""

import json

import streamlit as st

from app_pages import common

LEVEL_ICONS = {"demonstrated": "🟢 demonstrated", "equivalent": "🟡 equivalent", "listed": "🟠 listed only",
               "not_evidenced": "⚪ not evidenced"}


def _semantic(result: dict) -> None:
    t = result["thresholds"]
    st.subheader("1. Evidence found in the CV (fast semantic matching, no LLM)")
    st.caption(f"The CV is cut into {result['snippet_count']} snippets (positions, projects, degrees, skills…). For "
               "each requirement, the closest snippets by BGE-M3 cosine are rescored by the BGE reranker; a strong "
               f"score (≥ {t['demonstrated']}) inside a position or project = demonstrated, inside a skills list = "
               f"listed; a weaker one (≥ {t['equivalent']}) in a position or project = equivalent.")
    st.dataframe([{"Id": r["id"], "Requirement": r["text"],
                   "Type": ("must" if r["kind"] == "must" else "nice") + f" · {r['importance']}",
                   "Level": LEVEL_ICONS.get(r["level"], r["level"]), "Status": common.REQ_STATUS.get(r["status"]),
                   "Years": r["years_found"]} for r in result["requirements"]], hide_index=True,
                 column_config={"Requirement": st.column_config.TextColumn(width="large")})
    g = result["semantic_grade"]
    st.caption(f"Fast estimate: grade {g['grade']}, must-have coverage {g['must_coverage']:.0%}, nice-to-have "
               f"coverage {g['nice_coverage']:.0%} (rubric KB-RR-002).")
    pick = st.selectbox("See the snippets compared with one requirement",
                        [r["id"] for r in result["requirements"]],
                        format_func=lambda i: next(f"{r['id']}: {r['text'][:90]}" for r in result["requirements"]
                                                   if r["id"] == i), key="matching_requirement")
    chosen = next(r for r in result["requirements"] if r["id"] == pick)
    st.dataframe([{"Snippet": s["where"], "Section": s["section"], "Cosine (BGE-M3)": s["cosine"],
                   "Reranker score": s["rerank"], "Text": s["text"]} for s in chosen["snippets"]], hide_index=True,
                 column_config={"Text": st.column_config.TextColumn(width="large")})


def _knowledge(result: dict) -> None:
    st.subheader("2. Company rules retrieved from the HR knowledge base (RAG)")
    st.caption("Three searches per job (BGE-M3 → Qdrant → BGE reranker). The best 2 results of each go into the "
               "prompt: " + (", ".join(result["rules_in_prompt"]) or "none (knowledge base not indexed?)") + ".")
    for k in result["knowledge"]:
        st.markdown(f"**“{k['query']}”**" + (f" — filtered on the job profile `{k['job_profile']}`"
                                             if k["job_profile"] else ""))
        if k.get("error"):
            st.error(k["error"])
        st.dataframe([{"Document": h["doc_id"], "Section": h["section"], "Reranker score": h["score"],
                       "Text": h["text"]} for h in k["hits"]], hide_index=True,
                     column_config={"Text": st.column_config.TextColumn(width="large")})


def _prompt(result: dict) -> None:
    st.subheader("3. Contextualization: the prompt Qwen receives")
    st.caption(f"{result['prompt_chars']} characters: the job's requirements, the retrieved rules, the experience "
               "timeline computed by code (KB-RR-003) and the CV text. Then the JSON schema of the answer is added.")
    with st.expander("System prompt (how to judge)"):
        st.code(result["system_prompt"], language="text", wrap_lines=True)
    with st.expander("User message (the context)"):
        st.code(result["prompt"], language="text", wrap_lines=True)


def _analysis(candidate_id: str, job_id: str, semantic: dict | None) -> None:
    st.subheader("4. What Qwen answered, and the checks made by code")
    found = (common.safe("GET", "/analyses", params={"candidate_id": candidate_id, "job_id": job_id}) or {}).get("data")
    llm = [a for a in found or [] if a["method"] in ("llm", "agent")]  # the newest first
    if st.button("🔬 Run the full AI analysis now (about a minute)"):
        with st.spinner("Retrieval, prompt, Qwen on the Kaggle GPU, checks…"):
            created = common.safe("POST", "/analyses", json={"candidate_id": candidate_id, "job_id": job_id,
                                                             "method": "llm"}, timeout=common.LONG)
        if created:
            common.refresh()
            st.rerun()
    if not llm:
        st.info("No AI analysis of this pair yet: run it with the button above.")
        return
    analysis = common.safe("GET", f"/analyses/{llm[0]['id']}")
    report = (analysis or {}).get("report") or {}
    trace = report.get("trace") or {}
    if not trace:
        st.info("This analysis was made before traces were recorded: run it again to see its internals.")
        return
    c = st.columns(4)
    c[0].metric("Qwen time", f"{trace.get('llm_seconds')} s")
    c[1].metric("Prompt tokens", trace.get("prompt_tokens") or "—")
    c[2].metric("Answer tokens", trace.get("output_tokens") or "—")
    c[3].metric("Grade (rubric)", f"{report.get('grade')} · {report.get('grade_label')}")
    checks = trace.get("checks") or {}
    if checks.get("downgraded_ungrounded"):
        st.warning("Quotes not found in the CV → counted as only listed: " + ", ".join(checks["downgraded_ungrounded"]))
    else:
        st.success("Every quote given by Qwen was found in the CV.")
    if checks.get("not_assessed"):
        st.warning("Requirements the model skipped: " + ", ".join(checks["not_assessed"]))
    if trace.get("agent"):  # made by the application pipeline's screening agent
        common.show_agent_trace(trace["agent"], checks.get("verification") or [])
    fast = {r["id"]: r["level"] for r in (semantic or {}).get("requirements", [])}
    st.dataframe([{"Id": r["id"], "Fast semantic": LEVEL_ICONS.get(fast.get(r["id"]), "—"),
                   "AI analysis (final)": LEVEL_ICONS.get(r["level"], r["level"]), "Status": common.REQ_STATUS.get(r["status"]),
                   "CV quote": r.get("evidence"), "Found in CV": "yes" if r.get("grounded", True) else "no"}
                  for r in report.get("requirements") or []], hide_index=True,
                 column_config={"CV quote": st.column_config.TextColumn(width="large")})
    with st.expander("Qwen's raw answer, exactly as written"):
        raw = trace.get("raw_answer") or ""
        try:
            raw = json.dumps(json.loads(raw[raw.find("{"):]), indent=2, ensure_ascii=False)
        except ValueError:
            pass
        st.code(raw or "(not available)", language="json")
    with st.expander("The prompt this analysis used"):
        st.code(trace.get("prompt") or "", language="text", wrap_lines=True)


def render() -> None:
    st.title("🧩 Matching internals")
    st.caption("Part 5 of the architecture: retrieval, contextualization and LLM analysis for one candidate and one "
               "job. Steps 1–3 run without the LLM (seconds); step 4 shows a real analysis by Qwen2.5-7B.")
    if not common.need_api():
        return
    jobs, candidates = common.jobs(), common.candidates()
    if not jobs or not candidates:
        st.info("Load the demo data first (Dashboard → Set up).")
        return
    job_labels = {j["id"]: common.job_label(j) for j in jobs}
    candidate_labels = {c["id"]: common.candidate_label(c) for c in candidates}
    c1, c2 = st.columns(2)
    job_id = c1.selectbox("Job", list(job_labels), format_func=job_labels.get, key="observe_job")
    candidate_id = c2.selectbox("Candidate", list(candidate_labels), format_func=candidate_labels.get,
                                key="observe_candidate")
    key = f"{candidate_id}|{job_id}"
    if st.button("▶ Show what the analysis uses", type="primary"):
        with st.spinner("Semantic matching, HR knowledge retrieval, prompt…"):
            result = common.safe("POST", "/observe/matching", json={"candidate_id": candidate_id, "job_id": job_id},
                                 timeout=180)
        if result:
            st.session_state["observe_matching"] = {key: result}
    result = (st.session_state.get("observe_matching") or {}).get(key)
    if result:
        _semantic(result)
        _knowledge(result)
        _prompt(result)
    else:
        st.info("Press the button to see steps 1–3 for this pair.")
    st.divider()
    _analysis(candidate_id, job_id, result)
