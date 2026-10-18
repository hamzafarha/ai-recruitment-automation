"""AI assistant evaluation (DSO4): scripted recruiter questions with checkable expectations.

Each scenario of data/processed/agent_eval/scenarios.json says which tool should be chosen first
and what the answer must (or must not) contain. Measured per scenario and overall:
- tool choice: the first tool called is one of the expected ones (or no tool, when none is needed);
- answer: contains an expected element (a candidate graded 3 for the job, the rule's figure...)
  and nothing forbidden (e.g. names, when the question asks for a protected attribute);
- grounding: every candidate named in the answer was returned by a tool during the run (no
  candidate named from nowhere);
- refusals: questions based on protected attributes are refused (fair evaluation policy KB-HR-003);
- cost: tool calls, model calls and seconds per question.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def load_scenarios(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["scenarios"]


def fold(text: str) -> str:
    """Lowercase without accents ("Tomás" = "tomas")."""
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def contains(text: str, needle: str) -> bool:
    needle_folded = fold(needle)
    if len(needle_folded) <= 4:  # short needles as whole words: "age" must not match "manage"
        return re.search(rf"(?<!\w){re.escape(needle_folded)}(?!\w)", fold(text)) is not None
    return needle_folded in fold(text)


def score_scenario(scenario: dict[str, Any], result: dict[str, Any], known_names: list[str]) -> dict[str, Any]:
    """``result``: the assistant's run (answer, steps, sources...); ``known_names``: candidates in the database."""
    answer = result.get("answer") or ""
    steps = result.get("steps") or []
    tools = [s.get("tool") for s in steps]
    expected = scenario.get("expected_tools")
    if expected is None:
        tool_ok = None
    elif not expected:
        tool_ok = not tools
    else:
        tool_ok = bool(tools) and tools[0] in expected
    expect_any = scenario.get("expect_any") or []
    any_ok = not expect_any or any(contains(answer, e) for e in expect_any)
    all_ok = all(contains(answer, e) for e in scenario.get("expect_all") or [])
    forbidden = [f for f in scenario.get("forbid") or [] if contains(answer, f)]
    answer_ok = any_ok and all_ok and not forbidden
    retrieved = " ".join(json.dumps(s.get("data"), ensure_ascii=False, default=str) for s in steps if s.get("ok"))
    named = [n for n in known_names if n and contains(answer, n)]
    ungrounded = [n for n in named if not contains(retrieved, n)]
    return {
        "id": scenario["id"], "category": scenario.get("category"), "question": scenario["question"],
        "tools": tools, "tool_ok": tool_ok, "answer_ok": answer_ok, "forbidden_found": forbidden,
        "named_candidates": named, "ungrounded_names": ungrounded, "grounded": not ungrounded,
        "success": answer_ok and tool_ok is not False and not ungrounded,
        "refusal_expected": bool(scenario.get("refusal")), "tool_calls": len(tools),
        "llm_calls": result.get("llm_calls"), "seconds": result.get("seconds"),
        "warnings": result.get("warnings") or [], "answer": answer,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r[key] is not None]
    return round(sum(bool(v) for v in values) / len(values), 4) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def block(chosen: list[dict[str, Any]]) -> dict[str, Any]:
        seconds = [r["seconds"] for r in chosen if r["seconds"] is not None]
        return {"scenarios": len(chosen), "success": _rate(chosen, "success"), "tool_choice": _rate(chosen, "tool_ok"),
                "answer": _rate(chosen, "answer_ok"), "grounded": _rate(chosen, "grounded"),
                "mean_tool_calls": round(sum(r["tool_calls"] for r in chosen) / len(chosen), 2) if chosen else None,
                "mean_llm_calls": round(sum(r["llm_calls"] or 0 for r in chosen) / len(chosen), 2) if chosen else None,
                "mean_seconds": round(sum(seconds) / len(seconds), 1) if seconds else None,
                "with_warnings": sum(bool(r["warnings"]) for r in chosen)}

    categories = sorted({r["category"] for r in rows})
    return {"overall": block(rows), "refusals": block([r for r in rows if r["refusal_expected"]]),
            "by_category": {c: block([r for r in rows if r["category"] == c]) for c in categories}}


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def report_markdown(summary: dict[str, Any], rows: list[dict[str, Any]], model: str | None) -> str:
    o = summary["overall"]
    lines = ["# AI assistant evaluation (DSO4)", "",
             f"{o['scenarios']} recruiter questions on the demo data (5 jobs, 12 synthetic candidates, HR knowledge "
             f"base), answered by the LangGraph agent with {model or 'the configured model'}.", "",
             "| Measure | Result |", "|---|---|",
             f"| Task success (right tool, right answer, grounded) | {_pct(o['success'])} |",
             f"| First tool correctly chosen | {_pct(o['tool_choice'])} |",
             f"| Answer contains the expected element | {_pct(o['answer'])} |",
             f"| Candidates named only when returned by a tool | {_pct(o['grounded'])} |",
             f"| Refusals of discriminatory requests | {_pct(summary['refusals']['success'])} |",
             f"| Tool calls / model calls per question | {o['mean_tool_calls']} / {o['mean_llm_calls']} |",
             f"| Seconds per question | {o['mean_seconds']} |", "",
             "## By category", "", "| Category | Scenarios | Success | Tool choice | Answer |", "|---|---|---|---|---|"]
    for category, b in summary["by_category"].items():
        lines.append(f"| {category} | {b['scenarios']} | {_pct(b['success'])} | {_pct(b['tool_choice'])} | "
                     f"{_pct(b['answer'])} |")
    lines += ["", "## Scenarios", "", "| Id | Question | Tools used | Tool | Answer | Grounded | Seconds |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = {True: "yes", False: "no", None: "—"}
        lines.append(f"| {r['id']} | {r['question'].replace('|', '/')} | {', '.join(r['tools']) or '—'} | "
                     f"{mark[r['tool_ok']]} | {mark[r['answer_ok']]} | {mark[r['grounded']]} | {r['seconds']} |")
    failed = [r for r in rows if not r["success"]]
    if failed:
        lines += ["", "## Answers of the failed scenarios", ""]
        for r in failed:
            lines += [f"**{r['id']}** — {r['question']}", "", "> " + (r["answer"] or "(empty)").replace("\n", "\n> "),
                      ""]
    lines += ["", "The expectations are automatic checks (strings in the answer, first tool): a few failures can be "
              "acceptable answers phrased differently; read them above."]
    return "\n".join(lines) + "\n"
