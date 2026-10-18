"""Matching evaluation (DSO3): the system's grades against the 36 graded CV-job pairs of eval_v1.

Reference labels (labels/matches.csv): a grade 0-3 per pair and a status (met / partial / missing)
per must-have, made by applying the written rubric KB-RR-002 ("silver" labels, see the datasheet).
Three systems are evaluated on the same pairs:
- ``semantic``: fast matching (BGE-M3 + BGE reranker + skill lexicon + rubric). Its two evidence
  thresholds are calibrated on the dev pairs only (grid search), then applied unchanged to test;
- ``llm``: the fixed full analysis (HR knowledge retrieval + Qwen2.5-7B judgement + excerpt check +
  rubric), the Matching page's analysis;
- ``agent``: the application pipeline's screening agent (DSO4): CV evidence and HR rules retrieved per
  requirement, planning rounds by Qwen for weak evidence, Qwen judgement, every claim checked by code.
  Its checks use the dev-calibrated thresholds; nothing else is tuned on these pairs.
Metrics, per split:
- grades: exact accuracy, within-one accuracy, mean absolute error, quadratic weighted kappa,
  Spearman correlation;
- must-have statuses: accuracy and macro-F1 over met / partial / missing.
"""

import json
import math
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from recruitment_ai.evaluation.eval_set import read_rows, synthetic_split
from recruitment_ai.matching.evidence import profile_text

GRADES = (0, 1, 2, 3)
STATUSES = ("met", "partial", "missing")
SPLIT_DIRS = {"dev": "validation", "test": "test"}
DEMONSTRATED_GRID = [round(0.15 + 0.05 * i, 2) for i in range(14)]   # 0.15 .. 0.80
EQUIVALENT_GRID = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]


# --------------------------------------------------------------------------- data


def load_pairs(eval_dir: Path) -> list[dict[str, Any]]:
    """Graded pairs with their reference grade and must-have statuses (reference annotator only)."""
    summary_path = eval_dir / "labels" / "summary.json"
    primary = json.loads(summary_path.read_text(encoding="utf-8")).get("primary_annotator") \
        if summary_path.exists() else None
    labels = {r["pair_id"]: r for r in read_rows(eval_dir / "labels" / "matches.csv")
              if not primary or r["annotator"] == primary}
    pairs = []
    for row in read_rows(eval_dir / "labels" / "pairs.csv"):
        label = labels.get(row["pair_id"])
        if label is None or label.get("grade") in (None, ""):
            continue
        pairs.append({**row, "grade": int(label["grade"]), "reason": label.get("reason") or "",
                      "requirements": json.loads(label.get("requirements") or "{}")})
    return pairs


def load_jobs(eval_dir: Path) -> dict[str, dict[str, Any]]:
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((eval_dir / "jobs").glob("*.json"))]
    return {job["role"]: job for job in jobs}


def load_candidates(pairs: list[dict[str, Any]], eval_dir: Path, interim_dir: Path, texts: dict[str, str],
                    parse_text: Callable[[Path], str | None] | None = None) -> dict[str, dict[str, Any]]:
    """{cv_id: {"id", "name", "profile" (answer key), "cv_text"}} for every CV of the pairs.

    The profile is the CV's answer key (so matching is evaluated without extraction errors); the
    text is the CV's own text (the dataset's text for real CVs, the document's text for synthetic
    ones), which the LLM analysis quotes and the excerpt check verifies.
    """
    pool = {r["resume_id"]: r["eval_split"] for r in read_rows(eval_dir / "cv_pool.csv")}
    out: dict[str, dict[str, Any]] = {}
    for pair in pairs:
        cv_id = pair["cv_id"]
        if cv_id in out:
            continue
        if pair["cv_source"] == "synthetic":
            profile = json.loads((eval_dir / "synthetic" / "profiles" / f"{cv_id}.json")
                                 .read_text(encoding="utf-8"))["profile"]
            document = eval_dir / "synthetic" / "files" / f"{cv_id}.pdf"
            text = parse_text(document) if parse_text and document.exists() else None
        else:
            split = pool.get(cv_id) or pair["eval_split"]
            profile = json.loads((interim_dir / SPLIT_DIRS[split] / f"{cv_id}.json").read_text(encoding="utf-8"))
            text = texts.get(cv_id)
        out[cv_id] = {"id": cv_id, "name": profile.get("full_name") or cv_id, "profile": profile,
                      "cv_text": text or profile_text(profile)}
    return out


def split_of(pair: dict[str, Any]) -> str:
    return pair.get("eval_split") or synthetic_split(pair["cv_id"])


# --------------------------------------------------------------------------- metrics


def quadratic_kappa(gold: list[int], pred: list[int], labels: Iterable[int] = GRADES) -> float | None:
    """Cohen's kappa with quadratic weights: 1 = perfect agreement, 0 = chance level."""
    labels = list(labels)
    k, n = len(labels), len(gold)
    if n == 0:
        return None
    index = {label: i for i, label in enumerate(labels)}
    observed = [[0.0] * k for _ in range(k)]
    for g, p in zip(gold, pred, strict=True):
        observed[index[g]][index[p]] += 1
    gold_hist = [sum(row) for row in observed]
    pred_hist = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            weight = (i - j) ** 2 / (k - 1) ** 2
            num += weight * observed[i][j]
            den += weight * gold_hist[i] * pred_hist[j] / n
    return None if den == 0 else round(1 - num / den, 4)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for m in range(i, j + 1):
            ranks[order[m]] = (i + j) / 2 + 1  # ties share their average rank
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return None if sx == 0 or sy == 0 else round(cov / (sx * sy), 4)


def grade_metrics(gold: list[int], pred: list[int], scores: list[float] | None = None) -> dict[str, Any]:
    n = len(gold)
    if n == 0:
        return {"pairs": 0}
    diffs = [abs(g - p) for g, p in zip(gold, pred, strict=True)]
    confusion = Counter(f"{g}->{p}" for g, p in zip(gold, pred, strict=True))
    return {"pairs": n, "accuracy": round(sum(d == 0 for d in diffs) / n, 4),
            "within_one": round(sum(d <= 1 for d in diffs) / n, 4), "mae": round(sum(diffs) / n, 4),
            "qwk": quadratic_kappa(gold, pred),
            "spearman": spearman([float(g) for g in gold], [float(s) for s in (scores or pred)]),
            "confusion": dict(sorted(confusion.items()))}


def status_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    n = len(gold)
    if n == 0:
        return {"requirements": 0}
    per_status = {}
    for status in STATUSES:
        tp = sum(g == p == status for g, p in zip(gold, pred, strict=True))
        fp = sum(p == status and g != status for g, p in zip(gold, pred, strict=True))
        fn = sum(g == status and p != status for g, p in zip(gold, pred, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_status[status] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                              "support": sum(g == status for g in gold)}
    present = [s for s in STATUSES if per_status[s]["support"] or any(p == s for p in pred)]
    return {"requirements": n, "accuracy": round(sum(g == p for g, p in zip(gold, pred, strict=True)) / n, 4),
            "macro_f1": round(sum(per_status[s]["f1"] for s in present) / len(present), 4) if present else None,
            "per_status": per_status}


# --------------------------------------------------------------------------- scoring of reports


def pair_rows(pairs: list[dict[str, Any]], reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per pair that has a report: reference vs predicted grade and must-have statuses."""
    rows = []
    for pair in pairs:
        report = reports.get(pair["pair_id"])
        if report is None:
            continue
        predicted = {r["id"]: r for r in report.get("requirements") or []}
        requirements = [{"id": rid, "gold": status, "pred": (predicted.get(rid) or {}).get("status", "missing"),
                         "level": (predicted.get(rid) or {}).get("level"),
                         "evidence": (predicted.get(rid) or {}).get("evidence")}
                        for rid, status in pair["requirements"].items()]
        rows.append({"pair_id": pair["pair_id"], "split": split_of(pair), "job_role": pair["job_role"],
                     "cv_id": pair["cv_id"], "cv_source": pair["cv_source"], "gold_grade": pair["grade"],
                     "pred_grade": int(report["grade"]), "score": float(report["score"]),
                     "requirements": requirements, "warnings": report.get("warnings") or [],
                     "seconds": report.get("seconds")})
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("dev", "test", "all"):
        chosen = [r for r in rows if split == "all" or r["split"] == split]
        gold_status = [q["gold"] for r in chosen for q in r["requirements"]]
        pred_status = [q["pred"] for r in chosen for q in r["requirements"]]
        out[split] = {"grades": grade_metrics([r["gold_grade"] for r in chosen], [r["pred_grade"] for r in chosen],
                                              [r["score"] for r in chosen]),
                      "requirements": status_metrics(gold_status, pred_status)}
    return out


def objective(summary_split: dict[str, Any]) -> float:
    """Calibration target on dev: must-have status macro-F1 + grade accuracy (+ within-one as a tie-break)."""
    grades, reqs = summary_split["grades"], summary_split["requirements"]
    return (reqs.get("macro_f1") or 0.0) + (grades.get("accuracy") or 0.0) + 0.25 * (grades.get("within_one") or 0.0)


# --------------------------------------------------------------------------- report


def _change_kind(why: str) -> str:
    if "does not support" in why:
        return "quote not supporting the requirement"
    if "skills section" in why:
        return "skill only listed"
    if "did not use" in why:
        return "evidence the model left out"
    if "years from" in why:
        return "years counted from the positions"
    return "other"


def agent_summary(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """What the screening agent did on the pairs: model and tool calls, planning rounds, checks applied."""
    n = len(reports) or 1
    runs = [r.get("agent_run") or {} for r in reports]
    traces = [r.get("trace") or {} for r in reports]
    rounds = [(t.get("agent") or {}).get("planning_rounds") or 0 for t in traces]
    changes = [c for t in traces for c in (t.get("checks") or {}).get("verification") or []]
    return {"pairs": len(reports),
            "mean_llm_calls": round(sum(r.get("llm_calls") or 0 for r in runs) / n, 2),
            "mean_tool_calls": round(sum(r.get("tool_calls") or 0 for r in runs) / n, 2),
            "mean_planning_rounds": round(sum(rounds) / n, 2),
            "pairs_with_planning": sum(r > 0 for r in rounds),
            "checks_applied": len(changes),
            "checks_by_kind": dict(Counter(_change_kind(c.get("why") or "") for c in changes)),
            "pairs_with_checks": sum(bool((t.get("checks") or {}).get("verification")) for t in traces),
            "fallbacks": sum(any("AI analysis failed" in w for w in r.get("warnings") or []) for r in reports)}


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def report_markdown(results: dict[str, dict[str, Any]], thresholds: dict[str, Any] | None,
                    pairs: list[dict[str, Any]], notes: list[str] | None = None,
                    agent: dict[str, Any] | None = None) -> str:
    """``results``: {system name: {"summary": summarize(...), "rows": [...], "seconds": ...}}."""
    lines = ["# Matching evaluation (DSO3, and the screening agent: DSO4)", "",
             f"{len(pairs)} graded CV-job pairs of eval_v1 (dev {sum(split_of(p) == 'dev' for p in pairs)}, "
             f"test {sum(split_of(p) == 'test' for p in pairs)}). Reference: silver labels made by applying the "
             "rubric KB-RR-002 (grade 0-3 and met / partial / missing per must-have). Candidate profiles are the "
             "answer keys, so this measures matching alone, without extraction errors.", ""]
    if thresholds:
        lines += ["**Semantic thresholds calibrated on dev** (reranker probability): demonstrated >= "
                  f"{thresholds['demonstrated']}, equivalent >= {thresholds['equivalent']}.", ""]
    for split, title in (("test", "Test pairs (final numbers)"), ("dev", "Dev pairs (used for calibration)"),
                         ("all", "All pairs")):
        lines += [f"## {title}", "",
                  "| System | Pairs | Grade accuracy | Within one grade | MAE | Quadratic kappa | Spearman | "
                  "Must-have status accuracy | Must-have macro-F1 |", "|---|---|---|---|---|---|---|---|---|"]
        for name, result in results.items():
            s = result["summary"][split]
            g, r = s["grades"], s["requirements"]
            if not g.get("pairs"):
                continue
            lines.append(f"| {name} | {g['pairs']} | {_pct(g['accuracy'])} | {_pct(g['within_one'])} | "
                         f"{_num(g['mae'])} | {_num(g['qwk'])} | {_num(g['spearman'])} | {_pct(r.get('accuracy'))} | "
                         f"{_num(r.get('macro_f1'))} |")
        lines.append("")
    for name, result in results.items():
        lines += [f"## Per pair — {name}", "", "| Pair | Split | Job | CV | Reference | Predicted | Score | "
                  "Must-haves agreeing |", "|---|---|---|---|---|---|---|---|"]
        for r in result["rows"]:
            agree = sum(q["gold"] == q["pred"] for q in r["requirements"])
            lines.append(f"| {r['pair_id']} | {r['split']} | {r['job_role']} | {r['cv_id']} | {r['gold_grade']} | "
                         f"{r['pred_grade']} | {r['score']:g} | {agree}/{len(r['requirements'])} |")
        if result.get("seconds") is not None:
            lines.append(f"\nAverage time per pair: {result['seconds']:.1f} s.")
        lines.append("")
    if agent:
        kinds = ", ".join(f"{k}: {v}" for k, v in sorted(agent["checks_by_kind"].items())) or "none"
        lines += ["## Screening agent (DSO4): what it did", "",
                  f"- {agent['pairs']} pairs; per pair {agent['mean_llm_calls']} Qwen calls and "
                  f"{agent['mean_tool_calls']} tool calls on average",
                  f"- planning rounds by Qwen (weak evidence after the systematic first round): "
                  f"{agent['mean_planning_rounds']} per pair, on {agent['pairs_with_planning']} pairs",
                  f"- checks by code that changed the model's judgement: {agent['checks_applied']} on "
                  f"{agent['pairs_with_checks']} pairs ({kinds})",
                  f"- analyses that fell back to the fast estimate: {agent['fallbacks']}",
                  "- Compare the **agent** line with the **llm** line above: same model, same pairs; the "
                  "difference is the agent (retrieval per requirement, iterations, checks).", ""]
    lines += ["## How to read", "",
              "- **Within one grade** matters most for screening: a grade-2 candidate predicted 3 (or 1) still "
              "reaches the recruiter's review; 0 vs 3 would not.",
              "- **Quadratic kappa** penalises big disagreements more than small ones (1 = perfect, 0 = chance).",
              "- The labels are silver (model-made with the written rubric): treat differences of a few points "
              "as noise; adding human labels on the same pairs is the next step."]
    if notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"
