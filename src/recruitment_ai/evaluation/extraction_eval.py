"""Score an extracted CandidateProfile against its answer key (DSO1).

Per CV, per section:
- identity (name, email, phone, location): correct / wrong / missed / hallucinated per field;
- experience, education: predicted entries are paired with answer-key entries (greedy, by
  text similarity), giving precision / recall / F1; matched jobs also get date accuracy;
- skills: set precision / recall / F1, two skills being equal when the EN/FR lexicon maps them
  to the same concept ("ReactJS" = "React") or their texts are near-identical.
Counts are kept (not only ratios) so that results can be summed over CVs (micro-averages).
"""

import difflib
import re
from collections.abc import Callable
from typing import Any

from recruitment_ai.data.normalization import text_key

MATCH_THRESHOLD = 0.6  # minimum similarity to pair two job / education entries
SKILL_SIMILARITY = 0.85  # minimum text similarity for two skills without a lexicon concept


def similarity(a: str | None, b: str | None) -> float:
    """0..1 similarity of two short texts (case and punctuation insensitive, containment = 0.9)."""
    ka, kb = text_key(a or ""), text_key(b or "")
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    if min(len(ka), len(kb)) >= 4 and (ka in kb or kb in ka):
        return 0.9
    return difflib.SequenceMatcher(None, ka, kb).ratio()


def _digits(value: str | None) -> str:
    return re.sub(r"\D", "", value or "")


# --------------------------------------------------------------------------- identity


def _identity_equal(field: str, gold: str, pred: str) -> bool:
    if field == "phone":
        g, p = _digits(gold), _digits(pred)
        return len(g) >= 7 and (g[-9:] == p[-9:] or g.endswith(p) or p.endswith(g))
    if field == "email":
        return gold.strip().lower() == pred.strip().lower()
    if field == "location":
        return similarity(gold, pred) >= 0.8
    return similarity(gold, pred) >= 0.85


def score_identity(gold: dict[str, Any], pred: dict[str, Any]) -> dict[str, int]:
    counts = {"correct": 0, "wrong": 0, "missed": 0, "hallucinated": 0, "gold_fields": 0}
    for field in ("full_name", "email", "phone", "location"):
        g, p = gold.get(field), pred.get(field)
        if g:
            counts["gold_fields"] += 1
            if not p:
                counts["missed"] += 1
            elif _identity_equal(field, g, p):
                counts["correct"] += 1
            else:
                counts["wrong"] += 1
        elif p:
            counts["hallucinated"] += 1
    return counts


# --------------------------------------------------------------------------- entries


def pair_entries(gold: list[dict], pred: list[dict], score: Callable[[dict, dict], float]) -> list[tuple[int, int]]:
    """Greedy one-to-one pairing of entries, best similarity first, above MATCH_THRESHOLD."""
    candidates = sorted(
        ((score(g, p), gi, pi) for gi, g in enumerate(gold) for pi, p in enumerate(pred)), reverse=True
    )
    used_g, used_p, pairs = set(), set(), []
    for value, gi, pi in candidates:
        if value < MATCH_THRESHOLD:
            break
        if gi not in used_g and pi not in used_p:
            used_g.add(gi)
            used_p.add(pi)
            pairs.append((gi, pi))
    return pairs


def _job_score(g: dict, p: dict) -> float:
    title = similarity(g.get("title"), p.get("title"))
    if g.get("company") and p.get("company"):
        return 0.5 * title + 0.5 * similarity(g["company"], p["company"])
    return title * 0.9  # one side has no company: rely on the title only, slightly penalised


def _edu_score(g: dict, p: dict) -> float:
    parts = [similarity(g.get(k), p.get(k)) for k in ("institution", "degree") if g.get(k) and p.get(k)]
    return sum(parts) / len(parts) if parts else 0.0


def _same_year(a: str | None, b: str | None) -> bool:
    return bool(a and b and a[:4] == b[:4])


def score_entries(gold: list[dict], pred: list[dict], kind: str) -> dict[str, int]:
    pairs = pair_entries(gold, pred, _job_score if kind == "experience" else _edu_score)
    counts = {"gold": len(gold), "pred": len(pred), "matched": len(pairs)}
    if kind == "experience":
        counts.update(dates=0, dates_exact=0, dates_year=0)
        for gi, pi in pairs:
            for key in ("start_date", "end_date"):
                g, p = gold[gi].get(key), pred[pi].get(key)
                if g:
                    counts["dates"] += 1
                    counts["dates_exact"] += g == p
                    counts["dates_year"] += g == p or _same_year(g, p)
    return counts


# --------------------------------------------------------------------------- skills


def _skill_keys(name: str, normalizer: Any | None) -> tuple[str | None, str]:
    concept = normalizer.normalize(name) if normalizer is not None else None
    return (concept.id if concept else None), text_key(name)


def score_skills(gold: list[dict], pred: list[dict], normalizer: Any | None) -> dict[str, int]:
    g_keys = [_skill_keys(s["name"], normalizer) for s in gold]
    p_keys = [_skill_keys(s["name"], normalizer) for s in pred]

    def same(a: tuple[str | None, str], b: tuple[str | None, str]) -> bool:
        if a[0] and b[0]:
            return a[0] == b[0]
        return a[1] == b[1] or difflib.SequenceMatcher(None, a[1], b[1]).ratio() >= SKILL_SIMILARITY

    matched_g = sum(any(same(g, p) for p in p_keys) for g in g_keys)
    matched_p = sum(any(same(p, g) for g in g_keys) for p in p_keys)
    technical = [k for k, s in zip(g_keys, gold, strict=True) if s.get("category") in ("Technical", "Tool")]
    return {
        "gold": len(gold), "pred": len(pred), "gold_found": matched_g, "pred_correct": matched_p,
        "gold_technical": len(technical), "technical_found": sum(any(same(g, p) for p in p_keys) for g in technical),
    }


# --------------------------------------------------------------------------- whole CV


def score_profile(gold: dict[str, Any], pred: dict[str, Any], normalizer: Any | None = None,
                  score_identity_fields: bool = True) -> dict[str, dict[str, int]]:
    """All section counts for one CV (``score_identity_fields=False`` for template identities)."""
    result = {
        "experience": score_entries(gold.get("experience", []), pred.get("experience", []), "experience"),
        "education": score_entries(gold.get("education", []), pred.get("education", []), "education"),
        "skills": score_skills(gold.get("skills", []), pred.get("skills", []), normalizer),
    }
    if score_identity_fields:
        result["identity"] = score_identity(gold, pred)
    return result


def _prf(correct_pred: int, pred: int, found_gold: int, gold: int) -> dict[str, float | None]:
    precision = correct_pred / pred if pred else None
    recall = found_gold / gold if gold else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else (0.0 if gold else None)
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize(scores: list[dict[str, dict[str, int]]]) -> dict[str, Any]:
    """Micro-averaged metrics over a list of per-CV score dicts."""

    def total(section: str, key: str) -> int:
        return sum(s[section][key] for s in scores if section in s)

    out: dict[str, Any] = {"cvs": len(scores)}
    for kind in ("experience", "education"):
        out[kind] = _prf(total(kind, "matched"), total(kind, "pred"), total(kind, "matched"), total(kind, "gold"))
    dates = total("experience", "dates")
    out["experience"]["date_exact"] = total("experience", "dates_exact") / dates if dates else None
    out["experience"]["date_year"] = total("experience", "dates_year") / dates if dates else None
    out["skills"] = _prf(total("skills", "pred_correct"), total("skills", "pred"),
                         total("skills", "gold_found"), total("skills", "gold"))
    tech = total("skills", "gold_technical")
    out["skills"]["technical_recall"] = total("skills", "technical_found") / tech if tech else None
    identity_cvs = [s for s in scores if "identity" in s]
    gold_fields = total("identity", "gold_fields")
    out["identity"] = {
        "cvs": len(identity_cvs),
        "accuracy": total("identity", "correct") / gold_fields if gold_fields else None,
        "hallucinations": total("identity", "hallucinated"),
    }
    f1s = [out[k]["f1"] for k in ("experience", "education", "skills") if out[k]["f1"] is not None]
    out["overall_f1"] = sum(f1s) / len(f1s) if f1s else None
    return out
