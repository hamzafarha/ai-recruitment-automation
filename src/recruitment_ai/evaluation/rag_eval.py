"""Evaluate knowledge retrieval (Step 3, DSO2) on the test questions of eval_v1/rag/queries.json.

Retrieved chunks are collapsed into a ranked list of documents (doc_id), because the useful
question is "did the right document come back?". Metrics per question:
- hit@k     : at least one expected document in the top k documents
- recall@k  : share of the expected documents in the top k
- reciprocal rank : 1 / rank of the first expected document (0 if absent within the retrieved list)
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

KS = (1, 3, 5)

CONFIGS = {  # name -> (use the question's role as filter, rerank)
    "dense": (False, False),
    "dense+rerank": (False, True),
    "dense+role_filter": (True, False),
    "dense+role_filter+rerank": (True, True),
}


def load_queries(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["queries"]


def ranked_documents(results: list[Any]) -> list[str]:
    """Unique document ids in rank order (several chunks of one document count once)."""
    seen: list[str] = []
    for r in results:
        doc_id = r.metadata.get("document_id")
        if doc_id and doc_id not in seen:
            seen.append(doc_id)
    return seen


def query_metrics(ranked: list[str], relevant: list[str]) -> dict[str, float]:
    found = [d for d in ranked if d in relevant]
    first = next((i for i, d in enumerate(ranked, 1) if d in relevant), None)
    metrics = {"rr": 1 / first if first else 0.0}
    for k in KS:
        top = ranked[:k]
        metrics[f"hit@{k}"] = float(any(d in relevant for d in top))
        metrics[f"recall@{k}"] = sum(d in top for d in relevant) / len(relevant)
    metrics["found_any"] = float(bool(found))
    return metrics


def mean(rows: list[dict[str, float]], key: str) -> float | None:
    values = [r[key] for r in rows]
    return sum(values) / len(values) if values else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["rr"] + [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS]
    return {"queries": len(rows), **{("mrr" if k == "rr" else k): mean(rows, k) for k in keys}}


def evaluate(search: Callable[..., list[Any]], queries: list[dict[str, Any]], doc_languages: dict[str, str],
             top_chunks: int = 10) -> dict[str, Any]:
    """Run every configuration; ``search(query, role=None, rerank=bool, top_k=int)`` returns chunks."""
    results: dict[str, Any] = {}
    for name, (use_role, rerank) in CONFIGS.items():
        rows = []
        for q in queries:
            hits = search(q["query"], role=q["role"] if use_role else None, rerank=rerank, top_k=top_chunks)
            ranked = ranked_documents(hits)
            cross = any(doc_languages.get(d, q["language"]) != q["language"] for d in q["relevant"])
            rows.append({"id": q["id"], "language": q["language"], "cross_lingual": cross,
                         "has_role": q["role"] is not None, "ranked": ranked[:5],
                         **query_metrics(ranked, q["relevant"])})
        groups = {
            "all": rows,
            "language=en": [r for r in rows if r["language"] == "en"],
            "language=fr": [r for r in rows if r["language"] == "fr"],
            "cross_lingual": [r for r in rows if r["cross_lingual"]],
            "role-specific questions": [r for r in rows if r["has_role"]],
        }
        results[name] = {"summary": {g: summarize(v) for g, v in groups.items() if v}, "rows": rows}
    return results


def report(results: dict[str, Any], queries: list[dict[str, Any]], chunk_count: int) -> str:
    fmt = lambda v: "—" if v is None else f"{v:.0%}"  # noqa: E731
    lines = [
        "# Retrieval evaluation — HR knowledge base", "",
        f"{len(queries)} questions · {chunk_count} indexed chunks · BGE-M3 dense retrieval, "
        "optional role filter, optional BGE-reranker-v2-m3", "",
        "| Configuration | Group | Questions | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, result in results.items():
        for group, s in result["summary"].items():
            lines.append(f"| {name} | {group} | {s['queries']} | {fmt(s['hit@1'])} | {fmt(s['hit@3'])} | "
                         f"{fmt(s['hit@5'])} | {fmt(s['recall@5'])} | {fmt(s['mrr'])} |")
    best = max(results, key=lambda n: results[n]["summary"]["all"]["mrr"])
    misses = [r for r in results[best]["rows"] if not r["hit@3"]]
    lines += ["", f"Best configuration (MRR): **{best}**.", ""]
    if misses:
        lines.append(f"Questions where **{best}** misses the expected document in the top 3:")
        by_id = {q["id"]: q for q in queries}
        for r in misses:
            q = by_id[r["id"]]
            lines.append(f"- {r['id']} ({q['language']}) \"{q['query']}\" — expected {q['relevant']}, got {r['ranked'][:3]}")
    return "\n".join(lines) + "\n"
