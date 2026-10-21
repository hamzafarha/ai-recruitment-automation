"""Observe each part of the architecture while it works (the interface's "Under the hood" pages).

Read-only views of what the pipeline stores and does, part by part of the diagram:
- ``overview``: live state of the 9 parts (and of the application pipeline);
- ``tables`` / ``table_rows``: the PostgreSQL rows (part 8);
- ``collections`` / ``points`` / ``point`` / ``vector_map``: the Qdrant collections, their points and
  a 2D map of their vectors (parts 5-6);
- ``explain_retrieval``: one search step by step: BGE-M3 vector, Qdrant hits, BGE reranker (part 5);
- ``explain_matching``: the evidence of every requirement, the HR rules retrieved and the exact
  prompt the AI analysis sends to Qwen (part 5, contextualization);
- ``explain_extraction``: a CV page by page, the prompt, Qwen's raw answer, and the base model vs
  the fine-tuned adapter (parts 3-4);
- ``n8n_status``: which workflows are live, and their executions (parts 2, 7);
- ``ai_server``: the models loaded on the Kaggle AI server and its activity.
Nothing here changes data, except ``explain_extraction``, which runs the model (nothing is stored).
"""

import math
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from recruitment_ai.database.connection import Base
from recruitment_ai.database.models import (
    Analysis,
    Application,
    Candidate,
    Notification,
)
from recruitment_ai.database.repositories import CandidateRepository, JobRepository
from recruitment_ai.rag.knowledge_index import COLLECTION as KNOWLEDGE
from recruitment_ai.rag.search_index import CANDIDATES, JOBS
from recruitment_ai.services.errors import NotFoundError

TABLES = ["applications", "candidates", "jobs", "analyses", "reviews", "notifications", "emails", "agent_runs"]
COLLECTIONS = [KNOWLEDGE, CANDIDATES, JOBS]
MAX_TEXT = 600  # long text cells are cut in table views
WEBHOOKS = {"cv-intake": "Recruitment — Application pipeline", "new-job": "Recruitment — New job",
            "recruitment-events": "Recruitment — Events", "daily-digest": "Recruitment — Daily digest"}
COLOR_BY = {KNOWLEDGE: "document_id", CANDIDATES: "name", JOBS: "title"}


def _check_collection(name: str) -> None:
    if name not in COLLECTIONS:
        raise NotFoundError(f"Unknown collection {name!r}: use one of {', '.join(COLLECTIONS)}")


def _label(collection: str, payload: dict[str, Any]) -> str:
    """A readable name for a point."""
    if collection == KNOWLEDGE:
        return f"{payload.get('document_id', '?')} — {payload.get('section', '')}"
    if collection == CANDIDATES:
        return f"{payload.get('name') or 'Candidate'} — {payload.get('label', '')}"
    return f"{payload.get('title', 'Job')} — {payload.get('section', '')}"


# --------------------------------------------------------------------------- AI server and n8n


def ai_server(settings: Any) -> dict[str, Any]:
    """Models served by the Kaggle AI server and its activity (only when the app uses it)."""
    if "remote" not in (settings.llm_provider, settings.embedding_provider):
        return {"used": False}
    from recruitment_ai.llm.remote_client import AIServerClient

    try:
        info = AIServerClient(settings, timeout=8).request("GET", "/info", timeout=8)
    except RuntimeError as exc:
        return {"used": True, "reachable": False, "error": str(exc)}
    keys = ("model", "vision", "adapter", "embedding_model", "embedding_dimension", "reranking_model",
            "running_jobs", "requests", "started", "last_request")
    return {"used": True, "reachable": True, **{k: info.get(k) for k in keys}}


def _webhook_live(base: str, path: str) -> bool | None:
    """A GET on a POST webhook: n8n says "not registered for GET requests" when the workflow is live,
    and "The requested webhook ... is not registered" when it is not. Nothing is triggered."""
    try:
        response = httpx.get(f"{base}/webhook/{path}", timeout=3)
    except httpx.HTTPError:
        return None
    if response.status_code < 300:
        return True
    try:
        message = str(response.json().get("message", ""))
    except ValueError:
        message = response.text
    return not ("The requested webhook" in message and "is not registered" in message)


def n8n_status(settings: Any) -> dict[str, Any]:
    base = settings.n8n_url.rstrip("/")
    out: dict[str, Any] = {"url": base, "events_enabled": settings.n8n_events_enabled,
                           "api_key": bool(settings.n8n_api_key)}
    try:
        out["reachable"] = httpx.get(f"{base}/healthz", timeout=3).status_code == 200
    except httpx.HTTPError:
        out["reachable"] = False
    if not out["reachable"]:
        return out
    out["webhooks"] = [{"workflow": name, "path": f"/webhook/{path}", "live": _webhook_live(base, path)}
                       for path, name in WEBHOOKS.items()]
    if settings.n8n_api_key:  # optional: the n8n public API lists the executions
        headers = {"X-N8N-API-KEY": settings.n8n_api_key}
        try:
            workflows = httpx.get(f"{base}/api/v1/workflows", headers=headers, params={"limit": 100},
                                  timeout=10).json().get("data", [])
            executions = httpx.get(f"{base}/api/v1/executions", headers=headers, params={"limit": 30},
                                   timeout=10).json().get("data", [])
        except (httpx.HTTPError, ValueError) as exc:
            out["api_error"] = f"{type(exc).__name__}: {exc}"[:300]
            return out
        names = {w.get("id"): w.get("name") for w in workflows}
        out["workflows"] = [{"id": w.get("id"), "name": w.get("name"), "active": w.get("active"),
                             "updated": w.get("updatedAt")} for w in workflows]
        out["executions"] = []
        for e in executions:
            seconds = None
            if e.get("startedAt") and e.get("stoppedAt"):
                started = datetime.fromisoformat(e["startedAt"].replace("Z", "+00:00"))
                stopped = datetime.fromisoformat(e["stoppedAt"].replace("Z", "+00:00"))
                seconds = round((stopped - started).total_seconds(), 2)
            out["executions"].append({
                "id": e.get("id"), "workflow": names.get(e.get("workflowId"), e.get("workflowId")),
                "status": e.get("status") or ("success" if e.get("finished") else "error"), "mode": e.get("mode"),
                "started": e.get("startedAt"), "seconds": seconds,
                "link": f"{base}/workflow/{e.get('workflowId')}/executions/{e.get('id')}"})
    return out


# --------------------------------------------------------------------------- overview


def overview(db: Session, container: Any) -> dict[str, Any]:
    """Live state of every part of the architecture."""
    def grouped(column: Any) -> dict[str, int]:
        return {str(k): int(v) for k, v in db.query(column, func.count()).group_by(column).all()}

    counts = {name: int(db.execute(select(func.count()).select_from(Base.metadata.tables[name])).scalar() or 0)
              for name in TABLES}
    last_upload = (db.query(Candidate).filter(Candidate.source != "demo").order_by(Candidate.created_at.desc())
                   .first())
    try:
        vectors = {name: container.store.count(name) for name in COLLECTIONS}
    except Exception as exc:
        vectors = {"error": str(exc)[:200]}
    settings = container.settings
    return {
        "tables": counts,
        "candidates_by_source": grouped(Candidate.source),
        "applications_by_status": grouped(Application.status),
        "analyses_by_method": grouped(Analysis.method),
        "analyses_by_status": grouped(Analysis.status),
        "notifications_by_source": grouped(Notification.source),
        "last_extraction": ({k: (last_upload.extraction or {}).get(k) for k in
                             ("model", "adapter", "seconds", "parse_seconds", "input_type", "parser")}
                            | {"candidate": last_upload.full_name, "at": last_upload.created_at.isoformat()}
                            if last_upload else None),
        "vectors": vectors,
        "models": {"llm_provider": settings.llm_provider, "embedding_provider": settings.embedding_provider,
                   "embedding_model": settings.embedding_model, "reranking_model": settings.reranking_model},
        "ai_server": ai_server(settings),
        "n8n": n8n_status(settings),
    }


# --------------------------------------------------------------------------- PostgreSQL


def _cell(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, str) and len(value) > MAX_TEXT:
        return value[:MAX_TEXT] + f"… ({len(value)} characters)"
    return value


def tables(db: Session) -> dict[str, Any]:
    try:
        version = db.execute(text("SELECT version_num FROM alembic_version")).scalar()
    except Exception:  # SQLite in tests, or a database created without Alembic
        db.rollback()
        version = None
    rows = [{"table": name, "rows": int(db.execute(select(func.count()).select_from(Base.metadata.tables[name]))
                                         .scalar() or 0),
             "columns": [c.name for c in Base.metadata.tables[name].columns]} for name in TABLES]
    return {"tables": rows, "migration": version}


def table_rows(db: Session, name: str, limit: int = 20, offset: int = 0) -> dict[str, Any]:
    if name not in TABLES:
        raise NotFoundError(f"Unknown table {name!r}: use one of {', '.join(TABLES)}")
    table = Base.metadata.tables[name]
    query = select(table)
    if "created_at" in table.c:
        query = query.order_by(table.c.created_at.desc())
    rows = db.execute(query.offset(offset).limit(limit)).mappings().all()
    total = int(db.execute(select(func.count()).select_from(table)).scalar() or 0)
    return {"table": name, "total": total, "offset": offset,
            "columns": [{"name": c.name, "type": str(c.type)} for c in table.columns],
            "rows": [{c.name: _cell(row[c.name]) for c in table.columns} for row in rows]}


# --------------------------------------------------------------------------- Qdrant


def collections(container: Any) -> list[dict[str, Any]]:
    return [container.store.collection_info(name)
            or {"name": name, "points": 0, "status": "missing", "dimension": None, "distance": None}
            for name in COLLECTIONS]


def points(container: Any, name: str, limit: int = 20, offset: str | None = None) -> dict[str, Any]:
    _check_collection(name)
    items, next_offset = container.store.scroll(name, limit=limit, offset=offset or None)
    return {"collection": name, "next_offset": str(next_offset) if next_offset is not None else None,
            "points": [{"id": p["id"], "label": _label(name, p["payload"]), "payload": p["payload"]} for p in items]}


def point(container: Any, name: str, point_id: str) -> dict[str, Any]:
    _check_collection(name)
    found = container.store.get_point(name, point_id)
    if found is None:
        raise NotFoundError(f"No point {point_id} in {name}")
    vector = found["vector"] or []
    return {"collection": name, "id": found["id"], "label": _label(name, found["payload"]),
            "payload": found["payload"], "dimension": len(vector), "norm": round(math.sqrt(sum(v * v for v in vector)), 4),
            "vector": [round(float(v), 5) for v in vector]}


def vector_map(container: Any, name: str, limit: int = 400, color_by: str | None = None) -> dict[str, Any]:
    """The vectors of a collection projected on 2 dimensions (PCA): close points = similar texts."""
    import numpy as np

    _check_collection(name)
    color_by = color_by or COLOR_BY[name]
    items, _ = container.store.scroll(name, limit=limit, with_vectors=True)
    items = [p for p in items if p["vector"]]
    if len(items) < 3:
        return {"collection": name, "color_by": color_by, "explained_variance": None, "points": []}
    matrix = np.array([p["vector"] for p in items], dtype=float)
    matrix -= matrix.mean(axis=0)
    _, singular, components = np.linalg.svd(matrix, full_matrices=False)
    coords = matrix @ components[:2].T
    explained = (singular[:2] ** 2 / float((singular ** 2).sum() or 1)).round(3).tolist()
    return {"collection": name, "color_by": color_by, "explained_variance": explained,
            "points": [{"x": round(float(x), 4), "y": round(float(y), 4),
                        "group": str(p["payload"].get(color_by) or "?"), "label": _label(name, p["payload"]),
                        "text": str(p["payload"].get("text", ""))[:200]}
                       for p, (x, y) in zip(items, coords, strict=True)]}


def explain_retrieval(container: Any, query: str, collection: str = KNOWLEDGE, role: str | None = None,
                      top_k: int = 5, candidates: int = 15) -> dict[str, Any]:
    """One search, step by step: the query's BGE-M3 vector, the closest Qdrant points (cosine), then
    the same points reordered by the BGE cross-encoder (what the app and the assistant really use)."""
    _check_collection(collection)
    start = time.perf_counter()
    vector = container.embedder.embed_query(query)
    embed_seconds = round(time.perf_counter() - start, 3)
    criteria = {"roles": [role, "all"]} if role and collection == KNOWLEDGE else None
    start = time.perf_counter()
    hits = container.store.search(collection, vector, top_k=candidates, filter_criteria=criteria)
    search_seconds = round(time.perf_counter() - start, 3)
    start = time.perf_counter()
    scores = container.reranker.score_pairs([[query, h.text] for h in hits]) if hits else []
    rerank_seconds = round(time.perf_counter() - start, 3)
    dense = [{"dense_rank": i, "id": h.id, "cosine": round(h.score, 4), "label": _label(collection, h.metadata),
              "text": h.text[:500]} for i, h in enumerate(hits, 1)]
    if scores is not None and hits:
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        reranked = [{**dense[i], "rerank_rank": rank, "rerank_score": round(float(scores[i]), 4),
                     "moved": dense[i]["dense_rank"] - rank} for rank, i in enumerate(order, 1)]
    else:
        reranked = [{**d, "rerank_rank": d["dense_rank"], "rerank_score": None, "moved": 0} for d in dense]
    settings = container.settings
    return {"query": query, "collection": collection, "filter": criteria, "top_k": top_k,
            "embedding": {"model": settings.embedding_model, "dimension": len(vector),
                          "first_values": [round(float(v), 4) for v in vector[:12]],
                          "norm": round(math.sqrt(sum(float(v) * float(v) for v in vector)), 4)},
            "reranker": settings.reranking_model if scores is not None else None,
            "seconds": {"embedding": embed_seconds, "qdrant": search_seconds, "reranking": rerank_seconds},
            "dense": dense, "reranked": reranked}


# --------------------------------------------------------------------------- matching (contextualization)


def explain_matching(db: Session, container: Any, candidate_id: str, job_id: str) -> dict[str, Any]:
    """Everything that feeds an analysis, without calling the LLM: the evidence found for every
    requirement (fast semantic matching), the HR rules retrieved, and the exact prompt for Qwen."""
    from recruitment_ai.matching import rubric
    from recruitment_ai.matching.analyzer import (
        ANALYSIS_SYSTEM_PROMPT,
        EvidenceAnalyzer,
    )
    from recruitment_ai.matching.evidence import profile_text
    from recruitment_ai.services.jobs import job_payload
    from recruitment_ai.services.matching import candidate_payload

    candidate, job = CandidateRepository(db).get(candidate_id), JobRepository(db).get(job_id)
    if candidate is None or job is None:
        raise NotFoundError("Candidate or job not found")
    payload, job_data = candidate_payload(candidate), job_payload(job)
    matcher = container.semantic
    evidence = matcher.score_requirements(job_data, [payload])[0]
    assessments = matcher.assess(evidence)
    snippets, reranked = evidence["snippets"], evidence["reranked"]
    requirements = []
    for ri, a in enumerate(assessments):
        found = [{"where": snippets[j].label, "section": snippets[j].section, "text": snippets[j].text[:240],
                  "cosine": round(evidence["cosines"][ri][j], 4),
                  "rerank": round(evidence["rerank"][(ri, j)], 4) if (ri, j) in evidence["rerank"] else None}
                 for j in evidence["chosen"][ri]]
        found.sort(key=lambda f: f["rerank"] if f["rerank"] is not None else f["cosine"], reverse=True)
        requirements.append({"id": a.id, "text": a.text, "kind": a.kind, "importance": a.importance,
                             "min_years": a.min_years, "level": a.level, "status": a.status,
                             "years_found": a.years_found, "note": a.note, "snippets": found[:5]})
    knowledge = []
    for query, role in container.analyzer.knowledge_queries(job_data):
        try:
            hits = container.knowledge_search(query, role)
        except Exception as exc:
            knowledge.append({"query": query, "job_profile": role, "error": str(exc)[:200], "hits": []})
            continue
        knowledge.append({"query": query, "job_profile": role, "hits": [
            {"doc_id": (h.metadata or {}).get("document_id"), "title": (h.metadata or {}).get("title"),
             "section": (h.metadata or {}).get("section"), "score": round(h.relevance_score, 4),
             "text": h.text[:300]} for h in hits]})
    sources, _ = container.analyzer.retrieve_knowledge(job_data)
    cv_text = payload["cv_text"] or profile_text(payload["profile"])
    prompt = EvidenceAnalyzer.build_prompt(job_data, payload["profile"], cv_text, sources)
    t = matcher.thresholds
    return {"candidate": candidate.full_name, "job": job.title, "reranked": reranked,
            "thresholds": {"demonstrated": t.demonstrated, "equivalent": t.equivalent} if reranked
            else {"demonstrated": t.cosine_demonstrated, "equivalent": t.cosine_equivalent},
            "snippet_count": len(snippets), "requirements": requirements, "semantic_grade": rubric.grade(assessments),
            "knowledge": knowledge, "rules_in_prompt": [s.doc_id for s in sources],
            "system_prompt": ANALYSIS_SYSTEM_PROMPT, "prompt": prompt, "prompt_chars": len(prompt)}


# --------------------------------------------------------------------------- document processing


def _extraction_run(extractor: Any, cv_text: str, use_adapter: bool) -> dict[str, Any]:
    start = time.perf_counter()
    profile, stats = extractor.extract_text(cv_text, use_adapter=use_adapter)
    return {"adapter_used": use_adapter, "seconds": round(time.perf_counter() - start, 1),
            "model": stats.get("model"), "prompt_tokens": stats.get("prompt_tokens"),
            "output_tokens": stats.get("output_tokens"), "raw_answer": stats.get("raw_answer"),
            "profile": profile.model_dump(mode="json", exclude={"created_at", "id"})}


def _skill_names(profile: dict[str, Any]) -> set[str]:
    return {((s.get("name") if isinstance(s, dict) else str(s)) or "").strip().lower()
            for s in profile.get("skills") or []} - {""}


def compare_profiles(finetuned: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """What the fine-tuned adapter changes on this CV."""
    sections = ["experience", "education", "projects", "skills", "certifications", "languages"]
    skills_ft, skills_base = _skill_names(finetuned), _skill_names(base)
    return {"counts": {s: {"finetuned": len(finetuned.get(s) or []), "base": len(base.get(s) or [])} for s in sections},
            "identity": {f: {"finetuned": finetuned.get(f), "base": base.get(f)}
                         for f in ("full_name", "email", "phone", "location")},
            "skills_only_finetuned": sorted(skills_ft - skills_base),
            "skills_only_base": sorted(skills_base - skills_ft)}


def compare_candidate(db: Session, container: Any, candidate_id: str) -> dict[str, Any]:
    """Base Qwen2.5 (no adapter) on a stored candidate's CV text, next to the profile the fine-tuned model
    wrote when the candidate was stored. One generation (about a minute on the Kaggle GPU); the result is
    kept with the candidate (extraction.model_comparison) and shown in the application's extraction step."""
    from sqlalchemy.orm.attributes import flag_modified

    candidate = CandidateRepository(db).get(candidate_id)
    if candidate is None:
        raise NotFoundError(f"Candidate {candidate_id} not found")
    if not (candidate.cv_text or "").strip():
        raise ValueError("No CV text is stored for this candidate (demo candidates come from a profile, not a CV): "
                         "compare on a CV file in Model comparison instead")
    base = _extraction_run(container.extractor, candidate.cv_text, use_adapter=False)
    finetuned_profile = candidate.profile or {}
    extraction = candidate.extraction or {}
    result = {"at": datetime.utcnow().isoformat(timespec="seconds"),
              "finetuned": {"adapter_used": True, "model": extraction.get("model"), "adapter": extraction.get("adapter"),
                            "seconds": extraction.get("seconds"), "prompt_tokens": extraction.get("prompt_tokens"),
                            "output_tokens": extraction.get("output_tokens"), "profile": finetuned_profile},
              "base": {k: v for k, v in base.items() if k != "raw_answer"} | {"raw_answer": (base.get("raw_answer")
                                                                                              or "")[:20000]},
              "comparison": compare_profiles(finetuned_profile, base["profile"])}
    candidate.extraction = {**extraction, "model_comparison": result}
    flag_modified(candidate, "extraction")
    db.commit()
    return result


def explain_extraction(container: Any, filename: str, content: bytes, compare_base: bool = False) -> dict[str, Any]:
    """A CV through document processing (parts 3-4), step by step. Nothing is stored."""
    from recruitment_ai.extraction.extractor import (
        MAX_INPUT_CHARS,
        SYSTEM_PROMPT,
        CVExtraction,
        text_prompt,
    )
    from recruitment_ai.llm.huggingface_adapter import structured_prompt

    extractor = container.extractor
    with tempfile.TemporaryDirectory() as folder:  # CVs are personal data: the file is not kept
        path = Path(folder) / Path(filename).name
        path.write_bytes(content)
        start = time.perf_counter()
        parsed = extractor.parser.parse(path)
        parse_seconds = round(time.perf_counter() - start, 2)
    cv_text = parsed.raw_text
    user_prompt = structured_prompt(text_prompt(cv_text.strip(), MAX_INPUT_CHARS), CVExtraction)
    finetuned = _extraction_run(extractor, cv_text, use_adapter=True)
    base = _extraction_run(extractor, cv_text, use_adapter=False) if compare_base else None
    return {
        "filename": filename, "adapter": getattr(container.llm, "adapter", None),
        "parsing": {"file_type": parsed.file_type, "engine": parsed.metadata.get("engine"),
                    "input_type": parsed.metadata.get("input_type"), "seconds": parse_seconds,
                    "characters": len(cv_text),
                    "pages": [{"page": p.page_number, "method": p.layout_info.get("method", parsed.file_type),
                               "characters": len(p.text), "ocr_confidence": p.layout_info.get("ocr_confidence")}
                              for p in parsed.pages]},
        "text": cv_text[:15000],
        "prompt": {"system": SYSTEM_PROMPT, "user": user_prompt, "user_chars": len(user_prompt),
                   "cv_truncated": len(cv_text.strip()) > MAX_INPUT_CHARS},
        "finetuned": finetuned, "base": base,
        "comparison": compare_profiles(finetuned["profile"], base["profile"]) if base else None,
    }
