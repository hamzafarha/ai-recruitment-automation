"""Step 3 — build the HR knowledge index (Qdrant) and evaluate retrieval.

Usage (from the project root; Qdrant must be running: docker compose up -d qdrant):
  python scripts/eval_rag.py              # rebuild the index, evaluate the 4 configurations
  python scripts/eval_rag.py --no-build   # evaluate the existing index only

Outputs in data/evaluation/rag/: report.md, metrics.json, per_query.csv.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from recruitment_ai.config.settings import get_settings
from recruitment_ai.data.knowledge import KnowledgeBaseLoader
from recruitment_ai.embeddings.service import SentenceTransformerEmbeddingService
from recruitment_ai.evaluation.rag_eval import evaluate, load_queries, report
from recruitment_ai.rag.knowledge_index import KnowledgeIndex, job_chunks, kb_chunks
from recruitment_ai.reranking.reranker import OptionalReranker
from recruitment_ai.utils.logging import configure_logging
from recruitment_ai.vectorstore.qdrant import QdrantVectorStore

EVAL_DIR = Path("data/processed/eval_v1")
OUT = Path("data/evaluation/rag")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-build", action="store_true", help="skip rebuilding the index")
    args = parser.parse_args()
    configure_logging(level="WARNING")
    settings = get_settings()

    store = QdrantVectorStore(settings)
    ok, message = store.health_check()
    if not ok:
        sys.exit(f"Qdrant is not reachable ({message}). Start it with: docker compose up -d qdrant")
    embedder = SentenceTransformerEmbeddingService(settings)
    reranker = OptionalReranker(settings)
    reranker.cross_encoder  # load now: a reranker that silently fails would make "rerank" = "dense"  # noqa: B018
    index = KnowledgeIndex(store, embedder, reranker)

    docs = KnowledgeBaseLoader().load_all()
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((EVAL_DIR / "jobs").glob("*.json"))]
    chunks = kb_chunks(docs) + [c for job in jobs for c in job_chunks(job)]
    if not args.no_build:
        start = time.perf_counter()
        count = index.build(chunks, settings.embedding_dimension)
        print(f"  indexed {count} chunks ({len(docs)} KB documents + {len(jobs)} job postings) "
              f"in {time.perf_counter() - start:.1f}s -> collection '{index.collection}'")

    doc_languages = {d.doc_id: d.language for d in docs} | {f"JD-{j['role']}": "en" for j in jobs}
    queries = load_queries(EVAL_DIR / "rag" / "queries.json")
    start = time.perf_counter()
    results = evaluate(
        lambda q, role, rerank, top_k: index.search(q, role=role, top_k=top_k, candidates=30, rerank=rerank),
        queries, doc_languages,
    )
    print(f"  evaluated {len(queries)} questions x {len(results)} configurations in {time.perf_counter() - start:.0f}s")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(report(results, queries, len(chunks)), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps({n: r["summary"] for n, r in results.items()}, indent=2),
                                      encoding="utf-8")
    with (OUT / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["configuration", "id", "language", "cross_lingual", "hit@1", "hit@3", "rr", "top3"])
        for name, result in results.items():
            for r in result["rows"]:
                writer.writerow([name, r["id"], r["language"], r["cross_lingual"], r["hit@1"], r["hit@3"],
                                 round(r["rr"], 3), " ".join(r["ranked"][:3])])
    for name, result in results.items():
        s = result["summary"]["all"]
        print(f"  {name:26s} hit@1 {s['hit@1']:.0%} | hit@3 {s['hit@3']:.0%} | MRR {s['mrr']:.2f}")
    print(f"  -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
