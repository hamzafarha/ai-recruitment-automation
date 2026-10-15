"""Unified health check script verifying development environment and services."""

import sys
from pathlib import Path
from typing import Any

# Enable UTF-8 encoding on Windows console if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from recruitment_ai.config.settings import get_settings
from recruitment_ai.database.connection import check_db_connection
from recruitment_ai.vectorstore.qdrant import QdrantVectorStore


def check_python_environment() -> tuple[bool, str]:
    """Verify Python version is 3.12+."""
    major, minor = sys.version_info.major, sys.version_info.minor
    if major == 3 and minor >= 12:
        return True, f"Python {major}.{minor}.{sys.version_info.micro}"
    return False, f"Expected Python 3.12+, found {major}.{minor}"


def check_config() -> tuple[bool, str]:
    """Verify configuration loads properly."""
    try:
        settings = get_settings()
        return True, f"Settings valid for '{settings.app_name}' ({settings.app_env})"
    except Exception as exc:
        return False, str(exc)


def check_fastapi_app() -> tuple[bool, str]:
    """Verify FastAPI application structure and routes."""
    try:
        from api.main import app

        route_count = len(app.routes)
        return True, f"FastAPI initialized with {route_count} routes"
    except Exception as exc:
        return False, str(exc)


def check_gpu() -> tuple[bool, str]:
    """Verify PyTorch sees the CUDA GPU."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, f"torch {torch.__version__} has no CUDA (CPU-only build?)"
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(0)
        return True, f"{name} ({free / 1e9:.1f}/{total / 1e9:.1f} GB free)"
    except Exception as exc:
        return False, str(exc)


def check_embeddings() -> tuple[bool, str]:
    """Verify the real embedding model loads (not the hash fallback)."""
    from recruitment_ai.embeddings.service import SentenceTransformerEmbeddingService

    settings = get_settings()
    service = SentenceTransformerEmbeddingService(settings)
    if service.model is None:
        return False, f"{settings.embedding_model} failed to load (using fallback)"
    a, b, c = service.embed_documents(
        ["Python developer", "Développeur Python", "Pastry chef"]
    )
    sim_fr = sum(x * y for x, y in zip(a, b, strict=True))
    sim_off = sum(x * y for x, y in zip(a, c, strict=True))
    ok = len(a) == settings.embedding_dimension and sim_fr > sim_off
    return ok, (
        f"{settings.embedding_model} on {service.model.device}, dim={len(a)}, "
        f"sim(EN,FR)={sim_fr:.2f} > sim(unrelated)={sim_off:.2f}"
    )


def check_reranker() -> tuple[bool, str]:
    """Verify the BGE cross-encoder scores a relevant pair above an irrelevant one."""
    from recruitment_ai.reranking.reranker import OptionalReranker

    settings = get_settings()
    reranker = OptionalReranker(settings)
    query = "Python developer with machine learning experience"
    scores = reranker.cross_encoder.predict(
        [
            [query, "I have 5 years of experience building ML models in Python."],
            [query, "I managed a bakery team and handled customer orders."],
        ]
    )
    relevant, irrelevant = float(scores[0]), float(scores[1])
    # Scores are sigmoid probabilities: a clear match should be well above 0.5
    ok = relevant > 0.5 and relevant > irrelevant
    return ok, (
        f"{settings.reranking_model}: relevant={relevant:.4f} "
        f"> irrelevant={irrelevant:.4f}"
    )


def check_llm() -> tuple[bool, str]:
    """Verify the configured LLM answers, including structured JSON output."""
    import time

    from pydantic import BaseModel

    from recruitment_ai.llm import get_llm_provider

    class _Ping(BaseModel):
        skills: list[str]

    settings = get_settings()
    llm = get_llm_provider()
    start = time.perf_counter()
    result = llm.generate_structured(
        "Extract the skills from: 'I use Python, Docker and SQL daily.'",
        _Ping,
        system_prompt="Return only JSON.",
    )
    elapsed = time.perf_counter() - start
    model = (
        settings.openai_model
        if settings.llm_provider == "openai"
        else settings.llm_provider
    )
    return bool(result.skills), f"{model} → {result.skills} in {elapsed:.1f}s"


def _run_optional(name: str, check: Any) -> bool:
    """Run a check that may fail on import or load; never crash the report."""
    try:
        ok, msg = check()
    except Exception as exc:
        ok, msg = False, f"{type(exc).__name__}: {exc}"
    print(f"{'✓' if ok else '✗'} {name}: {msg}")
    return ok


def main() -> int:
    """Run all health checks and print structured report."""
    print("\n" + "=" * 55)
    print("  AI RECRUITMENT AUTOMATION - SYSTEM HEALTH CHECK")
    print("=" * 55 + "\n")

    all_passed = True

    # 1. Python Environment
    py_ok, py_msg = check_python_environment()
    sym = "✓" if py_ok else "✗"
    print(f"{sym} Python environment: {py_msg}")
    if not py_ok:
        all_passed = False

    # 2. Configuration
    cfg_ok, cfg_msg = check_config()
    sym = "✓" if cfg_ok else "✗"
    print(f"{sym} Configuration: {cfg_msg}")
    if not cfg_ok:
        all_passed = False

    # 3. PostgreSQL
    db_ok, db_msg = check_db_connection()
    sym = "✓" if db_ok else "✗"
    print(f"{sym} PostgreSQL: {db_msg}")
    if not db_ok:
        all_passed = False

    # 4. Qdrant
    settings = get_settings()
    qdrant = QdrantVectorStore(settings)
    q_ok, q_msg = qdrant.health_check()
    sym = "✓" if q_ok else "✗"
    print(f"{sym} Qdrant: {q_msg}")
    if not q_ok:
        all_passed = False

    # 5. FastAPI
    api_ok, api_msg = check_fastapi_app()
    sym = "✓" if api_ok else "✗"
    print(f"{sym} FastAPI Application: {api_msg}")
    if not api_ok:
        all_passed = False

    # 6-9. AI models (first run downloads bge-m3 + reranker, ~4.5 GB)
    print()
    for name, check in [
        ("GPU", check_gpu),
        ("Embeddings", check_embeddings),
        ("Reranker", check_reranker),
        ("LLM", check_llm),
    ]:
        if not _run_optional(name, check):
            all_passed = False

    print("\n" + "-" * 55)
    if all_passed:
        print("Core development environment is properly initialized!")
        print("Note: start services with 'docker compose up -d postgres qdrant n8n'.")
    else:
        print("Environment issues detected. Please review the errors above.")
    print("-" * 55 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
