"""Unified health check script verifying development environment and services."""

import sys
from pathlib import Path

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
    sym = "✓" if db_ok else "⚠"
    print(f"{sym} PostgreSQL: {db_msg}")

    # 4. Qdrant
    settings = get_settings()
    qdrant = QdrantVectorStore(settings)
    q_ok, q_msg = qdrant.health_check()
    sym = "✓" if q_ok else "⚠"
    print(f"{sym} Qdrant: {q_msg}")

    # 5. FastAPI
    api_ok, api_msg = check_fastapi_app()
    sym = "✓" if api_ok else "✗"
    print(f"{sym} FastAPI Application: {api_msg}")
    if not api_ok:
        all_passed = False

    print("\n" + "-" * 55)
    if all_passed:
        print("Core development environment is properly initialized!")
        print("Note: PostgreSQL & Qdrant can be launched with 'docker compose up -d'.")
    else:
        print("Environment issues detected. Please review the errors above.")
    print("-" * 55 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
