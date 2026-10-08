"""Environment setup automation script for AI Recruitment Automation."""

import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main() -> None:
    project_root = Path(__file__).parent.parent
    print(f"Initializing project at: {project_root}")

    # Upgrade pip
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install package with dependencies
    run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])

    # Seed synthetic data
    run([sys.executable, str(project_root / "scripts" / "seed_data.py")])

    print("Setup completed successfully.")


if __name__ == "__main__":
    main()
