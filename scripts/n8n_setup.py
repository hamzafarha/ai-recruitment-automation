"""Import and activate the n8n workflows of n8n/workflows/ (diagram part 2), then check them.

Usage (from the project root; Docker Desktop running):
  python scripts/n8n_setup.py            # start n8n, import + activate the 5 workflows, send a test event
  python scripts/n8n_setup.py --check    # only check that the webhooks answer

The main one is "Recruitment — Application pipeline" (n8n/workflows/1_application_pipeline.json): one
execution per application, from the CV to the final decision. It replaces the former "CV intake" workflow
(same id and webhook path: re-running this script replaces it).
Then set N8N_EVENTS_ENABLED=true in .env and restart the API: the other app events go to n8n too.
The workflows call the API at http://host.docker.internal:8000, so start the API with
``python -m uvicorn api.main:app --host 0.0.0.0 --port 8000``.
Re-running it is safe: the workflows keep their ids and are replaced.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

CONTAINER = "recruitment_n8n"
N8N = "http://localhost:5678"
# the workflows that start from a webhook or a schedule; the error-alert workflow needs no publishing:
# n8n runs it by itself when another workflow fails
TRIGGERED_IDS = ["RecruitCvIntake1", "RecruitNewJob001", "RecruitEvents001", "RecruitDigest001"]


def run(*cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    output = (result.stdout + result.stderr).strip()
    if output:
        print("  " + output[-800:].replace("\n", "\n  "))
    if check and result.returncode != 0:
        sys.exit(f"Command failed: {' '.join(cmd)}")
    return result


def wait_for_n8n(minutes: float = 3) -> None:
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        try:
            if httpx.get(f"{N8N}/healthz", timeout=3).status_code == 200:
                print("  n8n is up")
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    sys.exit("n8n did not start: check `docker logs recruitment_n8n`")


def activate() -> bool:
    """n8n 2.x publishes workflows (publish:workflow). On n8n 1.x that command does not exist, and workflows are
    activated with update:workflow (deprecated in 2.x, where it no longer activates anything)."""
    published = [run("docker", "exec", CONTAINER, "n8n", "publish:workflow", f"--id={wid}", check=False).returncode == 0
                 for wid in TRIGGERED_IDS]
    if all(published):
        return True
    return run("docker", "exec", CONTAINER, "n8n", "update:workflow", "--all", "--active=true", check=False).returncode == 0


def check() -> bool:
    body = {"event": "test", "title": "Test event from n8n_setup.py", "message": "", "payload": {}, "source": "app"}
    try:
        response = httpx.post(f"{N8N}/webhook/recruitment-events", json=body, timeout=20)
    except httpx.HTTPError as exc:
        print(f"  n8n not reachable: {exc}")
        return False
    if response.status_code != 200:
        print(f"  the recruitment-events webhook answered {response.status_code}: the workflow is not active")
        return False
    print("  the recruitment-events webhook answered: a 'n8n is connected' notification appears in the app "
          "(Notifications page) if the API is running with --host 0.0.0.0")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check the webhooks")
    args = parser.parse_args()
    if not args.check:
        files = sorted(Path("n8n/workflows").glob("*.json"))
        print(f"{len(files)} workflows: " + ", ".join(json.loads(f.read_text(encoding="utf-8"))["name"] for f in files))
        run("docker", "compose", "up", "-d", "n8n")
        wait_for_n8n()
        run("docker", "exec", CONTAINER, "n8n", "import:workflow", "--separate", "--input=/workflows")
        if not activate():
            print("  Could not activate the workflows from the command line: open http://localhost:5678, open each "
                  "'Recruitment — …' workflow and switch it to Active (Publish in recent n8n versions).")
        run("docker", "restart", CONTAINER)  # active workflows register their webhooks at start-up
        wait_for_n8n()
    ok = check()
    print("\nn8n editor: http://localhost:5678 (the first visit asks you to create the local owner account).")
    if ok:
        print("Next: set N8N_EVENTS_ENABLED=true in .env and restart the API.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
