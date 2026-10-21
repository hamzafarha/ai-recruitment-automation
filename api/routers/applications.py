"""The application pipeline (all parts of the diagram, services/pipeline.py): one workflow per
application, from the CV received to the final decision.

- POST /applications: intake checks; the caller (the n8n workflow) then runs the steps one by one:
  POST /applications/{id}/steps/{step} (document, extract, store, acknowledge, screen, report,
  notify_recruiters, decision, notify_candidate, close);
- POST /applications/submit: intake, then the API runs every step itself, in the background;
- POST /applications/via-n8n: a CV sent to the n8n workflow's webhook (the interface cannot call n8n
  itself: other origin), which then runs every step;
- POST /applications/{id}/continue: continue from the next (or failed) step with the API;
- GET /applications, /applications/{id}, /applications/steps: the applications and their step logs;
- GET /applications/{id}/outputs: what every step produced, in full (services/outputs.py).
"""

import base64
import binascii
from typing import Any, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from api.dependencies import Container, get_container, get_db
from api.errors import service_errors
from recruitment_ai.database.repositories import ApplicationRepository
from recruitment_ai.services import pipeline
from recruitment_ai.services.outputs import outputs as step_outputs

N8N_TIMEOUT = 900  # the n8n workflow answers after the extraction (about a minute on the Kaggle GPU)

router = APIRouter(prefix="/applications", tags=["Application pipeline"])


class ApplicationIn(BaseModel):
    filename: str = Field(min_length=3, max_length=255)
    content_base64: str = Field(min_length=10)
    source: Literal["upload", "portal", "email", "n8n"] = "upload"
    consent: bool = Field(default=False, description="The candidate accepted the data policy (KB-HR-004)")
    job_id: str | None = Field(default=None, description="The job applied to; none = spontaneous application")


class StepIn(BaseModel):
    content_base64: str | None = Field(default=None, description="The CV file: document step only")
    resume_url: str | None = Field(default=None, max_length=1000,
                                   description="notify_recruiters step: the n8n Wait node's resume URL")


def _decode(text: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("content_base64 is not valid base64") from exc


def _register(body: ApplicationIn, db: Session, container: Container, orchestrator: str) -> tuple[Any, bytes]:
    with service_errors(value_status=400):
        content = _decode(body.content_base64)
        app = pipeline.submit(db, container, body.filename, content, source=body.source, consent=body.consent,
                              job_id=body.job_id or None, orchestrator=orchestrator)
    return app, content


@router.get("/steps")
def list_steps() -> dict[str, Any]:
    """The steps of the workflow, in order, with the part of the architecture each one implements."""
    return {"data": pipeline.steps_catalogue()}


@router.post("", status_code=201)
def create_application(body: ApplicationIn, db: Session = Depends(get_db),
                       container: Container = Depends(get_container)) -> dict[str, Any]:
    """Intake checks and registration (the n8n workflow's first node); the caller runs the next steps."""
    app, _ = _register(body, db, container, orchestrator="n8n")
    return pipeline.application_dict(app, full=True)


@router.post("/submit", status_code=202)
def submit_application(body: ApplicationIn, background: BackgroundTasks, db: Session = Depends(get_db),
                       container: Container = Depends(get_container)) -> dict[str, Any]:
    """Intake checks, then every step run by the API in the background (a few minutes on the Kaggle GPU):
    follow the application on the Pipeline page."""
    app, content = _register(body, db, container, orchestrator="api")
    factory = sessionmaker(bind=db.get_bind(), autocommit=False, autoflush=False)
    background.add_task(pipeline.run_in_background, factory, container, app.id, content)
    return pipeline.application_dict(app, full=True)


@router.post("/via-n8n", status_code=201)
def submit_via_n8n(body: ApplicationIn, container: Container = Depends(get_container)) -> dict[str, Any]:
    """The CV goes to the n8n workflow "Application pipeline" (webhook cv-intake), which runs every step;
    n8n answers once the candidate is stored (after the extraction)."""
    url = f"{container.settings.n8n_url.rstrip('/')}/webhook/cv-intake"
    try:
        response = httpx.post(url, json=body.model_dump(), timeout=N8N_TIMEOUT)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"n8n is not reachable at {url} ({type(exc).__name__}): start it "
                            "(docker compose up -d n8n) and publish the workflows (python scripts/n8n_setup.py), "
                            "or send the CV straight to the API") from exc
    if response.status_code == 404:
        raise HTTPException(status_code=502, detail="n8n answered 404: the 'Application pipeline' workflow is not "
                            "imported or not active (python scripts/n8n_setup.py)")
    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail=f"n8n answered {response.status_code}: {response.text[:300]}")
    if response.status_code >= 400 or data.get("detail") or not data.get("application_id"):
        raise HTTPException(status_code=response.status_code if response.status_code >= 400 else 502,
                            detail=f"The application was not received: {data.get('detail') or data}")
    return data


@router.get("")
def list_applications(status: str | None = None, limit: int = 200, db: Session = Depends(get_db)) -> dict[str, Any]:
    applications = ApplicationRepository(db).filter(status=status, limit=limit)
    return {"count": len(applications), "data": [pipeline.application_dict(a) for a in applications]}


@router.get("/{application_id}")
def get_application(application_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    with service_errors():
        return pipeline.application_dict(pipeline.get(db, application_id), full=True)


@router.get("/{application_id}/outputs")
def get_outputs(application_id: str, db: Session = Depends(get_db),
                container: Container = Depends(get_container)) -> dict[str, Any]:
    """What every step produced, in full: CV text, extracted profile, Qdrant points, e-mails, the screening
    agent's tool calls and retrieved documents, reports, reviews."""
    with service_errors():
        return step_outputs(db, container, pipeline.get(db, application_id))


@router.post("/{application_id}/steps/{step}")
def run_step(application_id: str, step: str, body: StepIn | None = None, db: Session = Depends(get_db),
             container: Container = Depends(get_container)) -> dict[str, Any]:
    """Run one step of the workflow (the n8n nodes). 409 when it is not the application's next step; a step
    already done answers its stored result."""
    body = body or StepIn()
    with service_errors(value_status=422):
        content = _decode(body.content_base64) if body.content_base64 else None
        app = pipeline.get(db, application_id)
        result = pipeline.run_step(db, container, app, step, content=content, resume_url=body.resume_url)
    db.expire_all()
    return {"step": step, "result": result,
            "application": pipeline.application_dict(pipeline.get(db, application_id))}


@router.post("/{application_id}/continue")
def continue_application(application_id: str, db: Session = Depends(get_db),
                         container: Container = Depends(get_container)) -> dict[str, Any]:
    """Continue with the API from the next (or failed) step, until the recruiter's decision is awaited."""
    with service_errors(value_status=422):
        app = pipeline.continue_workflow(db, container, application_id)
    return pipeline.application_dict(app, full=True)
