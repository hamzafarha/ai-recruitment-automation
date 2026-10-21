"""Evaluation results of every part of the system (data/evaluation/, runs on Kaggle) and MLflow status."""

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from api.errors import service_errors
from recruitment_ai.services import evaluation

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/overview")
def overview() -> dict[str, Any]:
    """What each evaluation measures, with which technologies, and its headline number."""
    return evaluation.overview()


@router.get("/extraction")
def extraction(before: str | None = None, after: str | None = None) -> dict[str, Any]:
    """CV extraction: base Qwen2.5 vs Qwen2.5 + QLoRA adapter (or any two runs)."""
    with service_errors():
        return evaluation.extraction(before, after)


@router.get("/finetune")
def finetune() -> dict[str, Any]:
    return evaluation.finetune()


@router.get("/rag")
def rag() -> dict[str, Any]:
    return evaluation.rag()


@router.get("/matching")
def matching() -> dict[str, Any]:
    """Matching and the screening agent on the 36 graded pairs."""
    return evaluation.matching()


@router.get("/assistant")
def assistant() -> dict[str, Any]:
    return evaluation.assistant()


@router.get("/pipeline")
def pipeline_stats(db: Session = Depends(get_db)) -> dict[str, Any]:
    """The live application pipeline: timings and outcomes of every step."""
    return evaluation.pipeline_stats(db)


@router.get("/mlflow")
def mlflow(container: Container = Depends(get_container)) -> dict[str, Any]:
    return evaluation.mlflow_status(container.settings)
