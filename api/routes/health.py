from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse

from api.dependencies import AppState, get_state
from flightops_ml import __version__
from flightops_ml.inference.predictor import ModelNotReadyError

router = APIRouter()


@router.get("/health", tags=["health"])
def health() -> dict:
    """Liveness: the process is up (does not require a model)."""
    return {"status": "ok", "service": "flightops-api", "version": __version__}


@router.get("/ready", tags=["health"])
def ready(state: AppState = Depends(get_state)):
    """Readiness: a registered champion model is loaded and can serve."""
    try:
        state.predictor.ensure_ready()
    except ModelNotReadyError as exc:
        return JSONResponse(status_code=503, content={"status": "not_ready", "detail": str(exc)})
    info = state.predictor.info
    return {"status": "ready", "model_version": info.version if info else None}


@router.get("/metrics", tags=["observability"])
def metrics(state: AppState = Depends(get_state)) -> Response:
    info = state.predictor.info
    body = state.metrics.render(state.predictor.ready, info.version if info else None)
    return Response(content=body, media_type="text/plain; version=0.0.4")
