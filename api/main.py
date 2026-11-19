"""FlightOps ML inference API (FastAPI). Serves the registered MLflow model behind the `champion` alias.

There is deliberately NO promotion/rollback endpoint: governance actions live in the admin CLI and the
authenticated admin panel of the dashboard.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.dependencies import AppState
from api.metrics import Metrics
from api.routes import health, model, predict
from flightops_ml import __version__
from flightops_ml.inference.predictor import ModelNotReadyError, Predictor
from flightops_ml.logging_utils import get_logger
from flightops_ml.monitoring.prediction_log import PredictionLog

log = get_logger("flightops.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState(predictor=Predictor(), log=PredictionLog(), metrics=Metrics())
    app.state.flightops = state
    try:
        info = state.predictor.load()  # model loaded once per worker
        log.info("model loaded", extra={"version": info.version, "alias": info.alias})
    except ModelNotReadyError as exc:
        log.warning("starting without a model", extra={"reason": str(exc)})
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="FlightOps ML API", version=__version__, lifespan=lifespan,
                  description="Predictive-maintenance failure-probability API (AI4I 2020 is synthetic data).")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        state: AppState | None = getattr(request.app.state, "flightops", None)
        if state is not None:
            state.metrics.observe(request.url.path, response.status_code, time.perf_counter() - t0)
            if response.status_code >= 400 and request.url.path not in ("/metrics",):
                state.log.log_error(request.url.path, response.status_code, "http_error", "")
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        state: AppState = request.app.state.flightops
        state.metrics.inc("invalid_inputs")
        return JSONResponse(status_code=422, content={"error": "validation_error", "detail": exc.errors()},
                            )

    @app.exception_handler(ModelNotReadyError)
    async def not_ready_handler(request: Request, exc: ModelNotReadyError):
        return JSONResponse(status_code=503, content={"error": "model_not_ready", "detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.error("unhandled error", exc_info=exc, extra={"path": request.url.path})
        state: AppState | None = getattr(request.app.state, "flightops", None)
        if state is not None:
            state.metrics.inc("prediction_errors")
            state.log.log_error(request.url.path, 500, type(exc).__name__, str(exc))
        return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "Unexpected error"})

    app.include_router(health.router)
    app.include_router(model.router)
    app.include_router(predict.router)
    return app


app = create_app()
