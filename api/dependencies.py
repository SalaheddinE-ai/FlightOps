from __future__ import annotations

import hmac
import os
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request

from api.metrics import Metrics
from flightops_ml.inference.predictor import Predictor
from flightops_ml.monitoring.prediction_log import PredictionLog


@dataclass
class AppState:
    predictor: Predictor
    log: PredictionLog
    metrics: Metrics


def get_state(request: Request) -> AppState:
    return request.app.state.flightops


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If FLIGHTOPS_API_KEY is set, prediction endpoints require a matching X-API-Key header."""
    expected = os.environ.get("FLIGHTOPS_API_KEY")
    if not expected:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
