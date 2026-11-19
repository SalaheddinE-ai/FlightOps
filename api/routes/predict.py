from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import AppState, get_state, require_api_key
from api.schemas import BatchRequest, BatchResponse, MachineObservation, PredictionResponse
from flightops_ml.inference.predictor import ModelNotReadyError

router = APIRouter(tags=["prediction"], dependencies=[Depends(require_api_key)])


def _run(state: AppState, observations: list[MachineObservation], explain: bool) -> list[PredictionResponse]:
    try:
        state.predictor.ensure_ready()
    except ModelNotReadyError as exc:
        state.metrics.inc("prediction_errors")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    frame = pd.DataFrame([o.model_dump() for o in observations])
    t0 = time.perf_counter()
    results = state.predictor.predict_frame(frame)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / len(observations)
    info = state.predictor.info
    assert info is not None
    out: list[PredictionResponse] = []
    for i, (obs, res) in enumerate(zip(observations, results, strict=True)):
        rid = uuid.uuid4().hex
        explanation = state.predictor.explain(frame.iloc[[i]]) if explain else None
        state.log.log_prediction(
            request_id=rid, model_name=info.name, model_version=res["model_version"],
            threshold=res["threshold"], probability=res["failure_probability"],
            prediction=res["prediction"], risk_level=res["risk_level"], latency_ms=latency_ms,
            features=obs.model_dump(),
        )
        out.append(PredictionResponse(request_id=rid, timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
                                      explanation=explanation, **res))
    state.metrics.inc("predictions", len(out))
    return out


@router.post("/predict", response_model=PredictionResponse)
def predict(obs: MachineObservation, explain: bool = Query(False, description="Include SHAP contributions"),
            state: AppState = Depends(get_state)) -> PredictionResponse:
    return _run(state, [obs], explain)[0]


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest, state: AppState = Depends(get_state)) -> BatchResponse:
    preds = _run(state, req.observations, explain=False)
    return BatchResponse(count=len(preds), model_version=preds[0].model_version, predictions=preds)
