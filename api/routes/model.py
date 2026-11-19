from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import AppState, get_state
from flightops_ml.data.schema import CATEGORICAL_FEATURES, RAW_FEATURES
from flightops_ml.inference.predictor import ModelNotReadyError

router = APIRouter(tags=["model"])


@router.get("/model/info")
def model_info(state: AppState = Depends(get_state)) -> dict:
    try:
        state.predictor.ensure_ready()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    i = state.predictor.info
    assert i is not None
    return {
        "model_name": i.name, "model_version": i.version, "run_id": i.run_id, "alias": i.alias,
        "business_status": i.business_status, "threshold": i.threshold,
        "dataset_version": i.dataset_version, "has_signature": i.has_signature,
        "feature_schema": [
            {"name": f, "type": "category" if f in CATEGORICAL_FEATURES else "float"} for f in RAW_FEATURES
        ],
    }
