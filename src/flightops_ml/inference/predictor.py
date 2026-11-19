"""Registry-backed predictor. Always serves the model version that carries the ``champion`` alias."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from flightops_ml.config import load_model_config, registered_model_name, tracking_uri
from flightops_ml.data.schema import RAW_FEATURES
from flightops_ml.evaluation.explainability import explain_row, supports_shap
from flightops_ml.registry.registration import ALIAS_CHAMPION, TAG
from flightops_ml.tracking.mlflow_tracking import get_client


class ModelNotReadyError(RuntimeError):
    pass


@dataclass
class LoadedModelInfo:
    name: str
    version: str
    alias: str
    run_id: str | None
    threshold: float
    business_status: str | None
    loaded_at: float = field(default_factory=time.time)
    has_signature: bool = False
    dataset_version: str | None = None


def risk_level(probability: float, threshold: float, cfg: dict[str, Any] | None = None) -> str:
    """Project-defined risk bands (not an industry standard)."""
    cfg = cfg or load_model_config()["risk"]
    if probability >= max(threshold, cfg["critical_probability"]):
        return "CRITICAL"
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold * cfg["medium_fraction_of_threshold"]:
        return "MEDIUM"
    return "LOW"


class Predictor:
    def __init__(self, name: str | None = None, alias: str = ALIAS_CHAMPION):
        self.name = name or registered_model_name()
        self.alias = alias
        self._model = None
        self.info: LoadedModelInfo | None = None
        self._lock = threading.Lock()
        self._last_check = 0.0
        self._risk_cfg = load_model_config()["risk"]
        self._refresh_seconds = float(load_model_config()["serving"]["champion_refresh_seconds"])

    @property
    def ready(self) -> bool:
        return self._model is not None and self.info is not None

    def load(self) -> LoadedModelInfo:
        """Load the model version currently behind the alias. Raises ModelNotReadyError."""
        client = get_client(tracking_uri())
        try:
            mv = client.get_model_version_by_alias(self.name, self.alias)
        except MlflowException as exc:
            raise ModelNotReadyError(
                f"No model version has alias '{self.alias}' for '{self.name}'. Approve and deploy one first."
            ) from exc
        if TAG + "threshold" not in mv.tags:
            raise ModelNotReadyError(f"Version {mv.version} has no decision threshold tag; refusing to serve")
        try:
            model = mlflow.sklearn.load_model(f"models:/{self.name}@{self.alias}")
        except (MlflowException, OSError) as exc:
            raise ModelNotReadyError(f"Failed to load model version {mv.version}: {exc}") from exc
        try:
            sig = mlflow.models.get_model_info(f"models:/{self.name}/{mv.version}").signature
        except MlflowException:
            sig = None
        info = LoadedModelInfo(
            name=self.name, version=str(mv.version), alias=self.alias,
            run_id=mv.tags.get(TAG + "run_id") or mv.run_id,
            threshold=float(mv.tags[TAG + "threshold"]),
            business_status=mv.tags.get(TAG + "business_status"),
            has_signature=sig is not None, dataset_version=mv.tags.get(TAG + "dataset_version"),
        )
        with self._lock:
            self._model, self.info = model, info
        self._last_check = time.monotonic()
        return info

    def refresh_if_stale(self) -> None:
        """Pick up a promotion / rollback without restarting (alias is re-resolved periodically)."""
        if time.monotonic() - self._last_check < self._refresh_seconds:
            return
        self._last_check = time.monotonic()
        try:
            mv = get_client(tracking_uri()).get_model_version_by_alias(self.name, self.alias)
        except MlflowException:
            return
        if not self.ready or self.info is None or self.info.version != str(mv.version):
            self.load()

    def ensure_ready(self) -> None:
        if not self.ready:
            self.load()
        else:
            self.refresh_if_stale()

    def predict_frame(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        self.ensure_ready()
        assert self._model is not None and self.info is not None
        proba = self._model.predict_proba(frame[RAW_FEATURES])[:, 1]
        thr = self.info.threshold
        return [
            {
                "prediction": int(p >= thr),
                "failure_probability": float(p),
                "risk_level": risk_level(float(p), thr, self._risk_cfg),
                "model_version": self.info.version,
                "threshold": thr,
            }
            for p in proba
        ]

    def explain(self, frame: pd.DataFrame, top_k: int = 8) -> dict[str, Any] | None:
        self.ensure_ready()
        assert self._model is not None
        if not supports_shap(self._model):
            return None
        return explain_row(self._model, frame[RAW_FEATURES], top_k)

    @property
    def model(self):
        return self._model
