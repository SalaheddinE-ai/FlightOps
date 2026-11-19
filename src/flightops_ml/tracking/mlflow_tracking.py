"""MLflow tracking helpers (no deprecated model-stage APIs are used anywhere in the project)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from flightops_ml.config import experiment_name, get_paths, tracking_uri


def get_client(uri: str | None = None) -> MlflowClient:
    uri = uri or tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def ensure_experiment(name: str | None = None, uri: str | None = None) -> str:
    """Create the experiment if needed (local artifact root for SQLite stores) and activate it."""
    client = get_client(uri)
    name = name or experiment_name()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        active_uri = uri or tracking_uri()
        if active_uri.startswith("sqlite"):
            art = get_paths().mlflow_local / "artifacts"
            art.mkdir(parents=True, exist_ok=True)
            exp_id = client.create_experiment(name, artifact_location=art.as_uri())
        else:
            exp_id = client.create_experiment(name)
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def log_dict_artifact(payload: Any, artifact_file: str) -> None:
    mlflow.log_dict(json.loads(json.dumps(payload, default=str)), artifact_file)


def log_files(paths: dict[str, Path] | list[Path], artifact_dir: str | None = None) -> None:
    items = paths.values() if isinstance(paths, dict) else paths
    for p in items:
        mlflow.log_artifact(str(p), artifact_path=artifact_dir)


def log_metric_history(history: dict[str, dict[str, list[float]]], client: MlflowClient, run_id: str) -> None:
    """Log the REAL per-iteration XGBoost evaluation history as stepped MLflow metrics."""
    import time

    from mlflow.entities import Metric

    ts = int(time.time() * 1000)
    batch: list[Metric] = []
    for split in ("train", "validation"):
        for metric, values in history.get(split, {}).items():
            for step, value in enumerate(values, start=1):
                batch.append(Metric(f"{split}_{metric}", float(value), ts, step))
    for i in range(0, len(batch), 900):
        client.log_batch(run_id, metrics=batch[i : i + 900])
