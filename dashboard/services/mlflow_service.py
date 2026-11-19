"""MLflow data access for the dashboard (real records only)."""

from __future__ import annotations

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from dashboard.services.common import client, model_name
from flightops_ml.config import experiment_name
from flightops_ml.registry import registration


def experiment_id() -> str | None:
    exp = client().get_experiment_by_name(experiment_name())
    return exp.experiment_id if exp else None


def runs_frame(kind: str | None = "model_training", max_results: int = 500) -> pd.DataFrame:
    eid = experiment_id()
    if eid is None:
        return pd.DataFrame()
    flt = f"tags.`flightops.kind` = '{kind}'" if kind else ""
    runs = client().search_runs([eid], filter_string=flt, max_results=max_results,
                                order_by=["attributes.start_time DESC"])
    rows = []
    for r in runs:
        rows.append({"run_id": r.info.run_id, "name": r.info.run_name, "start": pd.to_datetime(r.info.start_time, unit="ms"),
                     "status": r.info.status, "algorithm": r.data.tags.get("flightops.algorithm"),
                     "dataset_version": r.data.tags.get("flightops.dataset_version"),
                     "git_commit": (r.data.tags.get("flightops.git_commit") or "")[:8],
                     **{f"param.{k}": v for k, v in r.data.params.items()},
                     **{f"metric.{k}": v for k, v in r.data.metrics.items()}})
    return pd.DataFrame(rows)


def load_history(run_id: str) -> dict | None:
    try:
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/training/training_history.json")
    except (MlflowException, OSError):
        return None


def load_run_json(run_id: str, artifact: str) -> dict | None:
    try:
        return mlflow.artifacts.load_dict(f"runs:/{run_id}/{artifact}")
    except (MlflowException, OSError):
        return None


def download_artifact(run_id: str, path: str) -> str | None:
    try:
        return mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
    except (MlflowException, OSError):
        return None


def versions() -> pd.DataFrame:
    try:
        return registration.versions_table(client(), model_name())
    except MlflowException:
        return pd.DataFrame()


def champion():
    try:
        return registration.get_champion(client(), model_name())
    except MlflowException:
        return None
