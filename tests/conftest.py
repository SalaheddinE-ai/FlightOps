"""Shared fixtures. All tests use SYNTHETIC data and an isolated SQLite MLflow store."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from flightops_ml.data.synthetic import generate_ai4i_like, write_demo_csv

REPO = Path(__file__).resolve().parents[1]
FAST_OVERRIDES = {
    "tuning": {"enabled": True, "n_trials": 2, "cv_folds": 3, "log_trials_to_mlflow": True},
    "models": {
        "random_forest": {"n_estimators": 30},
        "xgboost": {"n_estimators": 60, "early_stopping_rounds": 10},
    },
    "latency": {"n_samples": 5},
}


@pytest.fixture(scope="session")
def synthetic_df():
    return generate_ai4i_like(n=2500, seed=7)


@pytest.fixture()
def isolated_root(tmp_path, monkeypatch):
    """A throw-away project root with the real configs and an isolated MLflow SQLite store."""
    shutil.copytree(REPO / "configs", tmp_path / "configs")
    monkeypatch.setenv("FLIGHTOPS_ROOT", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    monkeypatch.delenv("FLIGHTOPS_ARTIFACTS_DIR", raising=False)
    monkeypatch.delenv("FLIGHTOPS_PREDICTION_DB", raising=False)
    monkeypatch.delenv("FLIGHTOPS_DATASET", raising=False)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    write_demo_csv(tmp_path / "data" / "raw" / "ai4i2020.csv", n=2500, seed=7)
    return tmp_path


@pytest.fixture(scope="session")
def trained_session(tmp_path_factory):
    """Train once (fast config), approve and deploy the champion. Shared by integration tests."""
    import os

    from flightops_ml.registry import promotion
    from flightops_ml.tracking.mlflow_tracking import get_client
    from flightops_ml.training.train import run_training_pipeline

    root = tmp_path_factory.mktemp("session_root")
    shutil.copytree(REPO / "configs", root / "configs")
    (root / "data" / "raw").mkdir(parents=True)
    write_demo_csv(root / "data" / "raw" / "ai4i2020.csv", n=2500, seed=7)
    saved = {k: os.environ.get(k) for k in ("FLIGHTOPS_ROOT", "MLFLOW_TRACKING_URI")}
    os.environ["FLIGHTOPS_ROOT"] = str(root)
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    try:
        result = run_training_pipeline(FAST_OVERRIDES, trigger="pytest")
        client = get_client()
        name = "flightops-failure-classifier"
        v = result.registered_version
        assert v is not None
        if all(g["passed"] for g in result.gates):
            promotion.approve(client, name, v, "pytest", "session fixture")
            promotion.deploy(client, name, v, "pytest")
        yield {"root": root, "result": result, "client": client, "name": name, "version": v}
    finally:
        for k, val in saved.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val
