"""Central configuration: paths, YAML configs and environment-driven settings."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATASET_FILENAME = "ai4i2020.csv"


def project_root() -> Path:
    """Project root. Override with FLIGHTOPS_ROOT (used by Docker and the test-suite)."""
    env = os.environ.get("FLIGHTOPS_ROOT")
    return Path(env).resolve() if env else Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def configs(self) -> Path:
        return Path(os.environ.get("FLIGHTOPS_CONFIG_DIR", self.root / "configs"))

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def data_reference(self) -> Path:
        return self.root / "data" / "reference"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def evaluations(self) -> Path:
        return self.reports / "evaluations"

    @property
    def monitoring_reports(self) -> Path:
        return self.reports / "monitoring"

    @property
    def artifacts(self) -> Path:
        return Path(os.environ.get("FLIGHTOPS_ARTIFACTS_DIR", self.root / "artifacts"))

    @property
    def jobs(self) -> Path:
        return self.artifacts / "jobs"

    @property
    def governance(self) -> Path:
        return self.artifacts / "governance"

    @property
    def prediction_db(self) -> Path:
        return Path(
            os.environ.get("FLIGHTOPS_PREDICTION_DB", self.artifacts / "monitoring" / "predictions.db")
        )

    @property
    def mlflow_local(self) -> Path:
        return self.root / "mlflow_local"

    def dataset_path(self) -> Path:
        env = os.environ.get("FLIGHTOPS_DATASET")
        if env:
            p = Path(env)
            return p if p.is_absolute() else self.root / p
        return self.data_raw / DEFAULT_DATASET_FILENAME

    def ensure_dirs(self) -> None:
        for p in (
            self.data_processed,
            self.data_reference,
            self.figures,
            self.evaluations,
            self.monitoring_reports,
            self.jobs,
            self.governance,
            self.prediction_db.parent,
        ):
            p.mkdir(parents=True, exist_ok=True)


def get_paths() -> Paths:
    return Paths(root=project_root())


def deep_merge(base: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(name: str) -> dict[str, Any]:
    path = get_paths().configs / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file {path} must contain a YAML mapping")
    return data


def load_training_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return deep_merge(load_yaml("training"), overrides)


def load_model_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return deep_merge(load_yaml("model"), overrides)


def load_monitoring_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    return deep_merge(load_yaml("monitoring"), overrides)


def tracking_uri() -> str:
    """MLflow tracking URI. Defaults to a local SQLite store (no server required)."""
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    db = get_paths().mlflow_local / "mlflow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db.as_posix()}"


def registered_model_name() -> str:
    return os.environ.get(
        "FLIGHTOPS_MODEL_NAME",
        load_training_config()["mlflow"]["registered_model_name"],
    )


def experiment_name() -> str:
    return os.environ.get(
        "FLIGHTOPS_EXPERIMENT", load_training_config()["mlflow"]["experiment_name"]
    )
