"""MLflow Model Registry: registration, business status (tags) and aliases.

Business lifecycle (EXPERIMENTAL -> VALIDATED -> APPROVED -> DEPLOYED -> ARCHIVED) is stored as a
model-version TAG. Aliases are used for pointers: ``candidate`` (latest registered candidate) and
``champion`` (the version served by the API). "Deployed" == carries the ``champion`` alias.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from flightops_ml.registry import audit

TAG = "flightops."
STATUS_TAG = TAG + "business_status"
ALIAS_CANDIDATE = "candidate"
ALIAS_CHAMPION = "champion"


class BusinessStatus(StrEnum):
    EXPERIMENTAL = "EXPERIMENTAL"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    DEPLOYED = "DEPLOYED"
    ARCHIVED = "ARCHIVED"


def set_tags(client: MlflowClient, name: str, version: str, tags: dict[str, Any]) -> None:
    for key, value in tags.items():
        if value is None:
            continue
        k = key if key.startswith(TAG) else TAG + key
        client.set_model_version_tag(name, str(version), k, str(value)[:4900])


def get_status(mv: ModelVersion) -> str | None:
    return mv.tags.get(STATUS_TAG)


def set_status(client: MlflowClient, name: str, version: str, status: BusinessStatus, reason: str = "") -> None:
    client.set_model_version_tag(name, str(version), STATUS_TAG, status.value)
    audit.append_event("status_change", model=name, version=str(version), status=status.value, reason=reason)


def register_candidate(
    client: MlflowClient,
    name: str,
    model_uri: str,
    tags: dict[str, Any],
    description: str = "",
) -> ModelVersion:
    """Register a trained model as EXPERIMENTAL and point the ``candidate`` alias at it."""
    mv = mlflow.register_model(model_uri=model_uri, name=name)
    version = str(mv.version)
    if description:
        client.update_model_version(name, version, description=description)
    set_tags(client, name, version, tags)
    client.set_model_version_tag(name, version, STATUS_TAG, BusinessStatus.EXPERIMENTAL.value)
    client.set_registered_model_alias(name, ALIAS_CANDIDATE, version)
    audit.append_event("register", model=name, version=version, status="EXPERIMENTAL",
                       run_id=tags.get("run_id"))
    return client.get_model_version(name, version)


def get_alias_version(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def get_champion(client: MlflowClient, name: str) -> ModelVersion | None:
    return get_alias_version(client, name, ALIAS_CHAMPION)


def list_versions(client: MlflowClient, name: str) -> list[ModelVersion]:
    versions = client.search_model_versions(f"name='{name}'")
    return sorted(versions, key=lambda v: int(v.version), reverse=True)


def versions_table(client: MlflowClient, name: str) -> pd.DataFrame:
    champion = get_champion(client, name)
    candidate = get_alias_version(client, name, ALIAS_CANDIDATE)
    rows = []
    for mv in list_versions(client, name):
        t = mv.tags
        rows.append(
            {
                "version": int(mv.version),
                "business_status": t.get(STATUS_TAG, "UNKNOWN"),
                "champion": bool(champion and str(champion.version) == str(mv.version)),
                "candidate": bool(candidate and str(candidate.version) == str(mv.version)),
                "val_pr_auc": _f(t.get(TAG + "val_pr_auc")),
                "test_pr_auc": _f(t.get(TAG + "test_pr_auc")),
                "test_recall": _f(t.get(TAG + "test_recall")),
                "test_f1": _f(t.get(TAG + "test_f1")),
                "threshold": _f(t.get(TAG + "threshold")),
                "algorithm": t.get(TAG + "algorithm"),
                "dataset_version": t.get(TAG + "dataset_version"),
                "run_id": t.get(TAG + "run_id") or mv.run_id,
                "created": pd.to_datetime(mv.creation_timestamp, unit="ms", utc=True),
            }
        )
    return pd.DataFrame(rows)


def _f(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
