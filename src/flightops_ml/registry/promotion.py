"""Gated, auditable model promotion: validate -> approve -> deploy (alias) -> rollback.

Nothing here is reachable from the public API. A newly trained model never replaces the champion
merely because one metric (e.g. accuracy) is higher: gates compare PR-AUC / recall / F1 on the same
fixed hold-out set and a human approval step is mandatory.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient

from flightops_ml.config import load_model_config
from flightops_ml.evaluation.metrics import classification_metrics
from flightops_ml.registry import audit
from flightops_ml.registry.registration import (
    ALIAS_CANDIDATE,
    ALIAS_CHAMPION,
    STATUS_TAG,
    TAG,
    BusinessStatus,
    get_champion,
    get_status,
    set_status,
    set_tags,
)
from flightops_ml.utils import utc_now_iso


class PromotionError(RuntimeError):
    pass


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_gates(
    candidate: dict[str, float],
    champion: dict[str, float] | None,
    gates: dict[str, Any],
    smoke_test_ok: bool,
    data_validation_ok: bool,
) -> list[GateResult]:
    """Pure function: apply the configured promotion gates."""
    res: list[GateResult] = []

    def floor(metric: str, key: str) -> None:
        v, lim = candidate.get(metric, float("nan")), gates[key]
        res.append(GateResult(f"{metric}_floor", bool(v >= lim), f"{metric}={v:.4f} (min {lim})"))

    floor("pr_auc", "min_pr_auc")
    floor("recall", "min_recall")
    floor("f1", "min_f1")
    if gates.get("require_smoke_test", True):
        res.append(GateResult("smoke_test", smoke_test_ok, "model loads from registry and predicts"))
    if gates.get("require_data_validation", True):
        res.append(GateResult("data_validation", data_validation_ok, "training data passed schema validation"))
    if champion is None:
        res.append(GateResult("champion_comparison", True, "no current champion - floors only"))
    else:
        need = champion["pr_auc"] + gates["min_pr_auc_improvement"]
        res.append(
            GateResult(
                "pr_auc_vs_champion",
                bool(candidate["pr_auc"] >= need),
                f"candidate {candidate['pr_auc']:.4f} vs champion {champion['pr_auc']:.4f} "
                f"(+{gates['min_pr_auc_improvement']} required)",
            )
        )
        drop = champion["recall"] - candidate["recall"]
        res.append(
            GateResult(
                "recall_regression",
                bool(drop <= gates["max_recall_regression"]),
                f"recall change {-drop:+.4f} (max regression {gates['max_recall_regression']})",
            )
        )
    return res


def _smoke_test(model_uri: str, sample: pd.DataFrame) -> bool:
    try:
        model = mlflow.sklearn.load_model(model_uri)
        proba = model.predict_proba(sample.head(5))[:, 1]
    except (mlflow.exceptions.MlflowException, OSError, ValueError, KeyError):
        return False
    return bool(len(proba) == min(5, len(sample)) and ((proba >= 0) & (proba <= 1)).all())


def champion_holdout_metrics(
    client: MlflowClient, name: str, X_holdout: pd.DataFrame, y_holdout  # noqa: N803
) -> dict[str, float] | None:
    champ = get_champion(client, name)
    if champ is None:
        return None
    model = mlflow.sklearn.load_model(f"models:/{name}@{ALIAS_CHAMPION}")
    thr = float(champ.tags.get(TAG + "threshold", 0.5))
    return classification_metrics(y_holdout, model.predict_proba(X_holdout)[:, 1], thr)


def validate_candidate(
    client: MlflowClient,
    name: str,
    version: str,
    candidate_holdout_metrics: dict[str, float],
    X_holdout: pd.DataFrame,  # noqa: N803
    y_holdout,
    data_validation_ok: bool = True,
    model_cfg: dict[str, Any] | None = None,
) -> list[GateResult]:
    """Run gates; EXPERIMENTAL -> VALIDATED when all pass. Always recorded in the audit log."""
    cfg = model_cfg or load_model_config()
    mv = client.get_model_version(name, str(version))
    if get_status(mv) not in (BusinessStatus.EXPERIMENTAL.value, BusinessStatus.VALIDATED.value):
        raise PromotionError(f"Version {version} is {get_status(mv)}; only EXPERIMENTAL can be validated")
    champion_metrics = champion_holdout_metrics(client, name, X_holdout, y_holdout)
    smoke = _smoke_test(f"models:/{name}/{version}", X_holdout)
    gates = evaluate_gates(candidate_holdout_metrics, champion_metrics, cfg["gates"], smoke, data_validation_ok)
    passed = all(g.passed for g in gates)
    set_tags(client, name, version, {
        "gate_report": json.dumps([g.as_dict() for g in gates]),
        "gates_passed": passed,
        "validated_at": utc_now_iso(),
    })
    if passed:
        set_status(client, name, version, BusinessStatus.VALIDATED, "all gates passed")
    audit.append_event("validation", model=name, version=str(version), passed=passed,
                       gates=[g.as_dict() for g in gates])
    return gates


def approve(client: MlflowClient, name: str, version: str, approver: str, reason: str) -> None:
    if not approver.strip() or not reason.strip():
        raise PromotionError("Approval requires an approver and a reason")
    mv = client.get_model_version(name, str(version))
    if get_status(mv) != BusinessStatus.VALIDATED.value:
        raise PromotionError(f"Version {version} is {get_status(mv)}; it must be VALIDATED before approval")
    set_tags(client, name, version, {"approved_by": approver, "approved_at": utc_now_iso(), "approval_reason": reason})
    set_status(client, name, version, BusinessStatus.APPROVED, f"approved by {approver}: {reason}")
    audit.append_event("approval", model=name, version=str(version), approver=approver, reason=reason)


def deploy(client: MlflowClient, name: str, version: str, operator: str) -> None:
    """Move the ``champion`` alias to an APPROVED version; archive the previous champion."""
    mv = client.get_model_version(name, str(version))
    if get_status(mv) != BusinessStatus.APPROVED.value:
        raise PromotionError(f"Version {version} is {get_status(mv)}; only APPROVED versions can be deployed")
    previous = get_champion(client, name)
    if previous is not None and str(previous.version) == str(version):
        raise PromotionError(f"Version {version} is already the champion")
    client.set_registered_model_alias(name, ALIAS_CHAMPION, str(version))
    now = utc_now_iso()
    set_tags(client, name, version, {"deployed_by": operator, "deployed_at": now})
    set_status(client, name, version, BusinessStatus.DEPLOYED, f"deployed by {operator}")
    cand = client.get_model_version_by_alias(name, ALIAS_CANDIDATE) if _has_alias(client, name, ALIAS_CANDIDATE) else None
    if cand is not None and str(cand.version) == str(version):
        client.delete_registered_model_alias(name, ALIAS_CANDIDATE)
    if previous is not None:
        set_tags(client, name, str(previous.version), {"superseded_by": version, "archived_at": now})
        set_status(client, name, str(previous.version), BusinessStatus.ARCHIVED, f"superseded by v{version}")
        client.set_registered_model_tag(name, TAG + "previous_champion", str(previous.version))
    audit.append_event("deploy", model=name, version=str(version), operator=operator,
                       previous_version=str(previous.version) if previous else None)


def rollback(client: MlflowClient, name: str, operator: str, reason: str) -> str:
    """Point ``champion`` back to the previously deployed version. Returns the restored version."""
    if not reason.strip():
        raise PromotionError("Rollback requires a reason")
    rm = client.get_registered_model(name)
    prev = rm.tags.get(TAG + "previous_champion")
    current = get_champion(client, name)
    if prev is None or current is None:
        raise PromotionError("No previous champion recorded - nothing to roll back to")
    client.set_registered_model_alias(name, ALIAS_CHAMPION, prev)
    set_status(client, name, prev, BusinessStatus.DEPLOYED, f"rollback by {operator}: {reason}")
    set_tags(client, name, prev, {"deployed_by": operator, "deployed_at": utc_now_iso()})
    set_status(client, name, str(current.version), BusinessStatus.ARCHIVED, f"rolled back: {reason}")
    client.delete_registered_model_tag(name, TAG + "previous_champion")
    audit.append_event("rollback", model=name, restored_version=prev,
                       from_version=str(current.version), operator=operator, reason=reason)
    return prev


def archive(client: MlflowClient, name: str, version: str, reason: str) -> None:
    champ = get_champion(client, name)
    if champ is not None and str(champ.version) == str(version):
        raise PromotionError("Cannot archive the deployed champion; deploy another version or roll back first")
    set_status(client, name, version, BusinessStatus.ARCHIVED, reason)


def _has_alias(client: MlflowClient, name: str, alias: str) -> bool:
    return alias in client.get_registered_model(name).aliases


__all__ = [
    "GateResult", "PromotionError", "STATUS_TAG", "approve", "archive", "deploy",
    "evaluate_gates", "rollback", "validate_candidate",
]
