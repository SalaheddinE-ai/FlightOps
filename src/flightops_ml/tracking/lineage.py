"""Model lineage built from REAL stored metadata. Missing links are reported, never invented."""

from __future__ import annotations

from typing import Any

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from flightops_ml.monitoring.prediction_log import PredictionLog
from flightops_ml.registry import audit
from flightops_ml.registry.registration import TAG

CHAIN = [
    "dataset", "validation_report", "preprocessing", "feature_engineering", "training_config",
    "experiment", "run", "model_artifact", "model_version", "deployment", "predictions",
]
LABELS = {
    "dataset": "Dataset version", "validation_report": "Data validation report",
    "preprocessing": "Preprocessing config", "feature_engineering": "Feature engineering",
    "training_config": "Training config", "experiment": "MLflow experiment", "run": "MLflow run",
    "model_artifact": "Model artifact", "model_version": "Registered version",
    "deployment": "Deployment", "predictions": "Prediction records",
}


def assemble_lineage(
    *,
    name: str,
    version: str,
    tags: dict[str, str],
    source: str | None,
    run_params: dict[str, str] | None,
    run_artifacts: list[str] | None,
    parent_artifacts: list[str] | None,
    deploy_events: list[dict[str, Any]],
    n_predictions: int,
) -> dict[str, Any]:
    """Pure function: build the lineage graph from metadata dictionaries."""
    t = lambda k: tags.get(TAG + k)  # noqa: E731
    run_artifacts = run_artifacts or []
    parent_artifacts = parent_artifacts or []

    def has(arts: list[str], needle: str) -> bool:
        return any(a.endswith(needle) for a in arts)

    spec: dict[str, tuple[bool, dict[str, Any], str]] = {
        "dataset": (bool(t("dataset_version")),
                    {"version": t("dataset_version"), "sha256": t("dataset_sha256"),
                     "path": (run_params or {}).get("dataset_path"), "git_commit": t("git_commit")},
                    "dataset_version tag not set"),
        "validation_report": (has(parent_artifacts, "data_quality_report.json"),
                              {"artifact": "data/data_quality_report.json", "passed": t("data_validation_passed")},
                              "no data_quality_report.json on the parent run"),
        "preprocessing": (has(run_artifacts, "preprocessing_config.json"),
                          {"artifact": "config/preprocessing_config.json"}, "preprocessing_config.json not logged"),
        "feature_engineering": (bool(t("feature_engineering_version")),
                                {"version": t("feature_engineering_version")}, "feature engineering version not recorded"),
        "training_config": (has(run_artifacts, "training_config.json"),
                            {"artifact": "config/training_config.json", "threshold": t("threshold"),
                             "strategy": t("threshold_strategy"), "algorithm": t("algorithm")},
                            "training_config.json not logged"),
        "experiment": (bool(t("experiment_id")), {"experiment_id": t("experiment_id")}, "experiment id not recorded"),
        "run": (bool(t("run_id")), {"run_id": t("run_id"), "parent_run_id": t("parent_run_id")}, "run id not recorded"),
        "model_artifact": (bool(source), {"source": source}, "registered version has no source"),
        "model_version": (True, {"name": name, "version": version, "business_status": t("business_status"),
                                 "approved_by": t("approved_by"), "approved_at": t("approved_at")}, ""),
        "deployment": (bool(deploy_events) or bool(t("deployed_at")),
                       {"deployed_at": t("deployed_at"), "deployed_by": t("deployed_by"), "events": deploy_events},
                       "this version was never deployed"),
        "predictions": (n_predictions > 0, {"n_predictions": n_predictions},
                        "no prediction records for this model version"),
    }
    nodes = [{"id": k, "kind": k, "label": LABELS[k], "status": "present" if spec[k][0] else "missing",
              "details": spec[k][1], "reason": "" if spec[k][0] else spec[k][2]} for k in CHAIN]
    edges = [{"source": a, "target": b} for a, b in zip(CHAIN, CHAIN[1:], strict=False)]
    return {"model": name, "version": version, "nodes": nodes, "edges": edges,
            "missing": [n["id"] for n in nodes if n["status"] == "missing"]}


def _list_all(client: MlflowClient, run_id: str | None, depth: int = 3) -> list[str]:
    if not run_id:
        return []
    out: list[str] = []

    def walk(path: str | None, d: int) -> None:
        try:
            items = client.list_artifacts(run_id, path)
        except MlflowException:
            return
        for it in items:
            if it.is_dir and d > 0:
                walk(it.path, d - 1)
            else:
                out.append(it.path)

    walk(None, depth)
    return out


def fetch_lineage(client: MlflowClient, name: str, version: str, log: PredictionLog | None = None) -> dict[str, Any]:
    mv = client.get_model_version(name, str(version))
    tags = dict(mv.tags)
    run_id = tags.get(TAG + "run_id") or mv.run_id
    run_params = None
    if run_id:
        try:
            run_params = dict(client.get_run(run_id).data.params)
        except MlflowException:
            run_params = None
    events = [e for e in audit.read_events(("deploy", "rollback", "approval"))
              if str(e.get("version") or e.get("restored_version")) == str(version)]
    log = log or PredictionLog()
    return assemble_lineage(
        name=name, version=str(version), tags=tags, source=mv.source, run_params=run_params,
        run_artifacts=_list_all(client, run_id), parent_artifacts=_list_all(client, tags.get(TAG + "parent_run_id")),
        deploy_events=events, n_predictions=log.count_by_version().get(str(version), 0),
    )
