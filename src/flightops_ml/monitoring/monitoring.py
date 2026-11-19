"""Monitoring report builder: volume, latency, errors, drift, and performance when labels exist."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from flightops_ml.config import get_paths, load_monitoring_config
from flightops_ml.data.schema import NUMERIC_FEATURES, RAW_FEATURES
from flightops_ml.evaluation.metrics import classification_metrics
from flightops_ml.monitoring.drift import (
    classify_drift,
    dataset_drift_summary,
    feature_drift,
    prediction_drift,
)
from flightops_ml.utils import utc_now_iso, write_json_atomic


def performance_with_labels(
    predictions: pd.DataFrame, labels: pd.DataFrame, cfg: dict[str, Any], baseline_recall: float | None = None
) -> dict[str, Any]:
    """Performance ONLY from labelled predictions. Never computed from unlabeled data."""
    if predictions.empty or labels.empty:
        return {"status": "insufficient_labels", "n_labels": 0}
    joined = predictions.merge(labels, on="request_id", how="inner")
    if len(joined) < cfg["performance"]["min_labels"]:
        return {"status": "insufficient_labels", "n_labels": int(len(joined))}
    thr = float(joined["threshold"].iloc[-1])
    m = classification_metrics(joined["label"], joined["probability"], thr)
    out = {"status": "ok", "n_labels": int(len(joined)),
           **{k: m[k] for k in ("precision", "recall", "f1", "pr_auc", "false_negative_rate")}}
    if baseline_recall is not None and not np.isnan(m["recall"]):
        out["degraded"] = bool(m["recall"] < baseline_recall - 0.10)
        out["baseline_recall"] = baseline_recall
    return out


def build_monitoring_report(
    reference_features: pd.DataFrame,
    predictions: pd.DataFrame,
    errors: pd.DataFrame | None = None,
    labels: pd.DataFrame | None = None,
    reference_proba: np.ndarray | None = None,
    baseline_recall: float | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_monitoring_config()
    errors = errors if errors is not None else pd.DataFrame()
    labels = labels if labels is not None else pd.DataFrame()
    n = len(predictions)
    report: dict[str, Any] = {"generated_at": utc_now_iso(), "n_predictions": n, "alerts": []}
    report["errors"] = {"count": int(len(errors)),
                        "rate": float(len(errors) / (n + len(errors))) if (n + len(errors)) else 0.0}
    if n:
        report["model_version_usage"] = predictions["model_version"].value_counts().to_dict()
        report["latency_ms"] = {"p50": float(predictions["latency_ms"].quantile(0.5)),
                                "p95": float(predictions["latency_ms"].quantile(0.95))}
        report["probability_distribution"] = {
            "mean": float(predictions["probability"].mean()),
            "p95": float(predictions["probability"].quantile(0.95)),
            "positive_rate": float(predictions["prediction"].mean()),
        }
        feats = predictions[[c for c in RAW_FEATURES if c in predictions]]
        report["missing_rate"] = float(feats[[c for c in NUMERIC_FEATURES if c in feats]].isna().mean().mean())
    drift_summary: dict[str, Any] = {"status": "insufficient_data"}
    pred_drift = None
    if n >= cfg["drift"]["min_rows"]:
        fd = feature_drift(reference_features, predictions[[c for c in RAW_FEATURES if c in predictions]], cfg["drift"])
        drift_summary = {"status": "ok", **dataset_drift_summary(fd, cfg["drift"]),
                         "features": fd.to_dict(orient="records")}
        if reference_proba is not None:
            pred_drift = prediction_drift(reference_proba, predictions["probability"].to_numpy(), cfg["drift"])
    report["drift"] = drift_summary
    report["prediction_drift"] = pred_drift
    perf = performance_with_labels(predictions, labels, cfg, baseline_recall)
    report["performance"] = perf
    report["interpretation"] = classify_drift(drift_summary, pred_drift, perf)

    a = cfg["alerts"]
    if report["errors"]["rate"] > a["error_rate"]:
        report["alerts"].append(f"API error rate {report['errors']['rate']:.1%} above {a['error_rate']:.1%}")
    if n and report["latency_ms"]["p95"] > a["latency_p95_ms"]:
        report["alerts"].append(f"p95 latency {report['latency_ms']['p95']:.0f} ms above {a['latency_p95_ms']} ms")
    if n and report["missing_rate"] > a["missing_rate"]:
        report["alerts"].append("Missing-value rate above threshold")
    if drift_summary.get("dataset_drift"):
        report["alerts"].append(f"Dataset drift: {drift_summary['drifted_features']} (covariate drift)")
    if perf.get("degraded"):
        report["alerts"].append("Recall dropped versus baseline on labelled predictions")
    report["status"] = "ALERT" if report["alerts"] else ("NO_DATA" if n == 0 else "OK")
    return report


def save_monitoring_report(report: dict[str, Any], out_dir: Path | None = None) -> Path:
    out_dir = out_dir or get_paths().monitoring_reports
    path = Path(out_dir) / f"monitoring_{time.strftime('%Y%m%d-%H%M%S')}.json"
    write_json_atomic(path, report)
    return path


def list_monitoring_reports(out_dir: Path | None = None) -> list[Path]:
    out_dir = Path(out_dir or get_paths().monitoring_reports)
    return sorted(out_dir.glob("monitoring_*.json"), reverse=True)
