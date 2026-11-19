"""Drift statistics (PSI, KS, chi-square) and synthetic drift scenarios.

Definitions kept strictly separate:
* covariate drift      - input feature distributions changed (no labels needed)
* prediction drift     - the distribution of predicted probabilities changed
* performance decay    - ONLY assessable with ground-truth labels
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from flightops_ml.config import load_monitoring_config
from flightops_ml.data.schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RAW_FEATURES

EPS = 1e-6


def psi_numeric(reference, current, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.unique()) | set(current.unique()))
    r = reference.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    c = current.value_counts(normalize=True).reindex(cats, fill_value=0).clip(lower=EPS)
    return float(np.sum((c - r) * np.log(c / r)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = cfg or load_monitoring_config()["drift"]
    rows = []
    for col in RAW_FEATURES:
        if col not in reference or col not in current:
            continue
        if col in CATEGORICAL_FEATURES:
            cats = sorted(set(reference[col]) | set(current[col]))
            table = np.array([[(reference[col] == k).sum() for k in cats], [(current[col] == k).sum() for k in cats]])
            p = float(stats.chi2_contingency(table + 0)[1]) if table.shape[1] > 1 else 1.0
            psi = psi_categorical(reference[col], current[col])
            rows.append({"feature": col, "kind": "categorical", "test": "chi2", "p_value": p, "psi": psi,
                         "drifted": bool(psi >= cfg["psi_threshold"])})
        else:
            ks = stats.ks_2samp(reference[col].dropna(), current[col].dropna())
            psi = psi_numeric(reference[col].dropna(), current[col].dropna(), cfg["bins"])
            rows.append({"feature": col, "kind": "numeric", "test": "ks", "statistic": float(ks.statistic),
                         "p_value": float(ks.pvalue), "psi": psi, "drifted": bool(psi >= cfg["psi_threshold"]),
                         "ref_mean": float(reference[col].mean()), "cur_mean": float(current[col].mean())})
    return pd.DataFrame(rows)


def prediction_drift(ref_proba, cur_proba, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_monitoring_config()["drift"]
    ks = stats.ks_2samp(ref_proba, cur_proba)
    psi = psi_numeric(ref_proba, cur_proba, cfg["bins"])
    return {"ks_statistic": float(ks.statistic), "p_value": float(ks.pvalue), "psi": psi,
            "drifted": bool(psi >= cfg["psi_threshold"])}


def dataset_drift_summary(drift_df: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_monitoring_config()["drift"]
    n = len(drift_df)
    share = float(drift_df["drifted"].mean()) if n else 0.0
    return {"n_features": n, "n_drifted": int(drift_df["drifted"].sum()) if n else 0,
            "share_drifted": share, "dataset_drift": bool(share >= cfg["dataset_drift_share"]),
            "drifted_features": drift_df.loc[drift_df["drifted"], "feature"].tolist() if n else []}


def simulate_shift(df: pd.DataFrame, scenario: str, cfg: dict[str, Any] | None = None, **overrides) -> pd.DataFrame:
    """SYNTHETIC covariate shift applied to feature columns. Not a model of real equipment degradation."""
    scenarios = (cfg or load_monitoring_config())["simulation"]["scenarios"]
    if scenario not in scenarios:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(scenarios)}")
    p = {**scenarios[scenario], **overrides}
    out = df.copy()
    if scenario == "temperature_shift":
        out["air_temperature"] = out["air_temperature"] + p["air_delta_k"]
        out["process_temperature"] = out["process_temperature"] + p["process_delta_k"]
    elif scenario == "torque_increase":
        out["torque"] = out["torque"] * p["factor"]
    elif scenario == "tool_wear_shift":
        out["tool_wear"] = out["tool_wear"] + p["delta_min"]
    return out


def classify_drift(
    drift_summary: dict[str, Any], pred_drift: dict[str, Any] | None, performance: dict[str, Any] | None
) -> dict[str, Any]:
    """Explain WHICH kind of drift was observed; never conflate them."""
    covariate = bool(drift_summary.get("dataset_drift") or drift_summary.get("n_drifted", 0) > 0)
    pdrift = bool(pred_drift and pred_drift["drifted"])
    if performance is None or performance.get("status") != "ok":
        perf_state = "not_assessed (insufficient ground-truth labels)"
    else:
        perf_state = "confirmed_degradation" if performance.get("degraded") else "no_degradation_detected"
    notes = []
    if covariate and perf_state.startswith("not_assessed"):
        notes.append("Input distributions changed, but without labels this is covariate drift only - "
                     "NOT confirmed performance degradation.")
    if pdrift:
        notes.append("Predicted-probability distribution changed (prediction drift).")
    if not covariate and not pdrift:
        notes.append("No drift above the configured thresholds.")
    return {"covariate_drift": covariate, "prediction_drift": pdrift,
            "performance": perf_state, "notes": notes}


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out_html: Path) -> Path | None:
    """Optional Evidently HTML drift report (supports the 0.7+ and the legacy 0.4 APIs).

    Returns None when Evidently is not installed; the built-in drift statistics above are always
    available and do not depend on it.
    """
    cols = [c for c in [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES] if c in reference and c in current]
    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:  # Evidently >= 0.7
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(current_data=current[cols], reference_data=reference[cols])
        snapshot.save_html(str(out_html))
        return out_html
    except ImportError:
        pass
    try:  # legacy API
        from evidently.metric_preset import DataDriftPreset as LegacyPreset
        from evidently.report import Report as LegacyReport
    except ImportError:
        return None
    report = LegacyReport(metrics=[LegacyPreset()])
    report.run(reference_data=reference[cols], current_data=current[cols])
    report.save_html(str(out_html))
    return out_html
