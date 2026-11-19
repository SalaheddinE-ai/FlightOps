"""Model-adaptation simulation (SYNTHETIC).

Baseline vs. shifted data, original model vs. a retrained candidate. The shifted dataset is generated
with the AI4I-like synthetic generator, so everything here is a controlled experiment - it does NOT
represent real equipment degradation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from flightops_ml.config import load_model_config
from flightops_ml.data.schema import RAW_FEATURES
from flightops_ml.data.synthetic import generate_ai4i_like
from flightops_ml.evaluation.metrics import classification_metrics
from flightops_ml.monitoring.drift import dataset_drift_summary, feature_drift
from flightops_ml.registry.promotion import evaluate_gates


def run_adaptation_simulation(
    original_pipeline,
    original_threshold: float,
    baseline_df: pd.DataFrame,
    shift: dict[str, float],
    n_shifted: int = 4000,
    seed: int = 123,
) -> dict[str, Any]:
    """shift keys: air_shift_k, process_shift_k, torque_factor, wear_shift_min."""
    shifted = generate_ai4i_like(n=n_shifted, seed=seed, **shift)
    drift = feature_drift(baseline_df[RAW_FEATURES], shifted[RAW_FEATURES])
    summary = dataset_drift_summary(drift)

    s_train, s_test = train_test_split(shifted, test_size=0.4, stratify=shifted["machine_failure"], random_state=seed)
    orig_proba = original_pipeline.predict_proba(s_test[RAW_FEATURES])[:, 1]
    orig_m = classification_metrics(s_test["machine_failure"], orig_proba, original_threshold)

    # Retrain a candidate: same pipeline structure, refit on baseline + shifted-train data.
    candidate = clone(original_pipeline)
    clf = candidate.named_steps["clf"]
    if hasattr(clf, "early_stopping_rounds"):  # no eval set during this simulated refit
        clf.set_params(early_stopping_rounds=None, callbacks=None)
    train_df = pd.concat([baseline_df, s_train], ignore_index=True)
    candidate.fit(train_df[RAW_FEATURES], train_df["machine_failure"])
    cand_proba = candidate.predict_proba(s_test[RAW_FEATURES])[:, 1]
    cand_m = classification_metrics(s_test["machine_failure"], cand_proba, original_threshold)

    gates = evaluate_gates(cand_m, orig_m, load_model_config()["gates"], smoke_test_ok=True, data_validation_ok=True)
    return {
        "label": "SYNTHETIC drift scenario - not real equipment degradation",
        "shift": shift,
        "drift_by_feature": drift.to_dict(orient="records"),
        "drift_summary": summary,
        "original_model": orig_m,
        "retrained_candidate": cand_m,
        "delta": {k: cand_m[k] - orig_m[k] for k in ("pr_auc", "recall", "precision", "f1")},
        "gates": [g.as_dict() for g in gates],
        "would_pass_gates": all(g.passed for g in gates),
        "note": "Even if gates pass, a real candidate still needs explicit human approval before deployment.",
    }
