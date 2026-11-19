import numpy as np
import pandas as pd
import pytest

from flightops_ml.data.schema import RAW_FEATURES
from flightops_ml.monitoring.drift import (
    classify_drift,
    dataset_drift_summary,
    feature_drift,
    psi_numeric,
    simulate_shift,
)
from flightops_ml.monitoring.monitoring import build_monitoring_report, performance_with_labels
from flightops_ml.monitoring.prediction_log import PredictionLog


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi_numeric(a, a) < 1e-6
    assert psi_numeric(a, a + 2) > 0.5


def test_no_drift_on_same_distribution(synthetic_df):
    ref = synthetic_df[RAW_FEATURES].iloc[:1200]
    cur = synthetic_df[RAW_FEATURES].iloc[1200:2400]
    assert not dataset_drift_summary(feature_drift(ref, cur))["dataset_drift"]


@pytest.mark.parametrize("scenario,feature", [("temperature_shift", "air_temperature"),
                                               ("torque_increase", "torque"), ("tool_wear_shift", "tool_wear")])
def test_simulated_shifts_are_detected(synthetic_df, scenario, feature):
    ref = synthetic_df[RAW_FEATURES]
    fd = feature_drift(ref, simulate_shift(ref, scenario))
    assert fd.set_index("feature").loc[feature, "drifted"]


def test_unknown_scenario_rejected(synthetic_df):
    with pytest.raises(ValueError):
        simulate_shift(synthetic_df[RAW_FEATURES], "nope")


def test_covariate_drift_is_not_reported_as_performance_degradation():
    cls = classify_drift({"dataset_drift": True, "n_drifted": 3}, None, {"status": "insufficient_labels"})
    assert cls["covariate_drift"] and cls["performance"].startswith("not_assessed")
    assert any("NOT confirmed performance degradation" in n for n in cls["notes"])


def test_performance_requires_labels_and_minimum_count(tmp_path):
    log = PredictionLog(tmp_path / "p.db")
    for i in range(60):
        log.log_prediction(request_id=f"r{i}", model_name="m", model_version="1", threshold=0.5,
                           probability=0.9 if i % 6 == 0 else 0.1, prediction=int(i % 6 == 0),
                           risk_level="LOW", latency_ms=5.0, features={"type": "M"})
    preds = log.predictions()
    cfg = {"performance": {"min_labels": 30}}
    assert performance_with_labels(preds, log.labels(), cfg)["status"] == "insufficient_labels"
    labels = pd.DataFrame({"request_id": [f"r{i}" for i in range(60)], "label": [int(i % 6 == 0) for i in range(60)]})
    assert log.add_labels(labels) == 60
    perf = performance_with_labels(log.predictions(), log.labels(), cfg)
    assert perf["status"] == "ok" and perf["recall"] == 1.0 and perf["false_negative_rate"] == 0.0
    with pytest.raises(ValueError):
        log.add_labels(pd.DataFrame({"request_id": ["x"], "label": [2]}))


def test_monitoring_report_without_traffic(synthetic_df):
    rep = build_monitoring_report(synthetic_df[RAW_FEATURES], pd.DataFrame())
    assert rep["status"] == "NO_DATA" and rep["performance"]["status"] == "insufficient_labels"


def test_monitoring_report_detects_drift_in_logged_traffic(synthetic_df, tmp_path):
    ref = synthetic_df[RAW_FEATURES].iloc[:1500]
    shifted = simulate_shift(synthetic_df[RAW_FEATURES].iloc[1500:1700], "temperature_shift")
    log = PredictionLog(tmp_path / "p.db")
    for i, row in enumerate(shifted.to_dict(orient="records")):
        log.log_prediction(request_id=f"r{i}", model_name="m", model_version="1", threshold=0.5, probability=0.05,
                           prediction=0, risk_level="LOW", latency_ms=3.0, features=row)
    rep = build_monitoring_report(ref, log.predictions(), log.errors(), log.labels(), np.full(500, 0.05))
    assert rep["status"] == "ALERT" and rep["drift"]["dataset_drift"]
    assert rep["interpretation"]["covariate_drift"] and rep["interpretation"]["performance"].startswith("not_assessed")
