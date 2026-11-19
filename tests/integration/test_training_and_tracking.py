import json

import mlflow
import numpy as np
import pytest

from flightops_ml.data.schema import EXCLUDED_FROM_MODEL
from flightops_ml.tracking.mlflow_tracking import get_client
from tests.conftest import FAST_OVERRIDES


def test_three_models_trained_and_selected_on_validation(trained_session):
    res = trained_session["result"]
    assert {r.name for r in res.results} == {"logistic_regression", "random_forest", "xgboost"}
    best = max(res.results, key=lambda r: r.val_metrics["pr_auc"])
    assert res.champion_name == best.name  # selection by validation PR-AUC, not accuracy / test
    for r in res.results:
        assert 0.0 < r.threshold < 1.0
        assert not any(c in EXCLUDED_FROM_MODEL for c in r.pipeline.named_steps["pre"].feature_names_in_)


def test_run_records_params_metrics_artifacts_and_metadata(trained_session):
    client = get_client()
    xgb = next(r for r in trained_session["result"].results if r.name == "xgboost")
    run = client.get_run(xgb.run_id)
    for p in ("algorithm", "learning_rate", "n_estimators", "max_depth", "random_seed", "decision_threshold",
              "feature_engineering", "validation_strategy", "class_weighting"):
        assert p in run.data.params, p
    for m in ("val_pr_auc", "test_f1", "test_recall", "test_roc_auc", "test_accuracy", "test_precision",
              "training_duration_s", "inference_latency_ms_p50", "best_iteration"):
        assert m in run.data.metrics, m
    for t in ("flightops.dataset_version", "flightops.dataset_sha256", "flightops.git_commit", "flightops.python"):
        assert t in run.data.tags, t
    arts = {a.path for a in client.list_artifacts(xgb.run_id)}
    assert {"figures", "explainability", "config", "evaluation", "training"} <= arts
    figs = {a.path for a in client.list_artifacts(xgb.run_id, "figures")}
    assert {"figures/confusion_matrix.png", "figures/roc_curve.png", "figures/pr_curve.png"} <= figs


def test_real_training_history_is_logged_and_consistent(trained_session):
    client = get_client()
    xgb = next(r for r in trained_session["result"].results if r.name == "xgboost")
    assert xgb.history is not None
    n = xgb.history["n_iterations_run"]
    assert len(xgb.history["train"]["logloss"]) == n == len(xgb.history["validation"]["aucpr"])
    stepped = client.get_metric_history(xgb.run_id, "validation_logloss")
    assert len(stepped) == n and [m.step for m in stepped][:3] == [1, 2, 3]
    local = mlflow.artifacts.load_dict(f"runs:/{xgb.run_id}/training/training_history.json")
    assert local["best_iteration"] == xgb.history["best_iteration"]


def test_optuna_trials_logged_as_nested_runs(trained_session):
    client = get_client()
    xgb = next(r for r in trained_session["result"].results if r.name == "xgboost")
    exp = client.get_run(xgb.run_id).info.experiment_id
    trials = client.search_runs([exp], f"tags.mlflow.parentRunId = '{xgb.run_id}'")
    assert len(trials) == FAST_OVERRIDES["tuning"]["n_trials"]
    assert all(t.data.tags.get("flightops.kind") == "optuna_trial" for t in trials)


def test_reproducibility_same_seed_same_metrics(isolated_root):
    from flightops_ml.training.train import run_training_pipeline

    ov = {**FAST_OVERRIDES, "models": {**FAST_OVERRIDES["models"], "enabled": ["random_forest"]}}
    a = run_training_pipeline(ov, register=False)
    b = run_training_pipeline(ov, register=False)
    # Multi-threaded tree ensembles sum per-tree outputs in a non-deterministic order, so results can differ
    # in the last floating-point bits. Compare with a tight tolerance instead of exact equality.
    for key in a.results[0].val_metrics:
        assert a.results[0].val_metrics[key] == pytest.approx(b.results[0].val_metrics[key], abs=1e-9, nan_ok=True)
        assert a.results[0].test_metrics[key] == pytest.approx(b.results[0].test_metrics[key], abs=1e-9, nan_ok=True)


def test_leakage_columns_never_reach_the_model(trained_session):
    pipe = trained_session["result"].best.pipeline
    names = set(pipe.named_steps["pre"].get_feature_names_out())
    assert not names & set(EXCLUDED_FROM_MODEL)


def test_model_signature_and_input_example_stored(trained_session):
    get_client()  # point MLflow's global registry URI at the session store
    info = mlflow.models.get_model_info(f"models:/{trained_session['name']}/{trained_session['version']}")
    assert info.signature is not None
    assert "air_temperature" in [c.name for c in info.signature.inputs.inputs]


def test_serialization_roundtrip_matches_original(trained_session):
    get_client()
    res = trained_session["result"]
    loaded = mlflow.sklearn.load_model(f"models:/{trained_session['name']}/{trained_session['version']}")
    x = res.splits.X_test.head(50)
    np.testing.assert_allclose(loaded.predict_proba(x)[:, 1], res.best.pipeline.predict_proba(x)[:, 1], atol=1e-9)


def test_summary_and_reference_files_written(trained_session):
    root = trained_session["root"]
    summary = json.loads((root / "reports" / "evaluations" / "latest_summary.json").read_text())
    assert summary["selected_model"] == trained_session["result"].champion_name
    assert (root / "data" / "reference" / "reference_features.csv").exists()
    assert (root / "reports" / "evaluations" / "data_quality_report.json").exists()


@pytest.mark.parametrize("flag", ["unknown"])
def test_unknown_model_name_rejected(isolated_root, flag):
    from flightops_ml.training.train import run_training_pipeline

    with pytest.raises((ValueError, KeyError)):
        run_training_pipeline({"models": {"enabled": [flag]}, "tuning": {"enabled": False}}, register=False)
