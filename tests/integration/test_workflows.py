import pandas as pd
import pytest

from flightops_ml.data.synthetic import write_demo_csv
from flightops_ml.monitoring.adaptation import run_adaptation_simulation
from flightops_ml.registry.registration import STATUS_TAG, get_champion
from flightops_ml.tracking.lineage import fetch_lineage
from flightops_ml.training import retraining
from tests.conftest import FAST_OVERRIDES


def test_retraining_creates_candidate_but_never_deploys(trained_session, tmp_path):
    c, name, v1 = trained_session["client"], trained_session["name"], trained_session["version"]
    new = write_demo_csv(tmp_path / "new.csv", n=2500, seed=99)
    out = retraining.run_retraining(new, trigger="new_labeled_data", overrides={
        **FAST_OVERRIDES, "models": {**FAST_OVERRIDES["models"], "enabled": ["random_forest"]}})
    assert out["candidate_version"] and out["candidate_version"] != v1
    assert c.get_model_version(name, out["candidate_version"]).tags[STATUS_TAG] in {"EXPERIMENTAL", "VALIDATED"}
    assert str(get_champion(c, name).version) == v1  # no automatic promotion
    versioned = list((trained_session["root"] / "data" / "processed" / "versions").glob("ai4i_*.csv"))
    assert versioned


def test_retraining_rejects_unlabeled_or_invalid_data(trained_session, tmp_path):
    df = pd.read_csv(write_demo_csv(tmp_path / "x.csv", n=300, seed=1))
    df.drop(columns=["Machine failure"]).to_csv(tmp_path / "nolabel.csv", index=False)
    with pytest.raises(retraining.RetrainingError):
        retraining.run_retraining(tmp_path / "nolabel.csv")
    df.assign(**{"Machine failure": 0}).to_csv(tmp_path / "oneclass.csv", index=False)
    with pytest.raises(retraining.RetrainingError):
        retraining.run_retraining(tmp_path / "oneclass.csv")
    with pytest.raises(retraining.RetrainingError):
        retraining.run_retraining(tmp_path / "x.csv", trigger="bogus")


def test_lineage_from_real_registry_metadata(trained_session):
    g = fetch_lineage(trained_session["client"], trained_session["name"], trained_session["version"])
    status = {n["id"]: n["status"] for n in g["nodes"]}
    for k in ("dataset", "validation_report", "preprocessing", "training_config", "run", "model_version"):
        assert status[k] == "present", (k, g["missing"])
    assert next(n for n in g["nodes"] if n["id"] == "dataset")["details"]["sha256"]


def test_adaptation_simulation_is_labelled_synthetic(trained_session):
    res = trained_session["result"]
    base = res.splits.X_train.assign(machine_failure=res.splits.y_train)
    out = run_adaptation_simulation(res.best.pipeline, res.best.threshold, base,
                                    {"air_shift_k": 3.0, "process_shift_k": 3.0}, n_shifted=1500)
    assert "SYNTHETIC" in out["label"]
    assert out["drift_summary"]["n_drifted"] >= 1
    assert set(out["delta"]) == {"pr_auc", "recall", "precision", "f1"}
    assert "human approval" in out["note"]


def test_shap_explanation_for_tree_model(trained_session):
    from flightops_ml.evaluation.explainability import explain_row, supports_shap

    pipe = trained_session["result"].best.pipeline
    if not supports_shap(pipe):
        pytest.skip("selected model is not tree-based")
    ex = explain_row(pipe, trained_session["result"].splits.X_test.head(1))
    assert ex["contributions"] and "feature" in ex["contributions"][0]
