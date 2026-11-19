from flightops_ml.config import load_model_config
from flightops_ml.inference.predictor import risk_level
from flightops_ml.registry.promotion import evaluate_gates

G = load_model_config()["gates"]
GOOD = {"pr_auc": 0.80, "recall": 0.70, "f1": 0.70}


def _names(res, passed):
    return {g.name for g in res if g.passed is passed}


def test_floors_only_without_champion():
    assert all(g.passed for g in evaluate_gates(GOOD, None, G, True, True))
    weak = {"pr_auc": 0.1, "recall": 0.1, "f1": 0.1}
    assert {"pr_auc_floor", "recall_floor", "f1_floor"} <= _names(evaluate_gates(weak, None, G, True, True), False)


def test_higher_accuracy_alone_never_promotes():
    champ = {"pr_auc": 0.85, "recall": 0.75, "f1": 0.72, "accuracy": 0.97}
    cand = {"pr_auc": 0.70, "recall": 0.60, "f1": 0.65, "accuracy": 0.99}  # better accuracy, worse where it matters
    failed = _names(evaluate_gates(cand, champ, G, True, True), False)
    assert "pr_auc_vs_champion" in failed and "recall_regression" in failed


def test_candidate_beating_champion_passes():
    champ = {"pr_auc": 0.78, "recall": 0.70, "f1": 0.68}
    assert all(g.passed for g in evaluate_gates({"pr_auc": 0.82, "recall": 0.71, "f1": 0.70}, champ, G, True, True))


def test_smoke_and_data_validation_gates():
    f = _names(evaluate_gates(GOOD, None, G, smoke_test_ok=False, data_validation_ok=False), False)
    assert f == {"smoke_test", "data_validation"}


def test_risk_levels():
    cfg = {"medium_fraction_of_threshold": 0.5, "critical_probability": 0.8}
    assert risk_level(0.05, 0.4, cfg) == "LOW"
    assert risk_level(0.25, 0.4, cfg) == "MEDIUM"
    assert risk_level(0.5, 0.4, cfg) == "HIGH"
    assert risk_level(0.95, 0.4, cfg) == "CRITICAL"
