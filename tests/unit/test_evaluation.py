import numpy as np
import pytest

from flightops_ml.evaluation.metrics import (
    classification_metrics,
    confusion_counts,
    save_evaluation_plots,
)
from flightops_ml.evaluation.threshold import select_thresholds, threshold_sweep


@pytest.fixture()
def scores():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    p = np.clip(0.05 + 0.6 * y * rng.random(2000) + 0.1 * rng.random(2000), 0, 1)
    return y, p


def test_confusion_and_metrics_consistent(scores):
    y, p = scores
    c = confusion_counts(y, p, 0.3)
    assert sum(c.values()) == len(y)
    m = classification_metrics(y, p, 0.3)
    assert m["recall"] == pytest.approx(c["tp"] / (c["tp"] + c["fn"]))
    assert 0 <= m["pr_auc"] <= 1 and m["brier"] >= 0


def test_single_class_gives_nan_not_fake_numbers():
    m = classification_metrics(np.zeros(10), np.linspace(0, 1, 10), 0.5)
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"])


def test_sweep_monotonic_and_counts(scores):
    y, p = scores
    sw = threshold_sweep(y, p)
    assert (sw.tp + sw.fn == y.sum()).all()
    assert (sw.recall.diff().dropna() <= 1e-12).all()  # recall never rises with the threshold


def test_four_threshold_strategies(scores):
    y, p = scores
    ch = select_thresholds(y, p, recall_target=0.8, cost_fn=10, cost_fp=1)
    assert set(ch) == {"default", "best_f1", "recall_constrained", "cost"}
    assert ch["default"].threshold == 0.5
    assert ch["recall_constrained"].recall >= 0.8 or not ch["recall_constrained"].constraint_met
    sw = threshold_sweep(y, p, cost_fn=10, cost_fp=1)
    assert sw.loc[sw.threshold.sub(ch["cost"].threshold).abs().idxmin(), "cost"] == sw.cost.min()
    # a higher false-negative cost can only push the threshold down (or keep it)
    ch2 = select_thresholds(y, p, cost_fn=100, cost_fp=1)
    assert ch2["cost"].threshold <= ch["cost"].threshold


def test_unreachable_recall_constraint_is_flagged(scores):
    y, p = scores
    assert select_thresholds(y, p, recall_target=1.01)["recall_constrained"].constraint_met is False


def test_plots_written(scores, tmp_path):
    y, p = scores
    paths = save_evaluation_plots(y, p, 0.3, tmp_path)
    assert {"confusion_matrix", "roc_curve", "pr_curve", "calibration"} <= set(paths)
    assert all(v.exists() for v in paths.values())
