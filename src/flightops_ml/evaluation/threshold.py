"""Decision-threshold analysis. Thresholds are chosen on the VALIDATION set only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STRATEGIES = ("default", "best_f1", "recall_constrained", "cost")


def threshold_sweep(
    y_true,
    proba,
    thresholds=None,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> pd.DataFrame:
    """Confusion counts and metrics at each threshold (vectorised)."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    ths = np.linspace(0.01, 0.99, 99) if thresholds is None else np.asarray(thresholds, dtype=float)
    rows = []
    pos = int(y.sum())
    for t in ths:
        pred = p >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(pos - tp)
        tn = int(len(y) - tp - fp - fn)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / pos if pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cost": fn * cost_fn + fp * cost_fp,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ThresholdChoice:
    strategy: str
    threshold: float
    precision: float
    recall: float
    f1: float
    constraint_met: bool = True

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _choice(strategy: str, row: pd.Series, met: bool = True) -> ThresholdChoice:
    return ThresholdChoice(
        strategy, float(row.threshold), float(row.precision), float(row.recall), float(row.f1), met
    )


def select_thresholds(
    y_val,
    proba_val,
    recall_target: float = 0.80,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
) -> dict[str, ThresholdChoice]:
    """Return the four candidate thresholds computed on validation data."""
    sweep = threshold_sweep(y_val, proba_val, cost_fn=cost_fn, cost_fp=cost_fp)
    default = threshold_sweep(y_val, proba_val, thresholds=[0.5]).iloc[0]

    best_f1 = sweep.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]

    feasible = sweep[sweep.recall >= recall_target]
    if len(feasible):
        rc = feasible.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
        rc_choice = _choice("recall_constrained", rc, True)
    else:
        rc = sweep.sort_values(["recall", "threshold"], ascending=[False, False]).iloc[0]
        rc_choice = _choice("recall_constrained", rc, False)

    cost = sweep.sort_values(["cost", "threshold"], ascending=[True, False]).iloc[0]
    return {
        "default": _choice("default", default),
        "best_f1": _choice("best_f1", best_f1),
        "recall_constrained": rc_choice,
        "cost": _choice("cost", cost),
    }
