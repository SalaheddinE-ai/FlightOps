"""Estimator factories."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")


def compute_scale_pos_weight(y_train) -> float:
    """negative/positive computed on TRAINING labels only."""
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    if pos == 0:
        raise ValueError("Training partition has no positive samples")
    return neg / pos


def build_xgboost(params: dict[str, Any], seed: int, scale_pos_weight: float | None, **extra):
    p = {
        k: v
        for k, v in params.items()
        if k
        in {
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "gamma",
            "reg_alpha",
            "reg_lambda",
        }
    }
    return XGBClassifier(
        objective="binary:logistic",
        tree_method="hist",
        random_state=seed,
        n_jobs=2,
        scale_pos_weight=scale_pos_weight if scale_pos_weight is not None else 1.0,
        **p,
        **extra,
    )


def build_sklearn_model(name: str, params: dict[str, Any], seed: int):
    if name == "logistic_regression":
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 2000),
            class_weight=params.get("class_weight"),
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth"),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            class_weight=params.get("class_weight"),
            random_state=seed,
            n_jobs=2,
        )
    raise ValueError(f"Unknown model: {name}")
