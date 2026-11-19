"""Optuna hyper-parameter optimisation with stratified CV inside the TRAINING partition only."""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from flightops_ml.data.preprocessing import build_preprocessor
from flightops_ml.training.models import build_xgboost, compute_scale_pos_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _cv_score(estimator, X, y, folds: int, seed: int, metric: str) -> float:  # noqa: N803
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(estimator, X, y, cv=cv, scoring=metric, n_jobs=1)))


def evaluate_class_weighting(
    X_train, y_train, engineered, xgb_params, seed: int, folds: int, metric: str  # noqa: N803
) -> dict[str, float]:
    """Compare scale_pos_weight=1 vs. negative/positive using CV on TRAIN. No assumption of benefit."""
    ratio = compute_scale_pos_weight(y_train)
    out = {}
    for label, spw in (("unweighted", 1.0), ("weighted", ratio)):
        pipe = Pipeline(
            [
                ("pre", build_preprocessor(engineered)),
                ("clf", build_xgboost({**xgb_params, "n_estimators": min(xgb_params.get("n_estimators", 200), 200)}, seed, spw)),
            ]
        )
        out[f"{label}_cv_{metric}"] = _cv_score(pipe, X_train, y_train, folds, seed, metric)
    out["scale_pos_weight_ratio"] = ratio
    return out


def tune_xgboost(
    X_train: pd.DataFrame,  # noqa: N803
    y_train: pd.Series,
    engineered: list[str],
    seed: int,
    n_trials: int,
    cv_folds: int,
    metric: str = "average_precision",
    timeout: float | None = None,
    log_trials_to_mlflow: bool = True,
    max_estimators: int = 500,
) -> tuple[dict[str, Any], pd.DataFrame, optuna.Study]:
    ratio = compute_scale_pos_weight(y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, max_estimators),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        # scale_pos_weight "when appropriate": let the search decide between none and a weighted range
        use_spw = trial.suggest_categorical("use_scale_pos_weight", [False, True])
        spw = trial.suggest_float("scale_pos_weight", 1.0, max(1.5, ratio), log=True) if use_spw else 1.0
        pipe = Pipeline(
            [("pre", build_preprocessor(engineered)), ("clf", build_xgboost(params, seed, spw))]
        )
        if log_trials_to_mlflow:
            with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
                mlflow.log_params({**params, "scale_pos_weight": spw})
                score = _cv_score(pipe, X_train, y_train, cv_folds, seed, metric)
                mlflow.log_metric(f"cv_{metric}", score)
                mlflow.set_tag("flightops.kind", "optuna_trial")
            return score
        return _cv_score(pipe, X_train, y_train, cv_folds, seed, metric)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed), study_name="xgboost"
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    best = dict(study.best_params)
    use_spw = best.pop("use_scale_pos_weight", False)
    spw_val = best.pop("scale_pos_weight", 1.0) if use_spw else 1.0
    best["scale_pos_weight"] = float(spw_val)
    trials_df = study.trials_dataframe()
    return best, trials_df, study
