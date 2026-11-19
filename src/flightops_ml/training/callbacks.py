"""XGBoost training callback that forwards real evaluation metrics to a progress reporter."""

from __future__ import annotations

import xgboost as xgb


class ProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self, reporter, total_iterations: int):
        self.reporter = reporter
        self.total = total_iterations

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        latest: dict[str, float] = {}
        for data_name, metrics in evals_log.items():
            split = "train" if data_name == "validation_0" else "val"
            for metric, values in metrics.items():
                latest[f"{split}_{metric}"] = float(values[-1])
        latest["max_iterations"] = float(self.total)
        self.reporter.iteration(epoch + 1, latest)
        return False  # never stop training from here; early stopping is handled by XGBoost
