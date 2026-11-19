"""Feature engineering implemented as a scikit-learn transformer.

Because it lives inside the fitted Pipeline that is logged to MLflow, training and inference apply
exactly the same transformations.

Justifications
--------------
temp_diff           process - air temperature [K]. Physical: heat-dissipation proxy; a small gap
                    combined with low speed is the documented heat-dissipation failure condition.
power_w             torque [Nm] * rotational speed [rad/s] = mechanical power [W]. Physical: the
                    documented power-failure mode depends on this product.
wear_torque         tool_wear [min] * torque [Nm]. Physical/statistical: overstrain is driven by the
                    product of wear and torque.
torque_temp_ratio   torque / process temperature [Nm/K]. Exploratory: statistical temperature-to-
                    torque relationship; disabled by default (weaker physical justification).

NOTE: AI4I 2020 is synthetic and its failure modes were generated from rules on these quantities,
so these features are expected to be very informative. That reflects the data-generating process,
not necessarily real equipment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_ENGINEERING_VERSION = "1.0.0"

AVAILABLE_FEATURES = ("temp_diff", "power_w", "wear_torque", "torque_temp_ratio")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str] | tuple[str, ...] = ("temp_diff", "power_w", "wear_torque")):
        self.features = features

    def _validate(self) -> list[str]:
        unknown = [f for f in self.features if f not in AVAILABLE_FEATURES]
        if unknown:
            raise ValueError(f"Unknown engineered features: {unknown}. Available: {AVAILABLE_FEATURES}")
        return list(self.features)

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - sklearn convention
        self._validate()
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        feats = self._validate()
        out = X.copy()
        if "temp_diff" in feats:
            out["temp_diff"] = out["process_temperature"] - out["air_temperature"]
        if "power_w" in feats:
            out["power_w"] = out["torque"] * out["rotational_speed"] * 2.0 * np.pi / 60.0
        if "wear_torque" in feats:
            out["wear_torque"] = out["tool_wear"] * out["torque"]
        if "torque_temp_ratio" in feats:
            out["torque_temp_ratio"] = out["torque"] / out["process_temperature"]
        return out

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(self.feature_names_in_)
        return np.asarray(base + self._validate(), dtype=object)
