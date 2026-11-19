"""Preprocessing pipeline and leakage-safe dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from flightops_ml.data.schema import (
    CATEGORICAL_FEATURES,
    EXCLUDED_FROM_MODEL,
    NUMERIC_FEATURES,
    RAW_FEATURES,
)
from flightops_ml.features.engineering import FeatureEngineer


class LeakageError(AssertionError):
    pass


def build_preprocessor(engineered: list[str] | tuple[str, ...]) -> Pipeline:
    """Unfitted preprocessing: feature engineering -> scaling/one-hot. Fit on TRAIN only."""
    numeric = NUMERIC_FEATURES + list(engineered)
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    ct.set_output(transform="pandas")
    return Pipeline([("fe", FeatureEngineer(features=list(engineered))), ("ct", ct)])


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.X_train), "validation": len(self.X_val), "test": len(self.X_test)}


def split_dataset(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    seed: int,
    train: float = 0.70,
    validation: float = 0.10,
    test: float = 0.20,
) -> DataSplits:
    """Stratified train/validation/test split with a fixed seed."""
    if abs(train + validation + test - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test, stratify=y, random_state=seed
    )
    val_fraction = validation / (train + validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction, stratify=y_tmp, random_state=seed
    )
    return DataSplits(X_train, y_train, X_val, y_val, X_test, y_test)


def assert_no_leakage(splits: DataSplits) -> None:
    """Raise LeakageError if split indices overlap or forbidden columns are used as inputs."""
    sets = {
        "train": set(splits.X_train.index),
        "validation": set(splits.X_val.index),
        "test": set(splits.X_test.index),
    }
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = sets[a] & sets[b]
            if overlap:
                raise LeakageError(f"{len(overlap)} rows overlap between {a} and {b}")
    for name, frame in (("train", splits.X_train), ("validation", splits.X_val), ("test", splits.X_test)):
        bad = [c for c in frame.columns if c in EXCLUDED_FROM_MODEL]
        if bad:
            raise LeakageError(f"{name} inputs contain excluded columns: {bad}")
        extra = [c for c in frame.columns if c not in RAW_FEATURES]
        if extra:
            raise LeakageError(f"{name} inputs contain unexpected columns: {extra}")
