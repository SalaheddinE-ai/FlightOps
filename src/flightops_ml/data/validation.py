"""Schema validation (Pandera) and data-quality reporting.

Only physically-necessary constraints are enforced (positive Kelvin temperatures, positive speed
and torque, non-negative wear, binary target). No industrial operating limits are invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

from flightops_ml.data.schema import (
    LEAKAGE_COLUMNS,
    NUMERIC_FEATURES,
    PRODUCT_TYPES,
    RAW_FEATURES,
    TARGET,
)
from flightops_ml.utils import write_json_atomic


class DataValidationError(ValueError):
    def __init__(self, message: str, failures: list[dict[str, Any]] | None = None):
        super().__init__(message)
        self.failures = failures or []


def build_schema(require_target: bool = True) -> DataFrameSchema:
    columns: dict[str, Column] = {
        "type": Column(str, Check.isin(PRODUCT_TYPES), nullable=False),
        "air_temperature": Column(float, Check.gt(0), nullable=False, coerce=True),
        "process_temperature": Column(float, Check.gt(0), nullable=False, coerce=True),
        "rotational_speed": Column(float, Check.gt(0), nullable=False, coerce=True),
        "torque": Column(float, Check.gt(0), nullable=False, coerce=True),
        "tool_wear": Column(float, Check.ge(0), nullable=False, coerce=True),
    }
    if require_target:
        columns[TARGET] = Column(int, Check.isin([0, 1]), nullable=False, coerce=True)
    return DataFrameSchema(columns, strict=False)


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    try:
        return build_schema(require_target).validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases.head(50).to_dict(orient="records")
        raise DataValidationError(
            f"Schema validation failed with {len(exc.failure_cases)} failure case(s)", failures
        ) from exc


@dataclass
class QualityReport:
    passed: bool
    n_rows: int
    n_columns: int
    missing_values: dict[str, int]
    duplicate_rows: int
    duplicate_ids: int
    dtypes: dict[str, str]
    target_distribution: dict[str, Any]
    numeric_profile: dict[str, dict[str, float]]
    categorical_distribution: dict[str, dict[str, int]]
    feature_histograms: dict[str, dict[str, list[float]]]
    warnings: list[str] = field(default_factory=list)
    schema_failures: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_quality_report(df: pd.DataFrame, bins: int = 20) -> QualityReport:
    warnings: list[str] = []
    schema_failures: list[dict[str, Any]] = []
    try:
        validate_schema(df)
        passed = True
    except DataValidationError as exc:
        passed = False
        schema_failures = exc.failures
        warnings.append(str(exc))

    present_features = [c for c in RAW_FEATURES if c in df.columns]
    feature_df = df[present_features]
    missing = {c: int(df[c].isna().sum()) for c in df.columns}
    if sum(missing.values()):
        warnings.append("Dataset contains missing values.")
    duplicate_rows = int(feature_df.duplicated().sum()) if present_features else 0
    duplicate_ids = int(df["udi"].duplicated().sum()) if "udi" in df.columns else 0

    target_dist: dict[str, Any] = {}
    if TARGET in df.columns:
        counts = df[TARGET].value_counts().sort_index()
        total = int(counts.sum())
        target_dist = {
            "counts": {str(k): int(v) for k, v in counts.items()},
            "rates": {str(k): float(v / total) for k, v in counts.items()},
        }
        pos = int(counts.get(1, 0))
        if pos == 0:
            warnings.append("No positive (failure) samples present.")
        else:
            target_dist["imbalance_ratio_neg_to_pos"] = float(counts.get(0, 0) / pos)

    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    profile = {
        c: {str(k): float(v) for k, v in df[c].describe().items()}
        for c in numeric_cols
        if pd.api.types.is_numeric_dtype(df[c])
    }
    hist: dict[str, dict[str, list[float]]] = {}
    for c in profile:
        counts_, edges = np.histogram(df[c].dropna(), bins=bins)
        hist[c] = {"counts": counts_.tolist(), "edges": edges.tolist()}
    cat_dist = {
        c: {str(k): int(v) for k, v in df[c].value_counts().items()}
        for c in ("type",)
        if c in df.columns
    }
    if {"air_temperature", "process_temperature"} <= set(df.columns):
        n_bad = int((df["process_temperature"] <= df["air_temperature"]).sum())
        if n_bad:
            warnings.append(
                f"{n_bad} rows have process temperature <= air temperature (informational only)."
            )
    leaks = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if leaks:
        warnings.append(f"Leakage columns present in raw data and excluded from inputs: {leaks}")

    return QualityReport(
        passed=passed,
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        missing_values=missing,
        duplicate_rows=duplicate_rows,
        duplicate_ids=duplicate_ids,
        dtypes={str(c): str(t) for c, t in df.dtypes.items()},
        target_distribution=target_dist,
        numeric_profile=profile,
        categorical_distribution=cat_dist,
        feature_histograms=hist,
        warnings=warnings,
        schema_failures=schema_failures,
    )


def validate_and_report(df: pd.DataFrame, out_path: Path | None = None) -> QualityReport:
    report = build_quality_report(df)
    if out_path is not None:
        write_json_atomic(out_path, report.as_dict())
    if not report.passed:
        raise DataValidationError("Dataset failed validation", report.schema_failures)
    return report
