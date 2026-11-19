import numpy as np
import pandas as pd
import pytest

from flightops_ml.data.ingestion import DatasetNotFoundError, load_raw_csv, select_model_frame
from flightops_ml.data.preprocessing import (
    LeakageError,
    assert_no_leakage,
    build_preprocessor,
    split_dataset,
)
from flightops_ml.data.schema import EXCLUDED_FROM_MODEL, RAW_FEATURES
from flightops_ml.data.synthetic import write_demo_csv
from flightops_ml.data.validation import DataValidationError, build_quality_report, validate_schema


def test_loader_renames_columns_and_missing_file_message(tmp_path):
    with pytest.raises(DatasetNotFoundError, match="ai4i2020"):
        load_raw_csv(tmp_path / "nope.csv")
    p = write_demo_csv(tmp_path / "d.csv", n=200, seed=1)
    df = load_raw_csv(p)
    assert {"air_temperature", "machine_failure", "twf"} <= set(df.columns)


def test_model_frame_excludes_leakage_columns(synthetic_df):
    X, y = select_model_frame(synthetic_df)
    assert list(X.columns) == RAW_FEATURES
    assert not set(EXCLUDED_FROM_MODEL) & set(X.columns)
    assert set(y.unique()) <= {0, 1}


def test_schema_accepts_valid_and_rejects_invalid(synthetic_df):
    validate_schema(synthetic_df)
    bad = synthetic_df.copy()
    bad.loc[0, "torque"] = -5
    bad.loc[1, "type"] = "X"
    with pytest.raises(DataValidationError) as e:
        validate_schema(bad)
    assert e.value.failures


def test_quality_report_contents(synthetic_df):
    rep = build_quality_report(synthetic_df)
    assert rep.passed and rep.n_rows == len(synthetic_df)
    assert rep.target_distribution["imbalance_ratio_neg_to_pos"] > 1
    assert "air_temperature" in rep.numeric_profile
    assert any("Leakage" in w for w in rep.warnings)
    dup = pd.concat([synthetic_df, synthetic_df.head(5)])
    assert build_quality_report(dup).duplicate_rows >= 5


def test_split_proportions_stratification_and_disjointness(synthetic_df):
    X, y = select_model_frame(synthetic_df)
    sp = split_dataset(X, y, seed=42)
    n = len(X)
    assert abs(len(sp.X_train) / n - 0.70) < 0.01
    assert abs(len(sp.X_val) / n - 0.10) < 0.01
    assert abs(len(sp.X_test) / n - 0.20) < 0.01
    rates = [sp.y_train.mean(), sp.y_val.mean(), sp.y_test.mean()]
    assert max(rates) - min(rates) < 0.01
    assert_no_leakage(sp)
    sp2 = split_dataset(X, y, seed=42)
    assert sp.X_test.index.equals(sp2.X_test.index)  # fixed seed -> reproducible


def test_leakage_detection(synthetic_df):
    X, y = select_model_frame(synthetic_df)
    sp = split_dataset(X, y, seed=1)
    sp.X_train = sp.X_train.assign(twf=0)
    with pytest.raises(LeakageError):
        assert_no_leakage(sp)
    sp = split_dataset(X, y, seed=1)
    sp.X_val = pd.concat([sp.X_val, sp.X_test.head(3)])
    with pytest.raises(LeakageError):
        assert_no_leakage(sp)


def test_preprocessor_is_fit_on_training_data_only(synthetic_df):
    X, y = select_model_frame(synthetic_df)
    sp = split_dataset(X, y, seed=3)
    pre = build_preprocessor(["temp_diff", "power_w", "wear_torque"])
    Xt = pre.fit_transform(sp.X_train)
    scaler = pre.named_steps["ct"].named_transformers_["num"]
    assert np.isclose(scaler.mean_[0], sp.X_train["air_temperature"].mean())
    assert not np.isclose(scaler.mean_[0], X["air_temperature"].mean(), atol=1e-12)
    assert abs(Xt["torque"].mean()) < 1e-6  # standardised on train
    out = pre.transform(sp.X_test)
    assert list(out.columns) == list(Xt.columns)


def test_unseen_category_does_not_crash(synthetic_df):
    X, y = select_model_frame(synthetic_df)
    pre = build_preprocessor(["temp_diff"]).fit(X)
    odd = X.head(2).copy()
    odd["type"] = "Z"
    assert pre.transform(odd).shape[0] == 2
