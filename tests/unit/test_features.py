import numpy as np
import pandas as pd
import pytest

from flightops_ml.features.engineering import FeatureEngineer


def _row():
    return pd.DataFrame([{"type": "M", "air_temperature": 300.0, "process_temperature": 310.5,
                          "rotational_speed": 1500.0, "torque": 40.0, "tool_wear": 100.0}])


def test_engineered_values_are_physically_correct():
    out = FeatureEngineer(["temp_diff", "power_w", "wear_torque", "torque_temp_ratio"]).fit_transform(_row())
    assert out.loc[0, "temp_diff"] == pytest.approx(10.5)
    assert out.loc[0, "power_w"] == pytest.approx(40 * 1500 * 2 * np.pi / 60)
    assert out.loc[0, "wear_torque"] == pytest.approx(4000)
    assert out.loc[0, "torque_temp_ratio"] == pytest.approx(40 / 310.5)


def test_unknown_feature_rejected_and_input_not_mutated():
    with pytest.raises(ValueError):
        FeatureEngineer(["nope"]).fit(_row())
    df = _row()
    FeatureEngineer().fit_transform(df)
    assert "temp_diff" not in df.columns
