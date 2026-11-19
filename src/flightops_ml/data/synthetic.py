"""SYNTHETIC AI4I-like data generator.

This is NOT the AI4I 2020 dataset. It follows the *published failure-mode rules* of AI4I 2020
(tool wear, heat dissipation, power, overstrain, random failures) so that tests, CI and the
drift/adaptation simulations can run without downloading anything. Any metric obtained on this data
says nothing about the real dataset, and even less about real equipment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flightops_ml.data.schema import RAW_TO_SNAKE

SNAKE_TO_RAW = {v: k for k, v in RAW_TO_SNAKE.items()}
OVERSTRAIN_LIMITS = {"L": 11000.0, "M": 12000.0, "H": 13000.0}


def generate_ai4i_like(
    n: int = 10_000,
    seed: int = 0,
    *,
    air_shift_k: float = 0.0,
    process_shift_k: float = 0.0,
    torque_factor: float = 1.0,
    wear_shift_min: float = 0.0,
) -> pd.DataFrame:
    """Generate an AI4I-like frame (snake_case columns, including failure-mode columns)."""
    rng = np.random.default_rng(seed)
    type_ = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300.0, 2.0, n) + air_shift_k
    process = air + 10.0 - air_shift_k + process_shift_k + rng.normal(0.0, 1.0, n)
    rpm = np.clip(rng.normal(1538.0, 179.0, n) + rng.exponential(30.0, n), 1168, 2886)
    # Power is drawn first and torque derived from it, which keeps torque and speed anti-correlated
    # (as in the original data) and the power-failure rate small.
    power_target = rng.normal(6450.0, 1050.0, n)
    torque = np.clip(power_target * 60.0 / (2 * np.pi * rpm), 3.8, 76.6) * torque_factor
    wear = np.clip(np.abs(rng.normal(108.0, 63.0, n)) + wear_shift_min, 0, 253 + wear_shift_min)

    power = torque * rpm * 2 * np.pi / 60.0
    temp_diff = process - air
    twf = (wear >= 200) & (wear <= 240) & (rng.random(n) < 0.12)
    hdf = (temp_diff < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    limit = np.vectorize(OVERSTRAIN_LIMITS.get)(type_)
    osf = (wear * torque) > limit
    rnf = rng.random(n) < 0.001
    failure = twf | hdf | pwf | osf | rnf

    return pd.DataFrame(
        {
            "udi": np.arange(1, n + 1),
            "product_id": [f"{t}{i:05d}" for t, i in zip(type_, rng.integers(10000, 99999, n), strict=True)],
            "type": type_,
            "air_temperature": np.round(air, 1),
            "process_temperature": np.round(process, 1),
            "rotational_speed": np.round(rpm).astype(int),
            "torque": np.round(torque, 1),
            "tool_wear": np.round(wear).astype(int),
            "machine_failure": failure.astype(int),
            "twf": twf.astype(int),
            "hdf": hdf.astype(int),
            "pwf": pwf.astype(int),
            "osf": osf.astype(int),
            "rnf": rnf.astype(int),
        }
    )


def write_demo_csv(path: Path, n: int = 10_000, seed: int = 0) -> Path:
    """Write the synthetic demo dataset using the ORIGINAL AI4I column names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_ai4i_like(n=n, seed=seed).rename(columns=SNAKE_TO_RAW)
    df.to_csv(path, index=False)
    return path
