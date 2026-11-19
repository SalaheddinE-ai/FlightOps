"""Dataset constants shared across the whole project."""

from __future__ import annotations

RAW_TO_SNAKE = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear",
]
RAW_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "machine_failure"

# Failure-mode columns are components of the target -> they MUST NOT be predictors (target leakage).
LEAKAGE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]
IDENTIFIER_COLUMNS = ["udi", "product_id"]
EXCLUDED_FROM_MODEL = IDENTIFIER_COLUMNS + LEAKAGE_COLUMNS

PRODUCT_TYPES = ["L", "M", "H"]
