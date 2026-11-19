"""Dataset access for the dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from flightops_ml.config import get_paths
from flightops_ml.data.ingestion import DatasetNotFoundError, load_raw_csv


@st.cache_data(show_spinner=False)
def load_dataset(path: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache when the file changes
    return load_raw_csv(path)


def get_dataset() -> pd.DataFrame | None:
    path = get_paths().dataset_path()
    if not path.exists():
        return None
    try:
        return load_dataset(str(path), path.stat().st_mtime)
    except DatasetNotFoundError:
        return None


def dataset_missing_message() -> str:
    return (f"Dataset not found at `{get_paths().dataset_path()}`. Place the Kaggle `ai4i2020.csv` in "
            "`data/raw/`, or generate the SYNTHETIC demo file: `python -m flightops_ml demo-data` and set "
            "`FLIGHTOPS_DATASET`.")
