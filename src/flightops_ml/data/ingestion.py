"""CSV ingestion and dataset fingerprinting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml

from flightops_ml.data.schema import RAW_FEATURES, RAW_TO_SNAKE, TARGET
from flightops_ml.utils import git_commit, sha256_file


class DatasetNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class DatasetFingerprint:
    path: str
    sha256: str
    size_bytes: int
    n_rows: int
    n_columns: int
    version: str          # "dvc:<md5>" when a .dvc file exists, otherwise "sha256:<12 hex>"
    git_commit: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load the AI4I CSV and normalise column names to snake_case."""
    path = Path(path)
    if not path.exists():
        raise DatasetNotFoundError(
            f"Dataset not found: {path}\n"
            "Download ai4i2020.csv from Kaggle (AI4I 2020 Predictive Maintenance) and place it in "
            "data/raw/, or generate the SYNTHETIC DEMO file with: "
            "python -m flightops_ml demo-data  (then set FLIGHTOPS_DATASET)."
        )
    df = pd.read_csv(path)
    df = df.rename(columns=RAW_TO_SNAKE)
    return df


def select_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) restricted to the model input features; excluded columns are dropped."""
    missing = [c for c in [*RAW_FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return df[RAW_FEATURES].copy(), df[TARGET].astype(int).copy()


def _dvc_md5(path: Path) -> str | None:
    dvc_file = path.with_name(path.name + ".dvc")
    if not dvc_file.exists():
        return None
    with dvc_file.open("r", encoding="utf-8") as fh:
        meta = yaml.safe_load(fh) or {}
    outs = meta.get("outs") or []
    return outs[0].get("md5") if outs else None


def fingerprint_dataset(path: Path, df: pd.DataFrame | None = None) -> DatasetFingerprint:
    path = Path(path)
    sha = sha256_file(path)
    if df is None:
        df = pd.read_csv(path, nrows=None)
    md5 = _dvc_md5(path)
    return DatasetFingerprint(
        path=str(path),
        sha256=sha,
        size_bytes=path.stat().st_size,
        n_rows=int(df.shape[0]),
        n_columns=int(df.shape[1]),
        version=f"dvc:{md5}" if md5 else f"sha256:{sha[:12]}",
        git_commit=git_commit(path.parent),
    )
