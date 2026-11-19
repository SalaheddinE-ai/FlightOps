"""SQLite prediction / error / label log used for monitoring and lineage."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from flightops_ml.config import get_paths
from flightops_ml.utils import utc_now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  request_id TEXT UNIQUE,
  ts TEXT NOT NULL,
  model_name TEXT, model_version TEXT, threshold REAL,
  probability REAL, prediction INTEGER, risk_level TEXT,
  latency_ms REAL, features_json TEXT
);
CREATE TABLE IF NOT EXISTS api_errors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, endpoint TEXT, status_code INTEGER, error_type TEXT, message TEXT
);
CREATE TABLE IF NOT EXISTS labels (
  request_id TEXT PRIMARY KEY, label INTEGER NOT NULL, labeled_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_version ON predictions(model_version);
"""


class PredictionLog:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else get_paths().prediction_db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path, timeout=15)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def log_prediction(self, *, request_id: str, model_name: str, model_version: str, threshold: float,
                       probability: float, prediction: int, risk_level: str, latency_ms: float,
                       features: dict[str, Any]) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT OR IGNORE INTO predictions (request_id, ts, model_name, model_version, threshold,"
                " probability, prediction, risk_level, latency_ms, features_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (request_id, utc_now_iso(), model_name, str(model_version), threshold, probability,
                 prediction, risk_level, latency_ms, json.dumps(features)),
            )

    def log_error(self, endpoint: str, status_code: int, error_type: str, message: str) -> None:
        with self._conn() as con:
            con.execute(
                "INSERT INTO api_errors (ts, endpoint, status_code, error_type, message) VALUES (?,?,?,?,?)",
                (utc_now_iso(), endpoint, status_code, error_type, message[:500]),
            )

    def add_labels(self, labels: pd.DataFrame) -> int:
        """labels: columns request_id, label (0/1). Returns number of rows written."""
        if not {"request_id", "label"} <= set(labels.columns):
            raise ValueError("labels must have columns request_id and label")
        if not labels["label"].isin([0, 1]).all():
            raise ValueError("label must be 0 or 1")
        rows = [(str(r), int(lab), utc_now_iso()) for r, lab in zip(labels["request_id"], labels["label"], strict=True)]
        with self._conn() as con:
            con.executemany("INSERT OR REPLACE INTO labels VALUES (?,?,?)", rows)
        return len(rows)

    def predictions(self, since: str | None = None, model_version: str | None = None) -> pd.DataFrame:
        q, args = "SELECT * FROM predictions WHERE 1=1", []
        if since:
            q += " AND ts >= ?"
            args.append(since)
        if model_version:
            q += " AND model_version = ?"
            args.append(str(model_version))
        with self._conn() as con:
            df = pd.read_sql_query(q + " ORDER BY id", con, params=args)
        if df.empty:
            return df
        feats = pd.json_normalize(df["features_json"].map(json.loads))
        return pd.concat([df.drop(columns=["features_json"]), feats], axis=1)

    def errors(self) -> pd.DataFrame:
        with self._conn() as con:
            return pd.read_sql_query("SELECT * FROM api_errors ORDER BY id", con)

    def labels(self) -> pd.DataFrame:
        with self._conn() as con:
            return pd.read_sql_query("SELECT * FROM labels", con)

    def count_by_version(self) -> dict[str, int]:
        with self._conn() as con:
            rows = con.execute("SELECT model_version, COUNT(*) FROM predictions GROUP BY model_version").fetchall()
        return {str(v): int(c) for v, c in rows}
