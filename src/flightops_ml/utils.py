"""Small helpers: hashing, git/environment provenance, atomic JSON writes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

TRACKED_PACKAGES = (
    "pandas",
    "numpy",
    "scikit-learn",
    "xgboost",
    "optuna",
    "shap",
    "mlflow",
    "pandera",
    "fastapi",
    "streamlit",
    "plotly",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit(cwd: Path | None = None) -> str | None:
    env_sha = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_COMMIT")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def python_version() -> str:
    return platform.python_version()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON so concurrent readers never observe a partial file (process-safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")
