"""Process-safe training-job state shared between the worker process and the dashboard.

State lives in one JSON file per job, written atomically (tmp file + os.replace), so a polling UI
never reads a half-written file.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from flightops_ml.config import get_paths
from flightops_ml.utils import read_json, utc_now_iso, write_json_atomic

TERMINAL = {"completed", "failed"}


def _job_path(job_id: str) -> Path:
    return get_paths().jobs / f"{job_id}.json"


def create_job(request: dict[str, Any]) -> str:
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    state = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "iteration": None,
        "latest_metrics": {},
        "model": None,
        "experiment": None,
        "run_id": None,
        "pid": None,
        "request": request,
        "created_at": utc_now_iso(),
        "started_at": None,
        "finished_at": None,
        "elapsed_seconds": 0.0,
        "error": None,
        "result": None,
    }
    write_json_atomic(_job_path(job_id), state)
    return job_id


def read_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        return read_json(path)
    except json.JSONDecodeError:
        return None


def update_job(job_id: str, **fields: Any) -> dict[str, Any]:
    state = read_job(job_id) or {"job_id": job_id}
    state.update(fields)
    if state.get("started_at") and state.get("status") not in TERMINAL:
        from datetime import datetime

        started = datetime.fromisoformat(state["started_at"])
        state["elapsed_seconds"] = round((datetime.now(started.tzinfo) - started).total_seconds(), 1)
    write_json_atomic(_job_path(job_id), state)
    return state


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    jobs_dir = get_paths().jobs
    if not jobs_dir.exists():
        return []
    states = []
    for p in sorted(jobs_dir.glob("*.json"), reverse=True)[:limit]:
        try:
            states.append(read_json(p))
        except json.JSONDecodeError:
            continue
    return states


def start_job_process(request: dict[str, Any]) -> str:
    """Create a job and launch the worker as a detached subprocess (non-blocking)."""
    job_id = create_job(request)
    cmd = [sys.executable, "-m", "workers.training_worker", "--job-id", job_id]
    kwargs: dict[str, Any] = {"cwd": str(get_paths().root)}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)  # noqa: S603
    update_job(job_id, pid=proc.pid)
    return job_id


class JobProgress:
    """ProgressReporter implementation persisting to the job file (throttled)."""

    def __init__(self, job_id: str, min_interval: float = 0.4):
        self.job_id = job_id
        self.min_interval = min_interval
        self._last = 0.0

    def stage(self, name: str, **extra: Any) -> None:
        update_job(self.job_id, stage=name, **extra)

    def run(self, run_id: str, model: str | None = None, experiment: str | None = None) -> None:
        update_job(self.job_id, run_id=run_id, model=model, experiment=experiment)

    def iteration(self, iteration: int, metrics: dict[str, float], force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < self.min_interval:
            return
        self._last = now
        update_job(self.job_id, iteration=iteration, latest_metrics=metrics)


class NullProgress:
    def stage(self, name: str, **extra: Any) -> None:
        return None

    def run(self, run_id: str, model: str | None = None, experiment: str | None = None) -> None:
        return None

    def iteration(self, iteration: int, metrics: dict[str, float], force: bool = False) -> None:
        return None
