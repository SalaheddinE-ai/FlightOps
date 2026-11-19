"""Background training worker. Usage: python -m workers.training_worker --job-id <id>

Runs a real training job and persists its state (stage, iteration, metrics, run id) to a JSON file that
the Streamlit dashboard polls, so the UI never blocks.
"""

from __future__ import annotations

import argparse
import traceback

from flightops_ml.training.jobs import JobProgress, read_job, update_job
from flightops_ml.utils import utc_now_iso


def run_job(job_id: str) -> int:
    job = read_job(job_id)
    if job is None:
        print(f"Unknown job {job_id}")
        return 2
    req = job.get("request", {})
    update_job(job_id, status="running", stage="starting", started_at=utc_now_iso())
    progress = JobProgress(job_id)
    try:
        overrides: dict = {}
        if req.get("n_trials") is not None:
            overrides["tuning"] = {"n_trials": int(req["n_trials"]), "enabled": bool(req.get("tune", True))}
        elif "tune" in req:
            overrides["tuning"] = {"enabled": bool(req["tune"])}
        if req.get("models"):
            overrides["models"] = {"enabled": list(req["models"])}
        if req.get("xgboost_params"):
            overrides.setdefault("models", {})["xgboost"] = dict(req["xgboost_params"])
        if req.get("threshold_strategy"):
            overrides["threshold"] = {"strategy": req["threshold_strategy"]}
        if req.get("kind") == "retrain":
            from flightops_ml.training.retraining import run_retraining

            outcome = run_retraining(req["data_path"], trigger=req.get("trigger", "manual"),
                                     reporter=progress, reason=req.get("reason", ""))
            update_job(job_id, status="completed", stage="completed", finished_at=utc_now_iso(), result=outcome)
            return 0
        from flightops_ml.training.train import run_training_pipeline

        result = run_training_pipeline(overrides, reporter=progress, trigger=f"job:{job_id}",
                                       dataset_path=None)
        update_job(job_id, status="completed", stage="completed", finished_at=utc_now_iso(),
                   result={"selected_model": result.champion_name, "registered_version": result.registered_version,
                           "parent_run_id": result.parent_run_id,
                           "gates_passed": bool(result.gates) and all(g["passed"] for g in result.gates)})
        return 0
    except Exception as exc:  # noqa: BLE001 - last-resort handler: record the failure in the job file
        update_job(job_id, status="failed", stage="failed", finished_at=utc_now_iso(),
                   error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
        return 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    raise SystemExit(run_job(ap.parse_args().job_id))
