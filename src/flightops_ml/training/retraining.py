"""Controlled retraining: validate -> version -> train candidate -> compare -> register -> gates.

The workflow STOPS before promotion. A retrained model is never deployed automatically; a human must
approve it (see flightops_ml.registry.promotion).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from flightops_ml.config import get_paths
from flightops_ml.data.ingestion import fingerprint_dataset, load_raw_csv
from flightops_ml.data.schema import TARGET
from flightops_ml.data.validation import DataValidationError, validate_and_report
from flightops_ml.logging_utils import get_logger
from flightops_ml.registry import audit
from flightops_ml.training.train import PipelineResult, run_training_pipeline

log = get_logger("flightops.retrain")
TRIGGERS = ("manual", "scheduled", "new_labeled_data", "monitoring_alert")


class RetrainingError(RuntimeError):
    pass


def run_retraining(
    new_data_path: Path,
    trigger: str = "manual",
    overrides: dict[str, Any] | None = None,
    reporter=None,
    reason: str = "",
) -> dict[str, Any]:
    if trigger not in TRIGGERS:
        raise RetrainingError(f"Unknown trigger '{trigger}'. Allowed: {TRIGGERS}")
    paths = get_paths()
    paths.ensure_dirs()
    audit.append_event("retraining_started", trigger=trigger, data=str(new_data_path), reason=reason)

    # 1-2. validate the new data and verify that ground-truth labels are present
    df = load_raw_csv(Path(new_data_path))
    if TARGET not in df.columns or df[TARGET].isna().any():
        raise RetrainingError(f"New data must contain a fully populated '{TARGET}' label column")
    if df[TARGET].nunique() < 2:
        raise RetrainingError("New data has a single class; cannot train a classifier")
    try:
        validate_and_report(df, paths.evaluations / "data_quality_report_retraining.json")
    except DataValidationError as exc:
        audit.append_event("retraining_rejected", trigger=trigger, reason=f"data validation: {exc}")
        raise

    # 3. version the dataset (content-addressed copy; also `dvc add` it for remote storage)
    fp = fingerprint_dataset(Path(new_data_path), df)
    versions_dir = paths.data_processed / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    versioned = versions_dir / f"ai4i_{fp.sha256[:12]}.csv"
    if not versioned.exists():
        shutil.copy2(new_data_path, versioned)

    # 4-8. train, evaluate, compare with champion, register candidate, apply gates
    result: PipelineResult = run_training_pipeline(
        overrides, dataset_path=versioned, register=True, reporter=reporter, trigger=f"retrain:{trigger}",
        quality_report_name="data_quality_report_retraining.json",
    )
    passed = bool(result.gates) and all(g["passed"] for g in result.gates)
    outcome = {
        "trigger": trigger,
        "dataset_version": result.fingerprint.version,
        "candidate_version": result.registered_version,
        "selected_model": result.champion_name,
        "gates_passed": passed,
        "gates": result.gates,
        # 9-10. approval + deployment are intentionally manual
        "next_step": "Review the candidate and approve it (admin) to enable deployment."
        if passed else "Candidate failed the validation gates and stays EXPERIMENTAL.",
    }
    audit.append_event("retraining_finished", **{k: v for k, v in outcome.items() if k != "gates"})
    log.info("retraining finished", extra={"candidate": result.registered_version, "gates_passed": passed})
    return outcome
