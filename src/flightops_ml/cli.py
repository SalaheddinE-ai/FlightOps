"""Command line interface: ``flightops <command>`` or ``python -m flightops_ml <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from flightops_ml.config import get_paths, load_monitoring_config, registered_model_name


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_demo_data(a) -> int:
    from flightops_ml.data.synthetic import write_demo_csv

    out = write_demo_csv(Path(a.out), n=a.n, seed=a.seed)
    print(f"Wrote SYNTHETIC demo dataset (NOT the real AI4I 2020 data) to {out}")
    print(f'Use it with:  $env:FLIGHTOPS_DATASET = "{out}"')
    return 0


def cmd_validate(a) -> int:
    from flightops_ml.data.ingestion import fingerprint_dataset, load_raw_csv
    from flightops_ml.data.validation import validate_and_report

    paths = get_paths()
    path = Path(a.data) if a.data else paths.dataset_path()
    df = load_raw_csv(path)
    rep = validate_and_report(df, paths.evaluations / "data_quality_report.json")
    _print({"fingerprint": fingerprint_dataset(path, df).as_dict(), "passed": rep.passed,
            "warnings": rep.warnings, "target": rep.target_distribution})
    return 0


def cmd_train(a) -> int:
    from flightops_ml.reporting.cml_report import build_cml_report
    from flightops_ml.training.train import run_training_pipeline

    overrides: dict = {"tuning": {"enabled": not a.no_tune}}
    if a.trials:
        overrides["tuning"]["n_trials"] = a.trials
    if a.models:
        overrides["models"] = {"enabled": a.models}
    res = run_training_pipeline(overrides, dataset_path=Path(a.data) if a.data else None,
                                register=not a.no_register, trigger=a.trigger)
    report = build_cml_report(res.summary_path)
    _print({"selected_model": res.champion_name, "registered_version": res.registered_version,
            "gates": res.gates, "summary": str(res.summary_path), "cml_report": str(report)})
    return 0


def _client():
    from flightops_ml.tracking.mlflow_tracking import get_client

    return get_client()


def cmd_registry(a) -> int:
    from flightops_ml.registry import promotion, registration

    name, client = registered_model_name(), _client()
    if a.action == "status":
        print(registration.versions_table(client, name).to_string(index=False))
    elif a.action == "approve":
        promotion.approve(client, name, str(a.version), a.user, a.reason)
    elif a.action == "deploy":
        promotion.deploy(client, name, str(a.version), a.user)
    elif a.action == "rollback":
        print("restored version", promotion.rollback(client, name, a.user, a.reason))
    elif a.action == "archive":
        promotion.archive(client, name, str(a.version), a.reason)
    return 0


def cmd_retrain(a) -> int:
    from flightops_ml.training.retraining import run_retraining

    _print(run_retraining(Path(a.data), trigger=a.trigger, reason=a.reason))
    return 0


def cmd_monitor(a) -> int:
    from flightops_ml.data.schema import RAW_FEATURES
    from flightops_ml.monitoring.drift import (
        classify_drift,
        dataset_drift_summary,
        evidently_report,
        feature_drift,
        simulate_shift,
    )
    from flightops_ml.monitoring.monitoring import build_monitoring_report, save_monitoring_report
    from flightops_ml.monitoring.prediction_log import PredictionLog

    paths = get_paths()
    ref = pd.read_csv(paths.data_reference / "reference_features.csv")
    if a.simulate:
        shifted = simulate_shift(ref[RAW_FEATURES], a.simulate)
        fd = feature_drift(ref, shifted)
        summ = dataset_drift_summary(fd)
        report = {"label": f"SYNTHETIC drift scenario '{a.simulate}' - not real equipment degradation",
                  "drift": {**summ, "features": fd.to_dict(orient="records")},
                  "interpretation": classify_drift(summ, None, None)}
        html = evidently_report(ref, shifted, paths.monitoring_reports / f"evidently_{a.simulate}.html")
        report["evidently_html"] = str(html) if html else "evidently not installed - built-in statistics used"
    else:
        log = PredictionLog()
        proba_path = paths.data_reference / "reference_proba.csv"
        meta = json.loads((paths.data_reference / "reference_meta.json").read_text())
        report = build_monitoring_report(
            ref, log.predictions(), log.errors(), log.labels(),
            pd.read_csv(proba_path)["probability"].to_numpy() if proba_path.exists() else None,
            baseline_recall=meta.get("baseline_test_recall"), cfg=load_monitoring_config())
    out = save_monitoring_report(report)
    _print({"saved": str(out), "status": report.get("status"), "interpretation": report.get("interpretation")})
    return 0


def cmd_cml(a) -> int:
    from flightops_ml.reporting.cml_report import build_cml_report

    print(build_cml_report())
    return 0


def cmd_leakage_check(a) -> int:
    """CI gate: fail when excluded columns reach the model inputs or splits overlap."""
    from flightops_ml.data.preprocessing import assert_no_leakage, split_dataset
    from flightops_ml.data.schema import EXCLUDED_FROM_MODEL, RAW_FEATURES
    from flightops_ml.data.synthetic import generate_ai4i_like

    bad = [c for c in RAW_FEATURES if c in EXCLUDED_FROM_MODEL]
    if bad:
        print(f"LEAKAGE: forbidden columns in RAW_FEATURES: {bad}")
        return 1
    df = generate_ai4i_like(2000, 1)
    sp = split_dataset(df[RAW_FEATURES], df["machine_failure"], 42)
    assert_no_leakage(sp)
    print("leakage check passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="flightops", description="FlightOps ML command line")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("demo-data", help="write a SYNTHETIC AI4I-like CSV for demos/CI")
    s.add_argument("--out", default=str(get_paths().data_raw / "ai4i2020_synthetic_demo.csv"))
    s.add_argument("--n", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_demo_data)
    s = sub.add_parser("validate-data")
    s.add_argument("--data")
    s.set_defaults(fn=cmd_validate)
    s = sub.add_parser("train")
    s.add_argument("--data")
    s.add_argument("--no-tune", action="store_true")
    s.add_argument("--no-register", action="store_true")
    s.add_argument("--trials", type=int)
    s.add_argument("--models", nargs="+", choices=["logistic_regression", "random_forest", "xgboost"])
    s.add_argument("--trigger", default="manual")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("registry", help="status | approve | deploy | rollback | archive (admin only)")
    s.add_argument("action", choices=["status", "approve", "deploy", "rollback", "archive"])
    s.add_argument("--version")
    s.add_argument("--user", default="cli-admin")
    s.add_argument("--reason", default="")
    s.set_defaults(fn=cmd_registry)
    s = sub.add_parser("retrain")
    s.add_argument("--data", required=True)
    s.add_argument("--trigger", default="manual")
    s.add_argument("--reason", default="")
    s.set_defaults(fn=cmd_retrain)
    s = sub.add_parser("monitor")
    s.add_argument("--simulate", choices=["temperature_shift", "torque_increase", "tool_wear_shift"])
    s.set_defaults(fn=cmd_monitor)
    sub.add_parser("cml-report").set_defaults(fn=cmd_cml)
    sub.add_parser("leakage-check").set_defaults(fn=cmd_leakage_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
