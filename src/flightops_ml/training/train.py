"""End-to-end training pipeline: ingest -> validate -> split -> train -> evaluate -> track -> register."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from flightops_ml.config import (
    get_paths,
    load_model_config,
    load_training_config,
    registered_model_name,
    tracking_uri,
)
from flightops_ml.data.ingestion import (
    DatasetFingerprint,
    fingerprint_dataset,
    load_raw_csv,
    select_model_frame,
)
from flightops_ml.data.preprocessing import (
    DataSplits,
    assert_no_leakage,
    build_preprocessor,
    split_dataset,
)
from flightops_ml.data.validation import validate_and_report
from flightops_ml.evaluation.explainability import (
    feature_importance,
    save_shap_plots,
    supports_shap,
)
from flightops_ml.evaluation.metrics import (
    calibration_data,
    classification_metrics,
    save_evaluation_plots,
)
from flightops_ml.evaluation.threshold import select_thresholds
from flightops_ml.features.engineering import FEATURE_ENGINEERING_VERSION
from flightops_ml.logging_utils import get_logger
from flightops_ml.registry import promotion, registration
from flightops_ml.tracking.mlflow_tracking import (
    ensure_experiment,
    get_client,
    log_dict_artifact,
    log_metric_history,
)
from flightops_ml.training.callbacks import ProgressCallback
from flightops_ml.training.jobs import NullProgress
from flightops_ml.training.models import (
    build_sklearn_model,
    build_xgboost,
    compute_scale_pos_weight,
)
from flightops_ml.training.tuning import evaluate_class_weighting, tune_xgboost
from flightops_ml.utils import (
    dependency_versions,
    git_commit,
    python_version,
    utc_now_iso,
    write_json_atomic,
)

log = get_logger("flightops.train")


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    run_id: str
    model_uri: str
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    thresholds: dict[str, dict]
    threshold: float
    strategy: str
    params: dict[str, Any]
    history: dict[str, Any] | None
    duration_s: float
    latency_ms: dict[str, float]


@dataclass
class PipelineResult:
    parent_run_id: str
    experiment_id: str
    fingerprint: DatasetFingerprint
    results: list[ModelResult]
    champion_name: str
    registered_version: str | None
    gates: list[dict[str, Any]] = field(default_factory=list)
    summary_path: Path | None = None
    splits: DataSplits | None = None

    @property
    def best(self) -> ModelResult:
        return next(r for r in self.results if r.name == self.champion_name)


def measure_latency(pipeline: Pipeline, X: pd.DataFrame, n: int = 50) -> dict[str, float]:  # noqa: N803
    sample = X.head(n)
    times = []
    for i in range(len(sample)):
        t0 = time.perf_counter()
        pipeline.predict_proba(sample.iloc[[i]])
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95)),
            "mean_ms": float(arr.mean())}


def fit_model(
    name: str,
    cfg: dict[str, Any],
    splits: DataSplits,
    seed: int,
    reporter=None,
    xgb_override: dict[str, Any] | None = None,
) -> tuple[Pipeline, dict[str, Any] | None, dict[str, Any]]:
    """Fit preprocessing on TRAIN only, then the estimator. Returns (pipeline, history, params)."""
    reporter = reporter or NullProgress()
    engineered = cfg["features"]["engineered"]
    pre = build_preprocessor(engineered)
    Xtr = pre.fit_transform(splits.X_train)  # noqa: N806 - fitted on training data only
    history = None
    if name == "xgboost":
        xp = {**cfg["models"]["xgboost"], **(xgb_override or {})}
        spw_cfg = xp.get("scale_pos_weight", "auto")
        spw = compute_scale_pos_weight(splits.y_train) if spw_cfg == "auto" else float(spw_cfg)
        rounds = int(xp.get("early_stopping_rounds", 40))
        clf = build_xgboost(
            xp, seed, spw,
            eval_metric=["logloss", "aucpr"],
            early_stopping_rounds=rounds,
            callbacks=[ProgressCallback(reporter, int(xp["n_estimators"]))],
        )
        Xv = pre.transform(splits.X_val)  # noqa: N806
        clf.fit(Xtr, splits.y_train, eval_set=[(Xtr, splits.y_train), (Xv, splits.y_val)], verbose=False)
        raw = clf.evals_result()
        history = {
            "train": {m: list(map(float, v)) for m, v in raw["validation_0"].items()},
            "validation": {m: list(map(float, v)) for m, v in raw["validation_1"].items()},
            "best_iteration": int(clf.best_iteration) + 1,
            "n_iterations_run": len(raw["validation_0"]["logloss"]),
            "early_stopped": len(raw["validation_0"]["logloss"]) < int(xp["n_estimators"]),
            "early_stopping_metric": "validation aucpr",
        }
        params = {**{k: xp[k] for k in ("n_estimators", "max_depth", "learning_rate", "subsample",
                                       "colsample_bytree", "min_child_weight", "gamma", "reg_alpha",
                                       "reg_lambda")}, "scale_pos_weight": spw,
                  "early_stopping_rounds": rounds}
    else:
        clf = build_sklearn_model(name, cfg["models"][name], seed)
        clf.fit(Xtr, splits.y_train)
        params = {k: v for k, v in cfg["models"][name].items()}
    return Pipeline([("pre", pre), ("clf", clf)]), history, params


def _proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
    return pipeline.predict_proba(X)[:, 1]


def _prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in metrics.items() if not np.isnan(v)}


def _train_and_log_one(
    name: str, cfg: dict[str, Any], splits: DataSplits, fp: DatasetFingerprint,
    quality_ok: bool, tune: bool, reporter, parent_tags: dict[str, str], out_dir: Path,
) -> ModelResult:
    seed = cfg["seed"]
    thr_cfg = cfg["threshold"]
    with mlflow.start_run(run_name=name, nested=True) as run:
        mlflow.set_tags(parent_tags | {"flightops.algorithm": name, "flightops.kind": "model_training"})
        reporter.run(run.info.run_id, model=name, experiment=run.info.experiment_id)
        reporter.stage(f"training {name}")
        t0 = time.perf_counter()

        xgb_override = None
        extra_artifacts: dict[str, Any] = {}
        if name == "xgboost":
            tuning = cfg["tuning"]
            cw = evaluate_class_weighting(
                splits.X_train, splits.y_train, cfg["features"]["engineered"],
                cfg["models"]["xgboost"], seed, min(tuning["cv_folds"], 3), tuning["metric"],
            )
            extra_artifacts["class_weighting_comparison"] = cw
            mlflow.log_metrics({k: v for k, v in cw.items()})
            if tune and tuning["enabled"]:
                reporter.stage("hyper-parameter optimisation (Optuna)")
                best, trials, _ = tune_xgboost(
                    splits.X_train, splits.y_train, cfg["features"]["engineered"], seed,
                    int(tuning["n_trials"]), int(tuning["cv_folds"]), tuning["metric"],
                    tuning.get("timeout_seconds"), bool(tuning.get("log_trials_to_mlflow", True)),
                )
                xgb_override = best
                trials_path = out_dir / "optuna_trials.csv"
                trials.to_csv(trials_path, index=False)
                mlflow.log_artifact(str(trials_path), "optimization")
                mlflow.log_param("tuning_best_cv_" + tuning["metric"], float(trials["value"].max()))
            reporter.stage("fitting xgboost")

        pipeline, history, params = fit_model(name, cfg, splits, seed, reporter, xgb_override)
        duration = time.perf_counter() - t0

        proba_val = _proba(pipeline, splits.X_val)
        choices = select_thresholds(
            splits.y_val, proba_val, thr_cfg["recall_target"],
            thr_cfg["cost_false_negative"], thr_cfg["cost_false_positive"],
        )
        strategy = thr_cfg["strategy"]
        threshold = choices[strategy].threshold
        val_metrics = classification_metrics(splits.y_val, proba_val, threshold)
        # FINAL evaluation: the test set is scored once per model, at a threshold fixed on validation.
        proba_test = _proba(pipeline, splits.X_test)
        test_metrics = classification_metrics(splits.y_test, proba_test, threshold)
        test_at_all = {k: classification_metrics(splits.y_test, proba_test, c.threshold)
                       for k, c in choices.items()}
        latency = measure_latency(pipeline, splits.X_test, cfg["latency"]["n_samples"])

        # ---- parameters
        mlflow.log_params({
            "algorithm": name,
            "random_seed": seed,
            "feature_engineering": ",".join(cfg["features"]["engineered"]),
            "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
            "validation_strategy": "stratified hold-out 70/10/20 (CV on train only for tuning)",
            "decision_threshold": round(threshold, 4),
            "threshold_strategy": strategy,
            "class_weighting": str(params.get("scale_pos_weight", params.get("class_weight"))),
            "tuned": bool(xgb_override is not None),
            **{k: (v if v is not None else "None") for k, v in params.items()},
        })
        # ---- metrics
        mlflow.log_metrics(_prefixed(val_metrics, "val") | _prefixed(test_metrics, "test") | {
            "training_duration_s": duration,
            "inference_latency_ms_p50": latency["p50_ms"],
            "inference_latency_ms_p95": latency["p95_ms"],
        })
        if history is not None:
            log_metric_history(history, get_client(), run.info.run_id)
            mlflow.log_metrics({"best_iteration": history["best_iteration"],
                                "n_iterations_run": history["n_iterations_run"]})
            log_dict_artifact(history, "training/training_history.json")

        # ---- artifacts
        fig_dir = out_dir / name
        plots = save_evaluation_plots(splits.y_test, proba_test, threshold, fig_dir)
        for p in plots.values():
            mlflow.log_artifact(str(p), "figures")
        if supports_shap(pipeline):
            shap_paths = save_shap_plots(pipeline, splits.X_val.head(500), fig_dir)
            for p in shap_paths.values():
                mlflow.log_artifact(str(p), "explainability")
        imp = feature_importance(pipeline)
        imp_path = fig_dir / "feature_importance.csv"
        imp.to_csv(imp_path, index=False)
        mlflow.log_artifact(str(imp_path), "explainability")

        log_dict_artifact({k: v for k, v in cfg.items()}, "config/training_config.json")
        log_dict_artifact({"engineered_features": cfg["features"]["engineered"],
                           "version": FEATURE_ENGINEERING_VERSION,
                           "scaler": "StandardScaler (fit on train)", "encoder": "OneHotEncoder(type)"},
                          "config/preprocessing_config.json")
        log_dict_artifact({"python": python_version(), "dependencies": dependency_versions()},
                          "config/environment.json")
        log_dict_artifact(fp.as_dict(), "config/dataset_fingerprint.json")
        log_dict_artifact({
            "model": name,
            "selected_threshold": {"strategy": strategy, "value": threshold},
            "thresholds_selected_on_validation": {k: v.as_dict() for k, v in choices.items()},
            "validation_metrics": val_metrics,
            "test_metrics_final_evaluation": test_metrics,
            "test_metrics_at_each_threshold_informational": test_at_all,
            "calibration_test": calibration_data(splits.y_test, proba_test),
            "class_weighting": extra_artifacts.get("class_weighting_comparison"),
        }, "evaluation/evaluation_report.json")

        # ---- model with signature + input example
        sig = infer_signature(splits.X_train.head(100), pipeline.predict_proba(splits.X_train.head(100))[:, 1])
        info = mlflow.sklearn.log_model(
            pipeline, name="model", signature=sig, input_example=splits.X_train.head(3),
            # The pipeline embeds our custom FeatureEngineer, which skops (MLflow's default) does not
            # trust. cloudpickle is acceptable here because models only come from our own registry.
            serialization_format="cloudpickle",
        )
        return ModelResult(
            name=name, pipeline=pipeline, run_id=run.info.run_id, model_uri=info.model_uri,
            val_metrics=val_metrics, test_metrics=test_metrics,
            thresholds={k: v.as_dict() for k, v in choices.items()}, threshold=threshold,
            strategy=strategy, params=params, history=history, duration_s=duration, latency_ms=latency,
        )


def select_best(results: list[ModelResult], primary: str, tie_breaker: str) -> ModelResult:
    """Choose by VALIDATION metrics only (never accuracy by default; never the test set)."""
    p, t = primary.removeprefix("val_"), tie_breaker.removeprefix("val_")
    return max(results, key=lambda r: (round(r.val_metrics[p], 6), r.val_metrics[t]))


def run_training_pipeline(
    overrides: dict[str, Any] | None = None,
    *,
    dataset_path: Path | None = None,
    register: bool = True,
    reporter=None,
    trigger: str = "manual",
    quality_report_name: str = "data_quality_report.json",
) -> PipelineResult:
    cfg = load_training_config(overrides)
    reporter = reporter or NullProgress()
    paths = get_paths()
    paths.ensure_dirs()
    seed = int(cfg["seed"])
    name = registered_model_name()

    reporter.stage("ingestion")
    ds_path = Path(dataset_path) if dataset_path else paths.dataset_path()
    df = load_raw_csv(ds_path)
    fp = fingerprint_dataset(ds_path, df)

    reporter.stage("validation")
    report = validate_and_report(df, paths.evaluations / quality_report_name)

    reporter.stage("splitting")
    X, y = select_model_frame(df)  # noqa: N806
    split_cfg = cfg["split"]
    splits = split_dataset(X, y, seed, split_cfg["train"], split_cfg["validation"], split_cfg["test"])
    assert_no_leakage(splits)

    exp_id = ensure_experiment()
    sha = git_commit(paths.root)
    out_dir = paths.figures / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[ModelResult] = []
    gates_out: list[dict[str, Any]] = []
    version: str | None = None

    with mlflow.start_run(run_name=f"pipeline-{time.strftime('%Y%m%d-%H%M%S')}") as parent:
        parent_tags = {
            "flightops.parent_run_id": parent.info.run_id,
            "flightops.dataset_version": fp.version,
            "flightops.dataset_sha256": fp.sha256,
            "flightops.git_commit": sha or "unknown",
            "flightops.trigger": trigger,
            "flightops.python": python_version(),
        }
        mlflow.set_tags(parent_tags | {"flightops.kind": "pipeline"})
        mlflow.log_params({"dataset_path": fp.path, "dataset_rows": fp.n_rows,
                           "dataset_size_bytes": fp.size_bytes, "seed": seed,
                           **{f"split_{k}": v for k, v in splits.sizes().items()}})
        log_dict_artifact(report.as_dict(), "data/data_quality_report.json")
        log_dict_artifact(fp.as_dict(), "data/dataset_fingerprint.json")

        for model_name in cfg["models"]["enabled"]:
            results.append(_train_and_log_one(
                model_name, cfg, splits, fp, report.passed, True, reporter, parent_tags, out_dir))

        sel = cfg["selection"]
        best = select_best(results, sel["primary_metric"], sel["tie_breaker"])
        comparison = pd.DataFrame([{
            "model": r.name, "run_id": r.run_id, "threshold": r.threshold,
            **{f"val_{k}": r.val_metrics[k] for k in ("pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy")},
            **{f"test_{k}": r.test_metrics[k] for k in ("pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy")},
            "train_s": r.duration_s, "latency_p95_ms": r.latency_ms["p95_ms"],
            "selected": r.name == best.name,
        } for r in results])
        cmp_path = out_dir / "model_comparison.csv"
        comparison.to_csv(cmp_path, index=False)
        mlflow.log_artifact(str(cmp_path), "comparison")
        mlflow.set_tag("flightops.selected_model", best.name)
        mlflow.log_param("selection_rule", f"max {sel['primary_metric']} (tie: {sel['tie_breaker']}) on validation")

        client = get_client()
        if register:
            reporter.stage("registering candidate")
            mv = registration.register_candidate(
                client, name, best.model_uri,
                tags={
                    "run_id": best.run_id, "parent_run_id": parent.info.run_id,
                    "experiment_id": exp_id, "algorithm": best.name,
                    "threshold": best.threshold, "threshold_strategy": best.strategy,
                    "dataset_version": fp.version, "dataset_sha256": fp.sha256,
                    "git_commit": sha or "unknown", "trigger": trigger,
                    "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
                    "data_validation_passed": report.passed,
                    **{f"val_{k}": f"{best.val_metrics[k]:.6f}" for k in ("pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy")},
                    **{f"test_{k}": f"{best.test_metrics[k]:.6f}" for k in ("pr_auc", "roc_auc", "f1", "recall", "precision", "accuracy")},
                },
                description=f"{best.name} trained on {fp.version}; selected on validation {sel['primary_metric']}.",
            )
            version = str(mv.version)
            reporter.stage("validation gates")
            gates = promotion.validate_candidate(
                client, name, version, best.test_metrics, splits.X_test, splits.y_test,
                data_validation_ok=report.passed, model_cfg=load_model_config(),
            )
            gates_out = [g.as_dict() for g in gates]
            mlflow.set_tag("flightops.registered_version", version)

        ref_dir = paths.data_reference
        ref_dir.mkdir(parents=True, exist_ok=True)
        splits.X_train.to_csv(ref_dir / "reference_features.csv", index=False)
        pd.DataFrame({"probability": _proba(best.pipeline, splits.X_val)}).to_csv(
            ref_dir / "reference_proba.csv", index=False)
        write_json_atomic(ref_dir / "reference_meta.json", {
            "dataset_version": fp.version, "model": best.name, "registered_version": version,
            "baseline_test_recall": best.test_metrics["recall"], "created_at": utc_now_iso()})

        summary = {
            "generated_at": utc_now_iso(), "model_name": name, "registered_version": version,
            "selected_model": best.name, "dataset_version": fp.version, "git_commit": sha,
            "parent_run_id": parent.info.run_id, "experiment_id": exp_id, "tracking_uri": tracking_uri(),
            "threshold": best.threshold, "threshold_strategy": best.strategy,
            "validation_metrics": best.val_metrics, "test_metrics": best.test_metrics,
            "comparison": comparison.to_dict(orient="records"), "gates": gates_out,
            "figures_dir": str(out_dir), "trigger": trigger,
        }
        summary_path = paths.evaluations / "latest_summary.json"
        write_json_atomic(summary_path, summary)
        mlflow.log_artifact(str(summary_path), "comparison")
        reporter.stage("completed")
        log.info("training pipeline finished", extra={"selected": best.name, "version": version})
        return PipelineResult(parent.info.run_id, exp_id, fp, results, best.name, version,
                              gates_out, summary_path, splits)
