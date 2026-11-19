"""SHAP explanations and feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def _parts(pipeline: Pipeline):
    return pipeline.named_steps["pre"], pipeline.named_steps["clf"]


def transformed_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
    pre, _ = _parts(pipeline)
    out = pre.transform(X)
    return out if isinstance(out, pd.DataFrame) else pd.DataFrame(out)


def supports_shap(pipeline: Pipeline) -> bool:
    _, clf = _parts(pipeline)
    return type(clf).__name__ in {"XGBClassifier", "RandomForestClassifier"}


def shap_values(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:  # noqa: N803
    """Return (values[n, f], base_values[n], feature_names) in log-odds space for XGBoost."""
    if not supports_shap(pipeline):
        raise ValueError("SHAP is supported for tree models (XGBoost, Random Forest) only")
    _, clf = _parts(pipeline)
    Xt = transformed_features(pipeline, X)  # noqa: N806
    explainer = shap.TreeExplainer(clf)
    values = explainer.shap_values(Xt)
    if isinstance(values, list):  # older shap: list per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # (n, f, classes)
        values = values[:, :, 1]
    expected = np.asarray(explainer.expected_value)
    base_value = float(expected.ravel()[-1])
    return values, np.full(len(Xt), base_value), list(Xt.columns)


def explain_row(pipeline: Pipeline, row: pd.DataFrame, top_k: int = 8) -> dict:
    values, base, names = shap_values(pipeline, row.iloc[[0]])
    contrib = sorted(zip(names, values[0].tolist(), strict=True), key=lambda kv: -abs(kv[1]))
    return {
        "base_value": float(base[0]),
        "unit": "log-odds" if type(pipeline.named_steps["clf"]).__name__ == "XGBClassifier" else "probability",
        "contributions": [{"feature": n, "shap_value": v} for n, v in contrib[:top_k]],
    }


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    pre, clf = _parts(pipeline)
    names = list(pre.get_feature_names_out())
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_)
    elif hasattr(clf, "coef_"):
        imp = np.abs(np.asarray(clf.coef_)).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": names, "importance": imp}).sort_values("importance", ascending=False)


def save_shap_plots(
    pipeline: Pipeline, X_sample: pd.DataFrame, out_dir: Path, prefix: str = ""  # noqa: N803
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if not supports_shap(pipeline):
        return paths
    values, _, names = shap_values(pipeline, X_sample)
    Xt = transformed_features(pipeline, X_sample)  # noqa: N806
    fig = plt.figure(figsize=(7, 4.5))
    shap.summary_plot(values, Xt, feature_names=names, show=False)
    paths["shap_summary"] = out_dir / f"{prefix}shap_summary.png"
    plt.tight_layout()
    fig.savefig(paths["shap_summary"], dpi=120, bbox_inches="tight")
    plt.close(fig)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=names).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.8))
    mean_abs.plot.barh(ax=ax)
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title("Global SHAP importance")
    paths["shap_importance"] = out_dir / f"{prefix}shap_importance.png"
    fig.tight_layout()
    fig.savefig(paths["shap_importance"], dpi=120)
    plt.close(fig)
    return paths
