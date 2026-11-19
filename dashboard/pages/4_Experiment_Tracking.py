"""Page 4 - Experiment Tracking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.ui import setup_page, style_fig  # noqa: E402
from dashboard.services import mlflow_service  # noqa: E402
from dashboard.services.common import client, mlflow_ui_url  # noqa: E402

setup_page("Experiment Tracking", "📈")
st.link_button("Open MLflow UI", mlflow_ui_url())
kind = st.radio("Run type", ["model_training", "optuna_trial", "pipeline"], horizontal=True)
runs = mlflow_service.runs_frame(kind)
if runs.empty:
    st.info("No runs recorded yet. Launch a training job first.")
    st.stop()

base_cols = [c for c in ["name", "algorithm", "start", "status", "dataset_version", "git_commit", "run_id"] if c in runs]
metric_cols = [c for c in runs if c.startswith("metric.")]
st.dataframe(runs[base_cols + [c for c in metric_cols if c in (
    "metric.val_pr_auc", "metric.test_f1", "metric.test_recall", "metric.test_pr_auc", "metric.cv_average_precision")]],
    width="stretch", hide_index=True)

st.subheader("Run detail")
rid = st.selectbox("Run", runs["run_id"], format_func=lambda r: f"{runs.set_index('run_id').loc[r, 'name']} ({r[:8]})")
run = client().get_run(rid)
c1, c2 = st.columns(2)
c1.write("**Parameters**")
c1.dataframe(pd.DataFrame(run.data.params.items(), columns=["param", "value"]), hide_index=True, width="stretch")
c2.write("**Metrics**")
c2.dataframe(pd.DataFrame(run.data.metrics.items(), columns=["metric", "value"]), hide_index=True, width="stretch")
st.write(f"**Artifact location:** `{run.info.artifact_uri}`")
arts = [a.path for a in client().list_artifacts(rid)]
st.write("**Artifacts:** " + (", ".join(f"`{a}`" for a in arts) or "none"))
for fig in ("figures/confusion_matrix.png", "figures/roc_curve.png", "figures/pr_curve.png",
            "explainability/shap_summary.png"):
    p = mlflow_service.download_artifact(rid, fig)
    if p:
        st.image(p, caption=fig, width=380)

st.subheader("Compare runs")
chosen = st.multiselect("Select runs to compare", runs["run_id"], default=list(runs["run_id"][:3]),
                        format_func=lambda r: f"{runs.set_index('run_id').loc[r, 'name']} ({r[:8]})")
if len(chosen) >= 2:
    sub = runs[runs["run_id"].isin(chosen)].set_index("run_id")
    pcols = [c for c in sub if c.startswith("param.")]
    diff = sub[pcols].loc[:, sub[pcols].nunique(dropna=False) > 1]
    st.write("**Hyper-parameter differences**")
    st.dataframe(diff, width="stretch")
    mcols = [c for c in ("metric.val_pr_auc", "metric.test_pr_auc", "metric.test_f1", "metric.test_recall",
                         "metric.test_precision") if c in sub]
    if mcols:
        long = sub[mcols].reset_index().melt("run_id", var_name="metric", value_name="value")
        long["run"] = long["run_id"].str[:8]
        st.plotly_chart(style_fig(px.bar(long, x="metric", y="value", color="run", barmode="group")), width="stretch")
        key = "metric.val_pr_auc" if "metric.val_pr_auc" in sub else mcols[0]
        best = sub[key].astype(float).idxmax()
        st.success(f"Best run by `{key}`: **{sub.loc[best, 'name']}** (`{best[:8]}`), dataset "
                   f"`{sub.loc[best, 'dataset_version']}`")
