"""Page 5 - Model Evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve  # noqa: E402

from dashboard.components.ui import banner, setup_page, style_fig, synthetic_notice  # noqa: E402
from dashboard.services import mlflow_service  # noqa: E402
from dashboard.services.common import model_name  # noqa: E402
from dashboard.services.data_service import dataset_missing_message, get_dataset  # noqa: E402
from flightops_ml.config import load_training_config  # noqa: E402
from flightops_ml.data.ingestion import select_model_frame  # noqa: E402
from flightops_ml.data.preprocessing import split_dataset  # noqa: E402
from flightops_ml.evaluation.explainability import shap_values, supports_shap  # noqa: E402
from flightops_ml.evaluation.threshold import threshold_sweep  # noqa: E402
from flightops_ml.registry.registration import TAG  # noqa: E402

setup_page("Model Evaluation", "🎯")
synthetic_notice()
vt = mlflow_service.versions()
if vt.empty:
    st.info("No registered model versions yet.")
    st.stop()
version = st.selectbox("Registered model version", vt["version"],
                       format_func=lambda v: f"v{v} - {vt.set_index('version').loc[v, 'algorithm']} - "
                                             f"{vt.set_index('version').loc[v, 'business_status']}")
mv = mlflow_service.client().get_model_version(model_name(), str(version))
run_id = mv.tags.get(TAG + "run_id")
report = mlflow_service.load_run_json(run_id, "evaluation/evaluation_report.json") if run_id else None
df = get_dataset()
if df is None:
    st.warning(dataset_missing_message())
    st.stop()
cfg = load_training_config()
X, y = select_model_frame(df)
splits = split_dataset(X, y, cfg["seed"], cfg["split"]["train"], cfg["split"]["validation"], cfg["split"]["test"])
model = mlflow.sklearn.load_model(f"models:/{model_name()}/{version}")
thr_saved = float(mv.tags[TAG + "threshold"])
st.caption(f"Splits are re-created with seed {cfg['seed']} from the current dataset. If the dataset differs from the "
           f"one used for training (`{mv.tags.get(TAG + 'dataset_version')}`), recomputed numbers can differ from the recorded ones.")

proba_val = model.predict_proba(splits.X_val)[:, 1]
proba_test = model.predict_proba(splits.X_test)[:, 1]

tab_thr, tab_curves, tab_report, tab_shap = st.tabs(["Threshold tuning", "ROC / PR curves", "Classification report", "SHAP"])
with tab_thr:
    banner("Thresholds are explored on the <b>validation</b> set. The test set is never used to choose a threshold.", info=True)
    cfg_t = cfg["threshold"]
    sweep = threshold_sweep(splits.y_val, proba_val, cost_fn=cfg_t["cost_false_negative"], cost_fp=cfg_t["cost_false_positive"])
    if report:
        ch = report["thresholds_selected_on_validation"]
        st.dataframe(pd.DataFrame(ch).T[["threshold", "precision", "recall", "f1", "constraint_met"]], width="stretch")
    thr = st.slider("Decision threshold", 0.01, 0.99, float(round(thr_saved, 2)), 0.01)
    row = sweep.iloc[(sweep.threshold - thr).abs().argmin()]
    c = st.columns(7)
    for col, (lab, k) in zip(c, [("TP", "tp"), ("FP", "fp"), ("TN", "tn"), ("FN", "fn"), ("Precision", "precision"),
                                 ("Recall", "recall"), ("F1", "f1")], strict=True):
        col.metric(lab, f"{row[k]:.3f}" if isinstance(row[k], float) else int(row[k]))
    fig = go.Figure()
    for k in ("precision", "recall", "f1"):
        fig.add_trace(go.Scatter(x=sweep.threshold, y=sweep[k], name=k))
    fig.add_vline(x=thr, line_dash="dash")
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    st.caption(f"Registered operating threshold: {thr_saved:.2f} ({mv.tags.get(TAG + 'threshold_strategy')}). "
               "Cost columns use the illustrative costs in `configs/training.yaml`.")
with tab_curves:
    c1, c2 = st.columns(2)
    fpr, tpr, _ = roc_curve(splits.y_test, proba_test)
    prec, rec, _ = precision_recall_curve(splits.y_test, proba_test)
    f1 = go.Figure(go.Scatter(x=fpr, y=tpr, name="ROC"))
    f1.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="chance"))
    f1.update_layout(title="ROC curve (test)", xaxis_title="FPR", yaxis_title="TPR")
    f2 = go.Figure(go.Scatter(x=rec, y=prec, name="PR"))
    f2.add_hline(y=float(splits.y_test.mean()), line_dash="dash")
    f2.update_layout(title="Precision-recall curve (test)", xaxis_title="Recall", yaxis_title="Precision")
    c1.plotly_chart(style_fig(f1), width="stretch")
    c2.plotly_chart(style_fig(f2), width="stretch")
    pred = (proba_test >= thr_saved).astype(int)
    cm = pd.crosstab(splits.y_test.rename("actual"), pd.Series(pred, index=splits.y_test.index, name="predicted"))
    cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    st.plotly_chart(style_fig(go.Figure(go.Heatmap(z=cm.values, x=["Normal", "Failure"], y=["Normal", "Failure"],
                                                    text=cm.values, texttemplate="%{text}", colorscale="Blues")).update_layout(
        title=f"Confusion matrix (test, thr={thr_saved:.2f})", xaxis_title="Predicted", yaxis_title="Actual"), 360),
        width="stretch")
with tab_report:
    rep = classification_report(splits.y_test, (proba_test >= thr_saved).astype(int), output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T, width="stretch")
    if report:
        st.write("**Recorded evaluation (MLflow artifact)**")
        st.json({"validation": report["validation_metrics"], "test_final": report["test_metrics_final_evaluation"]})
with tab_shap:
    if not supports_shap(model):
        st.info("SHAP is available for tree-based models (XGBoost / Random Forest).")
    else:
        sample = splits.X_val.head(400)
        vals, _, names = shap_values(model, sample)
        imp = pd.Series(np.abs(vals).mean(axis=0), index=names).sort_values()
        st.plotly_chart(style_fig(go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h")).update_layout(
            title="Global importance - mean |SHAP| (validation sample)"), 420), width="stretch")
        p = mlflow_service.download_artifact(run_id, "explainability/shap_summary.png") if run_id else None
        if p:
            st.image(p, caption="SHAP summary (logged with the run)", width=560)
