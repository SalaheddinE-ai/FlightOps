"""Page 9 - Model Monitoring and drift simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import json  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.ui import banner, setup_page, style_fig  # noqa: E402
from dashboard.services import mlflow_service  # noqa: E402
from dashboard.services.common import client, model_name  # noqa: E402
from dashboard.services.data_service import get_dataset  # noqa: E402
from flightops_ml.config import get_paths, load_monitoring_config  # noqa: E402
from flightops_ml.data.schema import NUMERIC_FEATURES, RAW_FEATURES  # noqa: E402
from flightops_ml.monitoring.adaptation import run_adaptation_simulation  # noqa: E402
from flightops_ml.monitoring.drift import (  # noqa: E402
    classify_drift,
    dataset_drift_summary,
    feature_drift,
    simulate_shift,
)
from flightops_ml.monitoring.monitoring import (  # noqa: E402
    build_monitoring_report,
    list_monitoring_reports,
    save_monitoring_report,
)
from flightops_ml.monitoring.prediction_log import PredictionLog  # noqa: E402

setup_page("Model Monitoring", "📡")
paths = get_paths()
log = PredictionLog()
ref_path = paths.data_reference / "reference_features.csv"
ref = pd.read_csv(ref_path) if ref_path.exists() else None
preds, errs, labels = log.predictions(), log.errors(), log.labels()

tab_live, tab_labels, tab_sim, tab_adapt, tab_hist = st.tabs(
    ["Live monitoring", "Ground-truth labels", "Drift simulation (synthetic)", "Adaptation simulation (synthetic)", "History"])

with tab_live:
    if ref is None:
        st.warning("No reference data yet - train a model first (reference is written at training time).")
    else:
        proba_ref = paths.data_reference / "reference_proba.csv"
        meta = json.loads((paths.data_reference / "reference_meta.json").read_text())
        if st.button("Generate monitoring report"):
            rep = build_monitoring_report(ref, preds, errs, labels,
                                          pd.read_csv(proba_ref)["probability"].to_numpy() if proba_ref.exists() else None,
                                          baseline_recall=meta.get("baseline_test_recall"), cfg=load_monitoring_config())
            st.caption(f"Saved to `{save_monitoring_report(rep)}`")
            st.session_state["mon_report"] = rep
        rep = st.session_state.get("mon_report")
        c = st.columns(4)
        c[0].metric("Predictions logged", len(preds))
        c[1].metric("API errors", len(errs))
        c[2].metric("p95 latency (ms)", f"{preds['latency_ms'].quantile(0.95):.1f}" if len(preds) else "n/a")
        c[3].metric("Positive rate", f"{preds['prediction'].mean():.1%}" if len(preds) else "n/a")
        if len(preds):
            left_col, right_col = st.columns(2)
            left_col.plotly_chart(style_fig(px.histogram(preds, x="probability", nbins=30, title="Predicted failure probability"), 300),
                           width="stretch")
            vol = preds.assign(ts=pd.to_datetime(preds["ts"])).set_index("ts").resample("1h").size().rename("n").reset_index()
            right_col.plotly_chart(style_fig(px.bar(vol, x="ts", y="n", title="Prediction volume per hour"), 300), width="stretch")
            st.write("**Model version usage:**", preds["model_version"].value_counts().to_dict())
        else:
            st.info("No predictions logged yet - use Real-Time Predictions or call the API.")
        if rep:
            st.subheader(f"Status: {rep['status']}")
            for a in rep["alerts"]:
                st.warning(a)
            st.json(rep["interpretation"])
            if rep["drift"].get("status") == "ok":
                st.dataframe(pd.DataFrame(rep["drift"]["features"]), width="stretch", hide_index=True)
            else:
                st.caption(f"Drift needs at least {load_monitoring_config()['drift']['min_rows']} logged predictions.")
            st.write("**Performance (labelled predictions only):**", rep["performance"])

with tab_labels:
    st.caption("Performance can only be computed once ground truth arrives. Upload a CSV with `request_id,label`.")
    up = st.file_uploader("Labels CSV", type=["csv"])
    if up is not None and st.button("Store labels"):
        try:
            st.success(f"Stored {log.add_labels(pd.read_csv(up))} labels")
        except ValueError as exc:
            st.error(str(exc))
    st.write(f"Labels stored: **{len(labels)}**")

with tab_sim:
    banner("<b>SYNTHETIC drift scenario.</b> Shifts are applied artificially to reference data; they do not represent real "
           "equipment degradation.")
    if ref is None:
        st.stop()
    scen = st.selectbox("Scenario", list(load_monitoring_config()["simulation"]["scenarios"]))
    shifted = simulate_shift(ref[RAW_FEATURES], scen)
    fd = feature_drift(ref, shifted)
    summ = dataset_drift_summary(fd)
    st.dataframe(fd, width="stretch", hide_index=True)
    feat = st.selectbox("Compare feature", NUMERIC_FEATURES)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ref[feat], name="baseline", opacity=0.6))
    fig.add_trace(go.Histogram(x=shifted[feat], name="shifted (synthetic)", opacity=0.6))
    fig.update_layout(barmode="overlay")
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    cls = classify_drift(summ, None, None)
    st.write("**Interpretation**")
    for n in cls["notes"]:
        st.write("• " + n)
    st.caption(f"Covariate drift: {cls['covariate_drift']} | Prediction drift: not evaluated here | "
               f"Performance: {cls['performance']}")

with tab_adapt:
    banner("<b>SYNTHETIC adaptation experiment.</b> A shifted dataset is generated, the deployed model is compared "
           "with a retrained candidate. Not real equipment degradation; nothing is deployed.")
    champ = mlflow_service.champion()
    df = get_dataset()
    if champ is None or df is None:
        st.info("Needs a deployed champion and the dataset.")
    else:
        c = st.columns(3)
        air = c[0].slider("Air/process temperature shift [K]", 0.0, 8.0, 3.0, 0.5)
        tq = c[1].slider("Torque factor", 1.0, 1.6, 1.2, 0.05)
        wear = c[2].slider("Tool-wear shift [min]", 0, 80, 20, 5)
        if st.button("Run simulation"):
            import mlflow

            from flightops_ml.registry.registration import TAG
            client()
            model = mlflow.sklearn.load_model(f"models:/{model_name()}@champion")
            with st.spinner("Retraining a candidate on baseline + shifted data..."):
                out = run_adaptation_simulation(
                    model, float(champ.tags[TAG + "threshold"]), df,
                    {"air_shift_k": air, "process_shift_k": air, "torque_factor": tq, "wear_shift_min": wear})
            st.warning(out["label"])
            st.dataframe(pd.DataFrame(out["drift_by_feature"]), width="stretch", hide_index=True)
            comp = pd.DataFrame({"original model": out["original_model"], "retrained candidate": out["retrained_candidate"]}
                                ).loc[["pr_auc", "recall", "precision", "f1"]]
            st.dataframe(comp, width="stretch")
            st.plotly_chart(style_fig(px.bar(comp.reset_index().melt("index"), x="index", y="value", color="variable",
                                             barmode="group"), 320), width="stretch")
            st.write("**Candidate promotion decision (gates only):** " + ("would pass" if out["would_pass_gates"] else "would NOT pass"))
            for g in out["gates"]:
                st.write(("✅ " if g["passed"] else "❌ ") + f"`{g['name']}` - {g['detail']}")
            st.caption(out["note"])

with tab_hist:
    reps = list_monitoring_reports()
    if not reps:
        st.info("No saved monitoring reports yet.")
    else:
        rows = []
        for p in reps:
            d = json.loads(p.read_text())
            rows.append({"file": p.name, "generated_at": d.get("generated_at"), "status": d.get("status"),
                         "n_predictions": d.get("n_predictions"), "alerts": len(d.get("alerts", []))})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
