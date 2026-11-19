"""Page 3 - Training Lab: launch jobs, live monitor, training replay, controlled retraining."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from dashboard.components.ui import banner, pill, setup_page, style_fig  # noqa: E402
from dashboard.services import mlflow_service  # noqa: E402
from flightops_ml.config import get_paths  # noqa: E402
from flightops_ml.training import jobs  # noqa: E402

setup_page("Training Lab", "🧪")
tab_launch, tab_live, tab_replay, tab_retrain = st.tabs(
    ["Launch training", "Live training monitor", "Training replay & learning curves", "Controlled retraining"])

with tab_launch:
    st.caption("Starts a REAL training job in a background process (the UI stays responsive).")
    models = st.multiselect("Models", ["logistic_regression", "random_forest", "xgboost"],
                            default=["logistic_regression", "random_forest", "xgboost"])
    c = st.columns(3)
    tune = c[0].toggle("Optuna tuning (XGBoost)", True)
    trials = c[1].number_input("Optuna trials", 1, 200, 20, disabled=not tune)
    strategy = c[2].selectbox("Threshold strategy (validation)", ["best_f1", "recall_constrained", "cost", "default"])
    with st.expander("Baseline XGBoost hyper-parameters (used when tuning is off)"):
        h = st.columns(4)
        xgb = {"learning_rate": h[0].number_input("learning_rate", 0.005, 0.5, 0.05, 0.005, format="%.3f"),
               "max_depth": h[1].number_input("max_depth", 2, 12, 4),
               "n_estimators": h[2].number_input("n_estimators (max)", 50, 3000, 600, 50),
               "subsample": h[3].number_input("subsample", 0.3, 1.0, 0.9, 0.05)}
    if st.button("🚀 Start training job", type="primary", disabled=not models):
        jid = jobs.start_job_process({"models": models, "tune": tune, "n_trials": int(trials),
                                      "threshold_strategy": strategy, "xgboost_params": xgb})
        st.session_state["job_id"] = jid
        st.success(f"Job `{jid}` started - open the *Live training monitor* tab.")

with tab_live:
    all_jobs = jobs.list_jobs(10)

    @st.fragment(run_every=2)
    def live(job_id: str | None):
        st_ = jobs.read_job(job_id) if job_id else None
        if not st_:
            st.info("No job selected.")
            return
        kind = {"completed": "ok", "failed": "bad", "running": "warn", "queued": "idle"}.get(st_["status"], "idle")
        st.markdown(f"Job `{st_['job_id']}` {pill(st_['status'], kind)}", unsafe_allow_html=True)
        c = st.columns(4)
        c[0].metric("Stage", st_.get("stage") or "-")
        c[1].metric("Iteration", st_.get("iteration") or "-")
        c[2].metric("Elapsed (s)", st_.get("elapsed_seconds", 0))
        c[3].metric("Model", st_.get("model") or "-")
        st.caption(f"Experiment `{st_.get('experiment')}` - MLflow run `{st_.get('run_id')}`")
        if st_.get("latest_metrics"):
            lm = st_["latest_metrics"]
            m = st.columns(4)
            for col, key in zip(m, ["train_logloss", "val_logloss", "train_aucpr", "val_aucpr"], strict=True):
                col.metric(key, f"{lm[key]:.4f}" if key in lm else "-")
            if "max_iterations" in lm and st_.get("iteration"):
                st.progress(min(1.0, st_["iteration"] / lm["max_iterations"]),
                            text="boosting iterations (early stopping may end sooner)")
        if st_.get("error"):
            st.error(st_["error"])
        if st_.get("result"):
            st.success(f"Result: {st_['result']}")

    opts = {j["job_id"]: j["job_id"] for j in all_jobs}
    default = st.session_state.get("job_id") or (all_jobs[0]["job_id"] if all_jobs else None)
    sel = st.selectbox("Job", list(opts), index=list(opts).index(default) if default in opts else 0) if opts else None
    live(sel)

with tab_replay:
    banner("<b>Training Replay</b> replays the evaluation history that was RECORDED during a past run. "
           "Nothing is retrained when you move the slider.", info=True)
    runs = mlflow_service.runs_frame("model_training")
    xr = runs[runs["algorithm"] == "xgboost"] if len(runs) else runs
    if not len(xr):
        st.info("No XGBoost training runs recorded yet.")
    else:
        labels = {r.run_id: f"{r['name']} - {r['start']:%Y-%m-%d %H:%M} - {r.run_id[:8]}" for _, r in xr.iterrows()}
        a_id = st.selectbox("Run A", list(labels), format_func=labels.get)
        compare = st.toggle("Compare with a second run")
        b_id = st.selectbox("Run B", list(labels), format_func=labels.get, index=min(1, len(labels) - 1)) if compare else None
        hist_a = mlflow_service.load_history(a_id)
        hist_b = mlflow_service.load_history(b_id) if b_id else None
        if hist_a is None:
            st.warning("This run has no recorded training history.")
        else:
            n = hist_a["n_iterations_run"]
            it = st.slider("Boosting iteration", 1, n, hist_a["best_iteration"])
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Log loss", "AUC-PR"))
            for tag, h, dash in (("A", hist_a, "solid"), ("B", hist_b, "dot")):
                if h is None:
                    continue
                for split, color in (("train", "#38bdf8"), ("validation", "#f59e0b")):
                    for col, metric in ((1, "logloss"), (2, "aucpr")):
                        ys = h[split][metric]
                        fig.add_trace(go.Scatter(x=list(range(1, len(ys) + 1)), y=ys, name=f"{tag} {split}",
                                                 line=dict(color=color, dash=dash), legendgroup=f"{tag}{split}",
                                                 showlegend=(col == 1)), row=1, col=col)
                        if it <= len(ys):
                            fig.add_trace(go.Scatter(x=[it], y=[ys[it - 1]], mode="markers", marker=dict(size=10, color=color),
                                                     showlegend=False), row=1, col=col)
            fig.add_vline(x=hist_a["best_iteration"], line_dash="dash", line_color="#22c55e",
                          annotation_text="best iteration (A)")
            st.plotly_chart(style_fig(fig, 420), width="stretch")
            c = st.columns(4)
            c[0].metric("Best iteration", hist_a["best_iteration"])
            c[1].metric("Iterations run", n)
            c[2].metric("Early stopped", "yes" if hist_a["early_stopped"] else "no")
            c[3].metric(f"val AUC-PR @ {it}", f"{hist_a['validation']['aucpr'][it - 1]:.4f}")
            st.caption(f"Early stopping monitored: {hist_a['early_stopping_metric']}.")

with tab_retrain:
    st.caption("Retraining produces a CANDIDATE only. Approval and deployment stay manual (Model Registry page).")
    up = st.file_uploader("New labelled dataset (AI4I CSV format, must include 'Machine failure')", type=["csv"])
    trig = st.selectbox("Trigger", ["manual", "new_labeled_data", "monitoring_alert", "scheduled"])
    reason = st.text_input("Reason (audit log)")
    if st.button("Run controlled retraining", disabled=up is None):
        dest = get_paths().artifacts / "uploads"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / up.name
        path.write_bytes(up.getvalue())
        jid = jobs.start_job_process({"kind": "retrain", "data_path": str(path), "trigger": trig, "reason": reason})
        st.session_state["job_id"] = jid
        st.success(f"Retraining job `{jid}` started - follow it in *Live training monitor*.")
