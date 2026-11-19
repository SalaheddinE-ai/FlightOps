"""Page 1 - Executive Overview (Streamlit entry point)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402

import streamlit as st  # noqa: E402

from dashboard.components.ui import fmt, kpi, pill, setup_page, synthetic_notice  # noqa: E402
from dashboard.services import api_client, mlflow_service  # noqa: E402
from dashboard.services.data_service import dataset_missing_message, get_dataset  # noqa: E402
from flightops_ml.config import get_paths  # noqa: E402
from flightops_ml.monitoring.monitoring import list_monitoring_reports  # noqa: E402
from flightops_ml.registry.registration import TAG  # noqa: E402

setup_page("Executive Overview")
synthetic_notice()

df = get_dataset()
champ = mlflow_service.champion()
api_ok, _ = api_client.health()
ready_ok, _ = api_client.ready()
reports = list_monitoring_reports()
mon_status = json.loads(reports[0].read_text()).get("status", "UNKNOWN") if reports else "NO REPORT"
summary_path = get_paths().evaluations / "latest_summary.json"
last_train = json.loads(summary_path.read_text())["generated_at"] if summary_path.exists() else "never"

c = st.columns(4)
kpi(c[0], "Dataset size", f"{len(df):,}" if df is not None else "n/a", "observations (synthetic)")
kpi(c[1], "Champion model", champ.tags.get(TAG + "algorithm", "none") if champ else "none",
    "alias `champion`" if champ else "no deployed model")
kpi(c[2], "Model version", f"v{champ.version}" if champ else "n/a",
    champ.tags.get(TAG + "business_status", "") if champ else "")
kpi(c[3], "Last training", last_train[:10] if last_train != "never" else "never", last_train[11:19])
c = st.columns(4)
t = champ.tags if champ else {}
kpi(c[0], "F1 (test)", fmt(float(t[TAG + "test_f1"])) if TAG + "test_f1" in t else "n/a",
    f"threshold {fmt(float(t[TAG + 'threshold']), 2)}" if TAG + "threshold" in t else "")
kpi(c[1], "Recall (test)", fmt(float(t[TAG + "test_recall"])) if TAG + "test_recall" in t else "n/a", "final hold-out")
kpi(c[2], "PR-AUC (test)", fmt(float(t[TAG + "test_pr_auc"])) if TAG + "test_pr_auc" in t else "n/a", "average precision")
kpi(c[3], "API", "online" if api_ok else "offline", "model ready" if ready_ok else "model not ready")

st.markdown("&nbsp;")
left_col, right_col = st.columns(2)
with left_col:
    st.subheader("Platform status")
    st.markdown(f"API {pill('healthy' if api_ok else 'unreachable', 'ok' if api_ok else 'bad')} &nbsp; "
                f"Model {pill('ready' if ready_ok else 'not ready', 'ok' if ready_ok else 'warn')} &nbsp; "
                f"Monitoring {pill(mon_status, 'ok' if mon_status == 'OK' else 'warn' if mon_status != 'NO REPORT' else 'idle')}",
                unsafe_allow_html=True)
with right_col:
    st.subheader("Getting started")
    if df is None:
        st.warning(dataset_missing_message())
    elif champ is None:
        st.info("No champion yet: **Training Lab** → launch a job → **Model Registry** → approve → deploy.")
    else:
        st.success("Champion deployed. Try **Real-Time Predictions** or inspect **Model Lineage**.")
