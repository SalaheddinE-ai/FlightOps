"""Page 6 - Model Registry & governance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import json  # noqa: E402

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.ui import admin_unlocked, banner, pill, setup_page  # noqa: E402
from dashboard.services import mlflow_service  # noqa: E402
from dashboard.services.common import client, mlflow_ui_url, model_name  # noqa: E402
from flightops_ml.registry import audit, promotion  # noqa: E402
from flightops_ml.registry.registration import TAG  # noqa: E402

setup_page("Model Registry", "🗂️")
st.link_button("Open MLflow Model Registry", f"{mlflow_ui_url()}/#/models/{model_name()}")
banner("Business status (EXPERIMENTAL → VALIDATED → APPROVED → DEPLOYED → ARCHIVED) lives in version tags. "
       "<b>Deployed</b> means the version carries the <code>champion</code> alias.", info=True)
vt = mlflow_service.versions()
if vt.empty:
    st.info("No registered versions yet. Run a training job.")
    st.stop()

champ = mlflow_service.champion()
cand = None
try:
    cand = client().get_model_version_by_alias(model_name(), "candidate")
except mlflow.exceptions.MlflowException:
    pass
c = st.columns(3)
c[0].metric("Registered versions", len(vt))
c[1].metric("Current champion", f"v{champ.version}" if champ else "none")
c[2].metric("Current candidate", f"v{cand.version}" if cand else "none")
st.dataframe(vt.drop(columns=["run_id"]), width="stretch", hide_index=True)

st.subheader("Version details")
v = st.selectbox("Version", vt["version"])
mv = client().get_model_version(model_name(), str(v))
st.markdown(f"Status {pill(mv.tags.get(TAG + 'business_status', '?'), 'ok')}  "
            f"Aliases: {', '.join(f'`{a}`' for a in mv.aliases) or 'none'}", unsafe_allow_html=True)
left_col, right_col = st.columns(2)
with left_col:
    st.write("**Tags**")
    st.dataframe(pd.DataFrame([(k.removeprefix(TAG), val) for k, val in mv.tags.items() if k != TAG + "gate_report"],
                              columns=["tag", "value"]), hide_index=True, width="stretch")
with right_col:
    st.write("**Validation gates**")
    gr = mv.tags.get(TAG + "gate_report")
    if gr:
        for g in json.loads(gr):
            st.write(("✅ " if g["passed"] else "❌ ") + f"`{g['name']}` - {g['detail']}")
    else:
        st.caption("Gates not evaluated for this version.")
    try:
        sig = mlflow.models.get_model_info(f"models:/{model_name()}/{v}").signature
        st.write("**Signature inputs:** " + ", ".join(i.name for i in sig.inputs.inputs) if sig else "No signature")
    except mlflow.exceptions.MlflowException:
        st.caption("Signature unavailable.")
    st.write(f"**Source run:** `{mv.tags.get(TAG + 'run_id')}`  \n**Model source:** `{mv.source}`")

st.subheader("Promotion & deployment history")
events = audit.read_events()
if events:
    ev = pd.DataFrame(events)
    st.dataframe(ev.iloc[::-1].astype(str), width="stretch", hide_index=True)
else:
    st.caption("No governance events recorded yet.")

st.subheader("Administration")
st.caption("Governance actions are not exposed through the public API. They require the admin token.")
if admin_unlocked():
    who = st.text_input("Your name (audit trail)")
    why = st.text_input("Reason")
    a, b, d, e = st.columns(4)
    try:
        if a.button("Approve selected", disabled=not (who and why)):
            promotion.approve(client(), model_name(), str(v), who, why)
            st.success(f"v{v} approved")
        if b.button("Deploy selected (set champion)", disabled=not who):
            promotion.deploy(client(), model_name(), str(v), who)
            st.success(f"v{v} is now the champion")
        if d.button("Rollback to previous champion", disabled=not (who and why)):
            st.success(f"Restored v{promotion.rollback(client(), model_name(), who, why)}")
        if e.button("Archive selected", disabled=not why):
            promotion.archive(client(), model_name(), str(v), why)
            st.success(f"v{v} archived")
    except promotion.PromotionError as exc:
        st.error(str(exc))
