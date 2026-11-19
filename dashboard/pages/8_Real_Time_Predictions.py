"""Page 8 - Real-Time Predictions (machine-condition simulator calling the FastAPI service)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import os  # noqa: E402

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.ui import banner, pill, setup_page, style_fig  # noqa: E402
from dashboard.services import api_client  # noqa: E402

setup_page("Real-Time Predictions", "⚙️")
banner("Moving a slider only changes the <b>input</b> sent to the deployed model (inference). It does <b>not</b> retrain "
       "or adapt the model - that only happens through a training job in the Training Lab.", info=True)
ok, info = api_client.model_info()
if not ok:
    st.error(f"Prediction API unavailable or no deployed model: {info}")
    st.stop()

c = st.columns(5)
ptype = c[0].selectbox("Product type", ["L", "M", "H"], index=1)
air = c[1].slider("Air temperature [K]", 290.0, 310.0, 298.1, 0.1)
proc = c[2].slider("Process temperature [K]", 300.0, 320.0, 308.6, 0.1)
rpm = c[3].slider("Rotational speed [rpm]", 1000, 3000, 1551, 10)
torque = c[4].slider("Torque [Nm]", 1.0, 80.0, 42.8, 0.1)
wear = st.slider("Tool wear [min]", 0, 300, 120)
payload = {"type": ptype, "air_temperature": air, "process_temperature": proc, "rotational_speed": rpm,
           "torque": torque, "tool_wear": wear}
ok, res = api_client.predict(payload, explain=True, api_key=os.environ.get("FLIGHTOPS_API_KEY"))
if not ok:
    st.error(f"Prediction failed: {res}")
    st.stop()

prev = st.session_state.get("prev_pred")
st.session_state["prev_pred"] = res
p, thr = res["failure_probability"], res["threshold"]
kind = {"LOW": "ok", "MEDIUM": "warn", "HIGH": "bad", "CRITICAL": "bad"}[res["risk_level"]]
left, right = st.columns([1, 1])
with left:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=p * 100, number={"suffix": "%"},
        delta={"reference": prev["failure_probability"] * 100, "suffix": " pts vs previous"} if prev else None,
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#38bdf8"},
               "threshold": {"line": {"color": "#f59e0b", "width": 4}, "value": thr * 100}}))
    st.plotly_chart(style_fig(gauge, 300), width="stretch")
    st.markdown(f"Risk {pill(res['risk_level'], kind)} &nbsp; Classification "
                f"{pill('FAILURE predicted' if res['prediction'] else 'normal', 'bad' if res['prediction'] else 'ok')}",
                unsafe_allow_html=True)
    st.caption(f"Decision threshold **{thr:.2f}** (from the deployed version) - model v{res['model_version']} - "
               f"{res['timestamp']} - request `{res['request_id'][:8]}`")
with right:
    ex = res.get("explanation")
    if ex:
        st.write("**SHAP explanation** (log-odds contribution to failure)")
        contrib = sorted(ex["contributions"], key=lambda c: c["shap_value"])
        fig = go.Figure(go.Bar(x=[c["shap_value"] for c in contrib], y=[c["feature"] for c in contrib], orientation="h",
                               marker_color=["#22c55e" if c["shap_value"] < 0 else "#ef4444" for c in contrib]))
        fig.update_layout(xaxis_title="← lowers risk | raises risk →")
        st.plotly_chart(style_fig(fig, 340), width="stretch")
        st.caption(f"Base value {ex['base_value']:.2f} ({ex['unit']}).")
    else:
        st.info("The deployed model type does not support SHAP explanations.")
