"""Page 7 - Model Lineage (interactive provenance graph from real metadata)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.ui import banner, setup_page, style_fig  # noqa: E402
from dashboard.services import mlflow_service  # noqa: E402
from dashboard.services.common import client, model_name  # noqa: E402
from flightops_ml.tracking.lineage import fetch_lineage  # noqa: E402

setup_page("Model Lineage", "🧬")
banner("Built only from metadata actually stored by the pipeline (tags, params, artifacts, audit log, prediction log). "
       "Red nodes mark lineage that is <b>missing</b> - nothing is fabricated.", info=True)
vt = mlflow_service.versions()
if vt.empty:
    st.info("No registered versions yet.")
    st.stop()
v = st.selectbox("Registered model version", vt["version"])
g = fetch_lineage(client(), model_name(), str(v))
nodes = g["nodes"]
xs = list(range(len(nodes)))
ys = [0.15 * (1 if i % 2 else -1) for i in xs]
fig = go.Figure()
for e in g["edges"]:
    i, j = (next(k for k, n in enumerate(nodes) if n["id"] == e[s]) for s in ("source", "target"))
    fig.add_trace(go.Scatter(x=[xs[i], xs[j]], y=[ys[i], ys[j]], mode="lines", line=dict(color="#475569", width=2),
                             hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(
    x=xs, y=ys, mode="markers+text", text=[n["label"] for n in nodes],
    textposition=["top center" if y > 0 else "bottom center" for y in ys],
    marker=dict(size=34, color=["#22c55e" if n["status"] == "present" else "#ef4444" for n in nodes],
                line=dict(color="#e2e8f0", width=1)),
    customdata=[n["id"] for n in nodes], hovertext=[n["reason"] or n["status"] for n in nodes],
    hoverinfo="text", showlegend=False, name="nodes"))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False, range=[-0.6, 0.6])
fig.update_layout(title="Dataset → Validation → Preprocessing → Features → Config → Experiment → Run → Artifact → Version → Deployment → Predictions")
event = st.plotly_chart(style_fig(fig, 460), width="stretch", on_select="rerun", selection_mode="points", key="lineage")
picked = None
if event and event.selection and event.selection.points:
    idx = event.selection.points[0].get("point_index")
    if idx is not None and idx < len(nodes):
        picked = nodes[idx]["id"]
node_id = st.selectbox("Inspect node (click a node above or choose here)", [n["id"] for n in nodes],
                       index=[n["id"] for n in nodes].index(picked) if picked else 0,
                       format_func=lambda i: next(n["label"] for n in nodes if n["id"] == i))
node = next(n for n in nodes if n["id"] == node_id)
if node["status"] == "missing":
    st.error(f"Missing lineage: {node['reason']}")
st.json(node["details"])
if g["missing"]:
    st.warning("Missing links: " + ", ".join(g["missing"]))
else:
    st.success("Complete lineage: every link was found in stored metadata.")
