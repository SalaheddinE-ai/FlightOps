"""Page 2 - Data Explorer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.ui import setup_page, style_fig, synthetic_notice  # noqa: E402
from dashboard.services.data_service import dataset_missing_message, get_dataset  # noqa: E402
from flightops_ml.data.schema import NUMERIC_FEATURES, TARGET  # noqa: E402
from flightops_ml.data.validation import build_quality_report  # noqa: E402

setup_page("Data Explorer", "🔎")
synthetic_notice()
df = get_dataset()
if df is None:
    st.warning(dataset_missing_message())
    st.stop()

rep = build_quality_report(df)
c = st.columns(4)
c[0].metric("Rows", f"{rep.n_rows:,}")
c[1].metric("Schema validation", "passed" if rep.passed else "FAILED")
c[2].metric("Missing values", sum(rep.missing_values.values()))
c[3].metric("Duplicate feature rows", rep.duplicate_rows)
for w in rep.warnings:
    st.caption(f"⚠️ {w}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Preview", "Class imbalance", "Distributions", "Correlation", "Statistics"])
with tab1:
    st.dataframe(df.head(200), width="stretch")
with tab2:
    td = rep.target_distribution
    counts = td["counts"]
    fig = px.bar(x=["Normal (0)", "Failure (1)"], y=[counts.get("0", 0), counts.get("1", 0)],
                 labels={"x": "", "y": "observations"}, title="Target distribution")
    st.plotly_chart(style_fig(fig, 320), width="stretch")
    st.write(f"Failure rate **{td['rates'].get('1', 0):.2%}** - imbalance ratio "
             f"**{td.get('imbalance_ratio_neg_to_pos', float('nan')):.1f} : 1** (negative : positive).")
    st.caption("Imbalance is handled with class weighting / `scale_pos_weight` computed on the training split only, "
               "and with PR-AUC-based model selection.")
with tab3:
    feat = st.selectbox("Feature", NUMERIC_FEATURES)
    fig = px.histogram(df, x=feat, color=df[TARGET].map({0: "normal", 1: "failure"}), nbins=50,
                       barmode="overlay", opacity=0.65, marginal="box")
    st.plotly_chart(style_fig(fig, 460), width="stretch")
    st.plotly_chart(style_fig(px.histogram(df, x="type", color=df[TARGET].map({0: "normal", 1: "failure"}),
                                           barmode="group", title="Product type"), 300), width="stretch")
with tab4:
    corr = df[[*NUMERIC_FEATURES, TARGET]].corr()
    st.plotly_chart(style_fig(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1), 520),
                    width="stretch")
with tab5:
    st.dataframe(df[NUMERIC_FEATURES].describe().T, width="stretch")
