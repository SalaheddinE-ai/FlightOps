"""Dark industrial-ops styling, KPI cards and small helpers."""

from __future__ import annotations

import hmac
import os

import plotly.graph_objects as go
import streamlit as st

CSS = """
<style>
:root { --accent:#38bdf8; --panel:#111827; --border:#1f2937; --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #0b1220 0%, #060a12 60%); }
section[data-testid="stSidebar"] { background:#0a0f1a; border-right:1px solid var(--border); }
.kpi { background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
.kpi .label { color:#94a3b8; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }
.kpi .value { color:#f1f5f9; font-size:1.55rem; font-weight:650; margin-top:2px; }
.kpi .sub { color:#64748b; font-size:.75rem; margin-top:2px; }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem; font-weight:600; }
.pill.ok { background:#052e16; color:var(--ok); } .pill.warn { background:#422006; color:var(--warn); }
.pill.bad { background:#450a0a; color:var(--bad); } .pill.idle { background:#1e293b; color:#94a3b8; }
.banner { border-left:3px solid var(--warn); background:#1c1507; padding:8px 12px; border-radius:6px; color:#fde68a; font-size:.85rem; }
.banner.info { border-left-color:var(--accent); background:#0b1b2b; color:#bae6fd; }
h1,h2,h3 { letter-spacing:-.01em; }
</style>
"""


def setup_page(title: str, icon: str = "✈️") -> None:
    st.set_page_config(page_title=f"FlightOps ML - {title}", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown("### ✈️ FlightOps ML")
    st.sidebar.caption("Automated ML pipeline for predictive maintenance")
    st.title(title)


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "idle") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def banner(text: str, info: bool = False) -> None:
    st.markdown(f'<div class="banner{" info" if info else ""}">{text}</div>', unsafe_allow_html=True)


def synthetic_notice() -> None:
    banner("AI4I 2020 is a <b>synthetic</b> predictive-maintenance dataset. Metrics shown here do not establish "
           "performance on real equipment.")


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10))
    return fig


def fmt(v, nd: int = 3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def admin_unlocked() -> bool:
    """Admin controls are disabled unless FLIGHTOPS_ADMIN_TOKEN is set AND the correct token is entered."""
    expected = os.environ.get("FLIGHTOPS_ADMIN_TOKEN")
    if not expected:
        st.info("Admin controls are disabled: set `FLIGHTOPS_ADMIN_TOKEN` to enable them.")
        return False
    if st.session_state.get("admin_ok"):
        return True
    token = st.text_input("Admin token", type="password", key="admin_token_input")
    if token and hmac.compare_digest(token, expected):
        st.session_state["admin_ok"] = True
        st.rerun()
    elif token:
        st.error("Invalid token")
    return False
