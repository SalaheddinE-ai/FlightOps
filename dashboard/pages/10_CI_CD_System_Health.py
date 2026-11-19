"""Page 10 - CI/CD and system health (no fabricated statuses)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import json  # noqa: E402
import os  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import httpx  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.components.ui import pill, setup_page  # noqa: E402
from dashboard.services.common import api_base_url, mlflow_ui_url  # noqa: E402
from flightops_ml.config import get_paths, tracking_uri  # noqa: E402
from flightops_ml.monitoring.monitoring import list_monitoring_reports  # noqa: E402

setup_page("CI/CD & System Health", "🩺")
paths = get_paths()


def mlflow_health_base() -> str:
    """Inside Docker the tracking URI points at the service name; locally fall back to the UI URL."""
    uri = tracking_uri()
    return uri.rstrip("/") if uri.startswith("http") else mlflow_ui_url()


def probe(url: str) -> tuple[bool, str]:
    try:
        r = httpx.get(url, timeout=3.0)
        return r.status_code == 200, f"HTTP {r.status_code}"
    except httpx.HTTPError as exc:
        return False, type(exc).__name__


st.subheader("Services")
checks = {"FastAPI /health": probe(f"{api_base_url()}/health"),
          "FastAPI /ready (model loaded)": probe(f"{api_base_url()}/ready"),
          "MLflow tracking server": probe(f"{mlflow_health_base()}/health")}
for name, (ok, detail) in checks.items():
    st.markdown(f"**{name}** {pill('up' if ok else 'down', 'ok' if ok else 'bad')} <small>{detail}</small>",
                unsafe_allow_html=True)
st.caption("MLflow shows 'down' when the platform runs against a local SQLite store without a tracking server.")

st.subheader("Latest training run")
sp = paths.evaluations / "latest_summary.json"
if sp.exists():
    s = json.loads(sp.read_text())
    st.write(f"`{s['selected_model']}` v{s.get('registered_version')} - dataset `{s['dataset_version']}` - {s['generated_at']}")
    st.write("Gates: " + (", ".join(("✅ " if g["passed"] else "❌ ") + g["name"] for g in s.get("gates", [])) or "n/a"))
else:
    st.info("No training run recorded.")

st.subheader("Test status")
junit = paths.reports / "junit.xml"
if junit.exists():
    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite[0]
    st.write(f"Last local pytest run: {suite.get('tests')} tests, {suite.get('failures')} failures, "
             f"{suite.get('errors')} errors, {suite.get('skipped')} skipped")
else:
    st.info("No local test report. Run `pytest --junitxml reports/junit.xml` to populate this panel.")

st.subheader("CI workflow status (GitHub Actions)")
repo, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
if not repo:
    st.info("Not configured. Set `GITHUB_REPOSITORY` (and `GITHUB_TOKEN` for private repos) to query workflow runs.")
else:
    try:
        r = httpx.get(f"https://api.github.com/repos/{repo}/actions/runs", params={"per_page": 5}, timeout=6.0,
                      headers={"Authorization": f"Bearer {token}"} if token else {})
        r.raise_for_status()
        for run in r.json().get("workflow_runs", []):
            st.write(f"{run['name']} - `{run['status']}` / `{run.get('conclusion')}` - {run['head_branch']} - {run['created_at']}")
    except httpx.HTTPError as exc:
        st.error(f"GitHub API unavailable: {exc}")

st.subheader("Monitoring status")
reps = list_monitoring_reports()
st.write(json.loads(reps[0].read_text()).get("status", "UNKNOWN") if reps else "No monitoring report generated yet.")
