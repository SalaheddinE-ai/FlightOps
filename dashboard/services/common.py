"""Shared dashboard services: paths, environment, cached MLflow client."""

from __future__ import annotations

import os

import streamlit as st

from flightops_ml.config import get_paths, registered_model_name, tracking_uri
from flightops_ml.tracking.mlflow_tracking import get_client


def api_base_url() -> str:
    return os.environ.get("FLIGHTOPS_API_URL", "http://localhost:8000").rstrip("/")


def mlflow_ui_url() -> str:
    return os.environ.get("MLFLOW_UI_URL", "http://localhost:5000").rstrip("/")


@st.cache_resource
def client():
    return get_client(tracking_uri())


def model_name() -> str:
    return registered_model_name()


__all__ = ["api_base_url", "client", "get_paths", "mlflow_ui_url", "model_name"]
