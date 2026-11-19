"""Run against a live `docker compose up` stack: FLIGHTOPS_DOCKER_TESTS=1 pytest -m docker"""

import os

import httpx
import pytest

pytestmark = [pytest.mark.docker,
              pytest.mark.skipif(os.environ.get("FLIGHTOPS_DOCKER_TESTS") != "1", reason="Docker stack not requested")]


def test_api_health():
    assert httpx.get("http://localhost:8000/health", timeout=10).json()["status"] == "ok"


def test_mlflow_health():
    assert httpx.get("http://localhost:5000/health", timeout=10).status_code == 200


def test_dashboard_up():
    assert httpx.get("http://localhost:8501/_stcore/health", timeout=10).status_code == 200
