"""HTTP client for the FastAPI service. Returns explicit error information - never fabricated values."""

from __future__ import annotations

from typing import Any

import httpx

from dashboard.services.common import api_base_url


def _get(path: str, timeout: float = 4.0) -> tuple[bool, Any]:
    try:
        r = httpx.get(f"{api_base_url()}{path}", timeout=timeout)
        return r.status_code == 200, (r.json() if "json" in r.headers.get("content-type", "") else r.text)
    except httpx.HTTPError as exc:
        return False, str(exc)


def health() -> tuple[bool, Any]:
    return _get("/health")


def ready() -> tuple[bool, Any]:
    return _get("/ready")


def model_info() -> tuple[bool, Any]:
    return _get("/model/info")


def predict(payload: dict[str, Any], explain: bool = True, api_key: str | None = None) -> tuple[bool, Any]:
    headers = {"X-API-Key": api_key} if api_key else {}
    try:
        r = httpx.post(f"{api_base_url()}/predict", params={"explain": str(explain).lower()}, json=payload,
                       headers=headers, timeout=15.0)
        return r.status_code == 200, r.json()
    except httpx.HTTPError as exc:
        return False, str(exc)
