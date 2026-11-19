"""Minimal in-process operational metrics exposed in Prometheus text format (no extra dependency)."""

from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[tuple[str, int], int] = defaultdict(int)
        self.latency_sum: dict[str, float] = defaultdict(float)
        self.latency_count: dict[str, int] = defaultdict(int)
        self.predictions = 0
        self.prediction_errors = 0
        self.invalid_inputs = 0

    def observe(self, path: str, status: int, seconds: float) -> None:
        with self._lock:
            self.requests[(path, status)] += 1
            self.latency_sum[path] += seconds
            self.latency_count[path] += 1

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + n)

    def render(self, model_loaded: bool, model_version: str | None) -> str:
        lines = ["# TYPE flightops_requests_total counter"]
        with self._lock:
            for (path, status), n in sorted(self.requests.items()):
                lines.append(f'flightops_requests_total{{path="{path}",status="{status}"}} {n}')
            lines.append("# TYPE flightops_request_latency_seconds summary")
            for path in sorted(self.latency_sum):
                lines.append(f'flightops_request_latency_seconds_sum{{path="{path}"}} {self.latency_sum[path]:.6f}')
                lines.append(f'flightops_request_latency_seconds_count{{path="{path}"}} {self.latency_count[path]}')
            lines += [
                "# TYPE flightops_predictions_total counter",
                f"flightops_predictions_total {self.predictions}",
                f"flightops_prediction_errors_total {self.prediction_errors}",
                f"flightops_invalid_inputs_total {self.invalid_inputs}",
                "# TYPE flightops_model_loaded gauge",
                f"flightops_model_loaded {1 if model_loaded else 0}",
                f'flightops_model_info{{version="{model_version or "none"}"}} 1',
            ]
        return "\n".join(lines) + "\n"
