"""Append-only governance audit log (JSONL). Every promotion decision is recorded."""

from __future__ import annotations

import json
from typing import Any

from flightops_ml.config import get_paths
from flightops_ml.utils import utc_now_iso


def _path():
    return get_paths().governance / "events.jsonl"


def append_event(event_type: str, **fields: Any) -> dict[str, Any]:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"ts": utc_now_iso(), "event": event_type, **fields}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, default=str) + "\n")
    return event


def read_events(event_types: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event_types is None or ev.get("event") in event_types:
                events.append(ev)
    return events
