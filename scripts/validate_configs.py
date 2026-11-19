"""CI helper: fail fast when the YAML configuration is inconsistent."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flightops_ml.config import load_model_config, load_monitoring_config, load_training_config  # noqa: E402
from flightops_ml.training.models import MODEL_NAMES  # noqa: E402


def main() -> int:
    t, m, mon = load_training_config(), load_model_config(), load_monitoring_config()
    errors: list[str] = []
    s = t["split"]
    if abs(s["train"] + s["validation"] + s["test"] - 1.0) > 1e-9:
        errors.append("split fractions must sum to 1")
    errors += [f"unknown model '{n}'" for n in t["models"]["enabled"] if n not in MODEL_NAMES]
    if t["threshold"]["strategy"] not in ("default", "best_f1", "recall_constrained", "cost"):
        errors.append("invalid threshold.strategy")
    if t["selection"]["primary_metric"] == "val_accuracy":
        errors.append("model selection must not be based on accuracy alone")
    if set(m["lifecycle"]["statuses"]) != {"EXPERIMENTAL", "VALIDATED", "APPROVED", "DEPLOYED", "ARCHIVED"}:
        errors.append("lifecycle statuses changed unexpectedly")
    if not 0 < mon["drift"]["psi_threshold"] < 5:
        errors.append("monitoring.drift.psi_threshold out of range")
    for e in errors:
        print("CONFIG ERROR:", e)
    print("configuration OK" if not errors else f"{len(errors)} configuration error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
