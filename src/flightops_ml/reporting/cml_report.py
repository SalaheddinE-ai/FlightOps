"""Markdown report for CML (``cml comment create``) built from the real latest training summary."""

from __future__ import annotations

from pathlib import Path

from flightops_ml.config import get_paths
from flightops_ml.utils import read_json


def build_cml_report(summary_path: Path | None = None, out_path: Path | None = None) -> Path:
    paths = get_paths()
    summary = read_json(summary_path or paths.evaluations / "latest_summary.json")
    out_path = out_path or paths.reports / "cml_report.md"
    t, v = summary["test_metrics"], summary["validation_metrics"]
    gates = summary.get("gates", [])
    status = "PASSED" if gates and all(g["passed"] for g in gates) else "NOT PASSED / NOT RUN"
    lines = [
        "## FlightOps ML - training report", "",
        f"**Model:** `{summary['selected_model']}`  |  **Registry version:** `{summary.get('registered_version')}`  "
        f"|  **Dataset:** `{summary['dataset_version']}`  |  **Commit:** `{(summary.get('git_commit') or 'n/a')[:8]}`",
        "", "> AI4I 2020 is a **synthetic** dataset. These metrics do not establish production performance.", "",
        f"### Metrics (threshold {summary['threshold']:.2f}, strategy `{summary['threshold_strategy']}`)", "",
        "| Split | Accuracy | Precision | Recall | F1 | PR-AUC | ROC-AUC |", "|---|---|---|---|---|---|---|",
    ]
    for label, m in (("validation", v), ("test (final)", t)):
        lines.append(f"| {label} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                     f"{m['f1']:.4f} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} |")
    lines += ["", "### Model comparison (selection on validation PR-AUC)", "",
              "| Model | val PR-AUC | val F1 | test PR-AUC | test recall | test F1 | selected |", "|---|---|---|---|---|---|---|"]
    for r in summary["comparison"]:
        lines.append(f"| {r['model']} | {r['val_pr_auc']:.4f} | {r['val_f1']:.4f} | {r['test_pr_auc']:.4f} | "
                     f"{r['test_recall']:.4f} | {r['test_f1']:.4f} | {'yes' if r['selected'] else ''} |")
    lines += ["", f"### Validation gates: **{status}**", ""]
    lines += [f"- {'PASS' if g['passed'] else 'FAIL'} `{g['name']}` - {g['detail']}" for g in gates]
    fig_dir = Path(summary["figures_dir"]) / summary["selected_model"]
    for fig, title in (("confusion_matrix", "Confusion matrix"), ("roc_curve", "ROC curve"), ("pr_curve", "Precision-recall curve")):
        p = fig_dir / f"{fig}.png"
        if p.exists():
            try:
                rel = p.relative_to(paths.root)
            except ValueError:
                rel = p
            lines += ["", f"**{title}**", "", f"![{title}]({rel.as_posix()})"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
