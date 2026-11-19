"""Classification metrics and evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def confusion_counts(y_true, proba, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    """Threshold-dependent and threshold-free metrics. Degenerate cases yield NaN, never fake."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    has_both = len(np.unique(y_true)) == 2
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if has_both else float("nan"),
        "brier": float(brier_score_loss(y_true, proba)),
        "threshold": float(threshold),
    }
    counts = confusion_counts(y_true, proba, threshold)
    out.update({k: float(v) for k, v in counts.items()})
    pos = counts["tp"] + counts["fn"]
    out["false_negative_rate"] = float(counts["fn"] / pos) if pos else float("nan")
    return out


def calibration_data(y_true, proba, n_bins: int = 10) -> dict[str, list[float]]:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return {"fraction_positive": frac_pos.tolist(), "mean_predicted": mean_pred.tolist()}


def save_evaluation_plots(y_true, proba, threshold: float, out_dir: Path, prefix: str = "") -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    paths: dict[str, Path] = {}

    cm = confusion_matrix(y_true, (proba >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["Normal", "Failure"])
    ax.set_yticks([0, 1], ["Normal", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (thr={threshold:.2f})")
    paths["confusion_matrix"] = out_dir / f"{prefix}confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(paths["confusion_matrix"], dpi=120)
    plt.close(fig)

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, proba)
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, proba):.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend()
        paths["roc_curve"] = out_dir / f"{prefix}roc_curve.png"
        fig.tight_layout()
        fig.savefig(paths["roc_curve"], dpi=120)
        plt.close(fig)

        prec, rec, _ = precision_recall_curve(y_true, proba)
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.plot(rec, prec, label=f"AP={average_precision_score(y_true, proba):.3f}")
        ax.axhline(y_true.mean(), ls="--", color="grey", label="prevalence")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-recall curve")
        ax.legend()
        paths["pr_curve"] = out_dir / f"{prefix}pr_curve.png"
        fig.tight_layout()
        fig.savefig(paths["pr_curve"], dpi=120)
        plt.close(fig)

        cal = calibration_data(y_true, proba, n_bins=min(10, max(2, int(y_true.sum()))))
        fig, ax = plt.subplots(figsize=(4.5, 3.6))
        ax.plot(cal["mean_predicted"], cal["fraction_positive"], "o-")
        ax.plot([0, 1], [0, 1], "--", color="grey")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed failure fraction")
        ax.set_title("Calibration")
        paths["calibration"] = out_dir / f"{prefix}calibration.png"
        fig.tight_layout()
        fig.savefig(paths["calibration"], dpi=120)
        plt.close(fig)
    return paths
