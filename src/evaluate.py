"""Metrics, cost-based threshold selection and evaluation plots.

Accuracy is deliberately reported but never used for any decision. With a 27%
churn rate, "always predict retained" scores 73% accuracy and catches nobody.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config


@dataclass
class Metrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier: float
    tn: int
    fp: int
    fn: int
    tp: int
    expected_cost_per_customer: float
    lift_at_10pct: float

    def as_dict(self) -> dict:
        return asdict(self)


def expected_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Expected cost per customer under the stated business assumptions.

    Each flagged customer costs `COST_PER_INTERVENTION` (the retention offer).
    Each churner we fail to flag costs `COST_PER_MISSED_CHURNER`. A correctly
    flagged churner is saved with probability `INTERVENTION_SUCCESS_RATE`, so the
    residual loss on a true positive is the un-saved fraction.

    These numbers are illustrative, but the *shape* is what matters: a false
    negative costs ~15x a false positive, which is why recall dominates the
    threshold choice.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    cost = (
        (tp + fp) * config.COST_PER_INTERVENTION
        + fn * config.COST_PER_MISSED_CHURNER
        + tp * config.COST_PER_MISSED_CHURNER * (1 - config.INTERVENTION_SUCCESS_RATE)
    )
    return float(cost / n)


def lift_at_k(y_true: np.ndarray, y_proba: np.ndarray, k: float = 0.10) -> float:
    """How much richer in churners the top-k% of the ranked list is vs. random.

    This is the number a campaign manager with a fixed budget actually cares
    about: "if I can only call 1,000 customers, how many churners do I reach?"
    """
    n_top = max(1, int(round(k * len(y_true))))
    order = np.argsort(-y_proba)[:n_top]
    base_rate = y_true.mean()
    return float(y_true[order].mean() / base_rate) if base_rate > 0 else float("nan")


def compute_metrics(y_true, y_proba, threshold: float) -> Metrics:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Metrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        pr_auc=float(average_precision_score(y_true, y_proba)),
        brier=float(brier_score_loss(y_true, y_proba)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        expected_cost_per_customer=expected_cost(y_true, y_pred),
        lift_at_10pct=lift_at_k(y_true, y_proba),
    )


def choose_threshold(y_true, y_proba, grid: np.ndarray | None = None) -> tuple[float, list[dict]]:
    """Pick the probability cut-off that minimises expected business cost.

    Crucially this is fitted on out-of-fold *validation* predictions, never on
    the test set - the threshold is a hyperparameter like any other.
    """
    grid = np.arange(0.05, 0.96, 0.01) if grid is None else grid
    sweep = []
    for t in grid:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        sweep.append(
            {
                "threshold": round(float(t), 3),
                "cost": expected_cost(y_true, y_pred),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )
    best = min(sweep, key=lambda r: r["cost"])
    return best["threshold"], sweep


def save_json(payload: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def make_evaluation_plots(y_true, y_proba, threshold: float, out_dir=config.REPORT_DIR) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import precision_recall_curve, roc_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    written = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    ax = axes[0, 0]
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=13,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["pred: retained", "pred: churn"])
    ax.set_yticks([0, 1], ["true: retained", "true: churn"])
    ax.set_title(f"Confusion matrix @ threshold {threshold:.2f}")

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax = axes[0, 1]
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")

    # Precision-Recall
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    ax = axes[1, 0]
    ax.plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_true, y_proba):.3f}")
    ax.axhline(y_true.mean(), ls="--", color="grey", label=f"base rate = {y_true.mean():.2f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="upper right")

    # Calibration
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    ax = axes[1, 1]
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed churn rate")
    ax.set_title(f"Calibration (Brier = {brier_score_loss(y_true, y_proba):.3f})")
    ax.legend(loc="upper left")

    fig.suptitle("Held-out test set evaluation", fontsize=14)
    fig.tight_layout()
    path = out_dir / "evaluation_curves.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    written.append(str(path))
    return written
