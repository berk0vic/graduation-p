"""
Evaluation metrics for imbalanced fraud detection.
"""
import numpy as np
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)


def optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Find threshold that maximizes F1 on minority (fraud) class."""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_f1, best_t = 0.0, 0.5
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float | None = None,
) -> dict:
    if threshold is None:
        threshold = optimal_threshold(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)

    return {
        "threshold": threshold,
        "f1_fraud": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_fraud": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "precision_fraud": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_scores),
        "avg_precision": average_precision_score(y_true, y_scores),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def print_report(y_true: np.ndarray, y_scores: np.ndarray) -> None:
    metrics = compute_metrics(y_true, y_scores)
    print(f"\n--- Fraud Detection Metrics (threshold={metrics['threshold']:.3f}) ---")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k:20s}: {v:.4f}")
    cm = np.array(metrics["confusion_matrix"])
    print(f"\n  Confusion Matrix:\n{cm}")
