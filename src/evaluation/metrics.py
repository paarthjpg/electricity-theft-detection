"""Evaluation metrics for imbalanced binary theft classification.

Accuracy alone is not meaningful here (theft prevalence ~8.5%); PR-AUC,
recall, and ranking metrics (Precision@K) matter more because utilities can
only physically inspect a limited number of flagged customers.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, roc_curve, precision_recall_curve,
)


def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    """y_score: continuous predicted probability of the positive (theft) class."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["confusion_matrix"] = cm.tolist()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else None
    metrics["false_negative_rate"] = float(fn / (fn + tp)) if (fn + tp) > 0 else None

    return metrics


def precision_recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k_fraction: float) -> dict:
    """Precision/recall among the top-k_fraction highest-risk customers —
    the operationally relevant metric when inspection capacity is limited.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    k = max(1, int(round(n * k_fraction)))
    order = np.argsort(-y_score)
    top_k_labels = y_true[order[:k]]
    n_positive_total = y_true.sum()

    precision_at_k = top_k_labels.sum() / k if k > 0 else 0.0
    recall_at_k = top_k_labels.sum() / n_positive_total if n_positive_total > 0 else None

    return {
        "k": int(k),
        "k_fraction": float(k_fraction),
        "precision_at_k": float(precision_at_k),
        "recall_at_k": float(recall_at_k) if recall_at_k is not None else None,
    }


def select_threshold_from_validation(y_val_true: np.ndarray, y_val_score: np.ndarray, strategy: str = "f1") -> float:
    """Choose a classification threshold using VALIDATION data only.
    Never call this with test-set scores/labels.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_val_true, y_val_score)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    precisions, recalls = precisions[:-1], recalls[:-1]
    if strategy == "f1":
        f1_scores = np.divide(
            2 * precisions * recalls, precisions + recalls,
            out=np.zeros_like(precisions), where=(precisions + recalls) > 0,
        )
        best_idx = int(np.argmax(f1_scores))
    elif strategy == "high_precision":
        valid = np.where(precisions >= 0.5)[0]
        best_idx = int(valid[np.argmax(recalls[valid])]) if len(valid) else int(np.argmax(precisions))
    else:
        raise ValueError(strategy)
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


def full_evaluation_report(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    report = compute_classification_metrics(y_true, y_score, threshold=threshold)
    report["precision_at_top_5pct"] = precision_recall_at_k(y_true, y_score, 0.05)
    report["precision_at_top_10pct"] = precision_recall_at_k(y_true, y_score, 0.10)
    return report
