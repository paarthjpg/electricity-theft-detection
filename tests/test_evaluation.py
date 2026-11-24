import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    compute_classification_metrics, precision_recall_at_k,
    select_threshold_from_validation, full_evaluation_report,
)


def test_metric_calculations_are_correct_on_known_example():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.9, 0.4])
    # threshold 0.5 -> predictions: [0,0,1,0,1,1,1,0]
    metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)
    assert metrics["true_positives"] == 3
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 2
    assert metrics["true_negatives"] == 2
    expected_precision = 3 / (3 + 1)
    expected_recall = 3 / (3 + 2)
    assert abs(metrics["precision"] - expected_precision) < 1e-9
    assert abs(metrics["recall"] - expected_recall) < 1e-9


def test_pr_auc_uses_continuous_scores_not_just_threshold():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_score_good = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    y_score_bad = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    good = compute_classification_metrics(y_true, y_score_good)
    bad = compute_classification_metrics(y_true, y_score_bad)
    assert good["pr_auc"] > bad["pr_auc"]


def test_undefined_metrics_handled_for_single_class():
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.4, 0.2, 0.3])
    metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["precision"] == 0.0  # zero_division=0, no positives predicted


def test_confusion_matrix_class_ordering_is_0_then_1():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.1, 0.9])
    metrics = compute_classification_metrics(y_true, y_score, threshold=0.5)
    cm = metrics["confusion_matrix"]
    assert cm == [[2, 0], [0, 2]]  # [[TN, FP], [FN, TP]]


def test_precision_at_k_ranks_by_score_descending():
    y_true = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.1, 0.8, 0.2, 0.3, 0.4, 0.95, 0.05, 0.15, 0.25])
    result = precision_recall_at_k(y_true, y_score, k_fraction=0.3)
    assert result["k"] == 3
    # top-3 by score: indices 6,0,2 (all positive) -> precision@3 == 1.0
    assert result["precision_at_k"] == 1.0


def test_threshold_selection_uses_only_provided_validation_arrays():
    rng = np.random.default_rng(0)
    y_val = rng.integers(0, 2, size=200)
    y_val_score = rng.random(200)
    threshold = select_threshold_from_validation(y_val, y_val_score, strategy="f1")
    assert 0.0 <= threshold <= 1.0


def test_full_report_includes_top_k_precision_fields():
    y_true = np.array([0] * 90 + [1] * 10)
    y_score = np.linspace(0, 1, 100)
    report = full_evaluation_report(y_true, y_score, threshold=0.5)
    assert "precision_at_top_5pct" in report
    assert "precision_at_top_10pct" in report
    assert report["precision_at_top_10pct"]["k"] == 10
