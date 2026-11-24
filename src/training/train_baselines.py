"""Train and evaluate the conventional ML baselines (Logistic Regression,
Random Forest, optional XGBoost) on the same splits/features as the GNNs.

Hyperparameter selection uses validation PR-AUC only; the test set is
touched once, at the very end, for final evaluation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.baselines import (
    get_logistic_regression_grid, get_random_forest_grid, get_xgboost_grid,
    build_logistic_regression, build_random_forest, build_xgboost, XGBOOST_AVAILABLE,
)
from src.evaluation.metrics import full_evaluation_report, select_threshold_from_validation


def _select_best(model_builder, grid, X_train, y_train, X_val, y_val):
    from src.evaluation.metrics import compute_classification_metrics
    best_score, best_model, best_params = -1.0, None, None
    for params in grid:
        model = model_builder(params)
        model.fit(X_train, y_train)
        val_score = model.predict_proba(X_val)[:, 1]
        pr_auc = compute_classification_metrics(y_val, val_score)["pr_auc"] or 0.0
        if pr_auc > best_score:
            best_score, best_model, best_params = pr_auc, model, params
    return best_model, best_params, best_score


def train_and_evaluate_baselines(processed_dir: Path, models_dir: Path, metrics_dir: Path, config: dict) -> dict:
    seed = config["seed"]
    X = np.load(processed_dir / "features_scaled.npy")
    y = np.load(processed_dir / "labels.npy")
    splits = np.load(processed_dir / "splits.npz")
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    results = {}

    for name, grid, builder in [
        ("logistic_regression", get_logistic_regression_grid(seed), build_logistic_regression),
        ("random_forest", get_random_forest_grid(seed), build_random_forest),
    ]:
        t0 = time.time()
        model, params, val_pr_auc = _select_best(builder, grid, X_train, y_train, X_val, y_val)
        train_time = time.time() - t0

        t0 = time.time()
        test_score = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - t0

        val_score = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold_from_validation(y_val, val_score, strategy="f1")
        report = full_evaluation_report(y_test, test_score, threshold)
        report["selected_params"] = params
        report["val_pr_auc_at_selection"] = val_pr_auc
        report["train_time_seconds"] = train_time
        report["inference_time_seconds"] = inference_time
        report["threshold_default_0.5"] = full_evaluation_report(y_test, test_score, 0.5)

        if name == "random_forest" and hasattr(model, "feature_importances_"):
            report["feature_importances"] = model.feature_importances_.tolist()

        joblib.dump(model, models_dir / f"{name}.joblib")
        results[name] = report
        print(f"{name}: val_pr_auc={val_pr_auc:.4f} test_pr_auc={report['pr_auc']:.4f} "
              f"test_f1={report['f1']:.4f} (params={params})")

    if XGBOOST_AVAILABLE:
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        t0 = time.time()
        model, params, val_pr_auc = _select_best(
            build_xgboost, get_xgboost_grid(seed, pos_weight), X_train, y_train, X_val, y_val
        )
        train_time = time.time() - t0
        t0 = time.time()
        test_score = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - t0
        val_score = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold_from_validation(y_val, val_score, strategy="f1")
        report = full_evaluation_report(y_test, test_score, threshold)
        report["selected_params"] = {k: v for k, v in params.items() if k != "use_label_encoder"}
        report["val_pr_auc_at_selection"] = val_pr_auc
        report["train_time_seconds"] = train_time
        report["inference_time_seconds"] = inference_time
        joblib.dump(model, models_dir / "xgboost.joblib")
        results["xgboost"] = report
        print(f"xgboost: val_pr_auc={val_pr_auc:.4f} test_pr_auc={report['pr_auc']:.4f} test_f1={report['f1']:.4f}")
    else:
        print("xgboost not available in this environment; skipping (optional model).")

    with open(metrics_dir / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    processed_dir = ROOT / config["paths"]["processed_dir"]
    models_dir = ROOT / config["paths"]["models_dir"]
    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    train_and_evaluate_baselines(processed_dir, models_dir, metrics_dir, config)


if __name__ == "__main__":
    main()
