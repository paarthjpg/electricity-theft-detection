"""Conventional ML baselines, trained on identical features/splits as the
GNNs for a fair comparison."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def get_logistic_regression_grid(seed: int) -> list[dict]:
    return [{"C": c, "class_weight": "balanced", "max_iter": 2000, "random_state": seed}
            for c in (0.01, 0.1, 1.0, 10.0)]


def get_random_forest_grid(seed: int) -> list[dict]:
    grid = []
    for n_estimators in (100, 300):
        for max_depth in (None, 12):
            grid.append({
                "n_estimators": n_estimators, "max_depth": max_depth,
                "class_weight": "balanced", "random_state": seed, "n_jobs": -1,
            })
    return grid


def get_xgboost_grid(seed: int, scale_pos_weight: float) -> list[dict]:
    grid = []
    for n_estimators in (100, 300):
        for max_depth in (4, 6):
            grid.append({
                "n_estimators": n_estimators, "max_depth": max_depth,
                "scale_pos_weight": scale_pos_weight, "random_state": seed,
                "eval_metric": "aucpr", "use_label_encoder": False,
            })
    return grid


def build_logistic_regression(params: dict) -> LogisticRegression:
    return LogisticRegression(**params)


def build_random_forest(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(**params)


def build_xgboost(params: dict):
    if not XGBOOST_AVAILABLE:
        raise RuntimeError("xgboost is not available in this environment")
    params = {k: v for k, v in params.items() if k != "use_label_encoder"}
    return xgb.XGBClassifier(**params)
