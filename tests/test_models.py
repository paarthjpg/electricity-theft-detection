import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.baselines import (
    get_logistic_regression_grid, get_random_forest_grid, get_xgboost_grid,
    build_logistic_regression, build_random_forest, build_xgboost, XGBOOST_AVAILABLE,
)


def _toy_data(n=80, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = rng.integers(0, 2, size=n)
    return X, y


def test_logistic_regression_grid_has_multiple_candidates():
    grid = get_logistic_regression_grid(seed=42)
    assert len(grid) > 1
    assert all(p["class_weight"] == "balanced" for p in grid)


def test_logistic_regression_fits_and_predicts_probabilities():
    X, y = _toy_data()
    params = get_logistic_regression_grid(seed=42)[0]
    model = build_logistic_regression(params)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (X.shape[0],)
    assert np.all((proba >= 0) & (proba <= 1))


def test_random_forest_grid_has_multiple_candidates():
    grid = get_random_forest_grid(seed=42)
    assert len(grid) > 1
    assert all(p["class_weight"] == "balanced" for p in grid)


def test_random_forest_fits_and_predicts_probabilities():
    X, y = _toy_data()
    params = get_random_forest_grid(seed=42)[0]
    model = build_random_forest(params)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (X.shape[0],)
    assert np.all((proba >= 0) & (proba <= 1))
    assert hasattr(model, "feature_importances_")


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not available in this environment")
def test_xgboost_fits_and_predicts_probabilities():
    X, y = _toy_data()
    params = get_xgboost_grid(seed=42, scale_pos_weight=1.0)[0]
    model = build_xgboost(params)
    model.fit(X, y)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == (X.shape[0],)
    assert np.all((proba >= 0) & (proba <= 1))
