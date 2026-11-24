import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predictor import TheftRiskPredictor

PROCESSED_QUICK = ROOT / "data" / "processed_quick"
MODELS_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"


def _config():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


@pytest.mark.skipif(
    not (MODELS_DIR / "smoketest_sage.pt").exists() or not PROCESSED_QUICK.exists(),
    reason="requires quick-mode processed artifacts + a trained smoketest checkpoint",
)
def test_predict_by_customer_id_for_train_and_held_out_customer():
    config = _config()
    predictor = TheftRiskPredictor(PROCESSED_QUICK, MODELS_DIR, METRICS_DIR, config,
                                    model_tag="smoketest_sage")

    train_row = int(predictor.splits["train"][0])
    train_customer_id = predictor.customer_ids[train_row]
    result = predictor.predict_by_customer_id(train_customer_id)
    assert result is not None
    assert result["split"] == "train"
    assert 0.0 <= result["theft_probability"] <= 1.0
    assert result["risk_category"] in ("Low", "Medium", "High")

    test_row = int(predictor.splits["test"][0])
    test_customer_id = predictor.customer_ids[test_row]
    result_unseen = predictor.predict_by_customer_id(test_customer_id)
    assert result_unseen is not None
    assert result_unseen["split"] == "test"
    assert 0.0 <= result_unseen["theft_probability"] <= 1.0


@pytest.mark.skipif(
    not (MODELS_DIR / "smoketest_sage.pt").exists() or not PROCESSED_QUICK.exists(),
    reason="requires quick-mode processed artifacts + a trained smoketest checkpoint",
)
def test_unknown_customer_id_returns_none():
    config = _config()
    predictor = TheftRiskPredictor(PROCESSED_QUICK, MODELS_DIR, METRICS_DIR, config,
                                    model_tag="smoketest_sage")
    assert predictor.predict_by_customer_id("NOT_A_REAL_ID") is None
    assert predictor.consumption_series("NOT_A_REAL_ID") is None


@pytest.mark.skipif(
    not (MODELS_DIR / "smoketest_sage.pt").exists() or not PROCESSED_QUICK.exists(),
    reason="requires quick-mode processed artifacts + a trained smoketest checkpoint",
)
def test_consumption_series_returns_dates_and_readings_of_equal_length():
    config = _config()
    predictor = TheftRiskPredictor(PROCESSED_QUICK, MODELS_DIR, METRICS_DIR, config,
                                    model_tag="smoketest_sage")
    cid = predictor.customer_ids[0]
    series = predictor.consumption_series(cid)
    assert series is not None
    assert len(series["dates"]) == len(series["readings"])
