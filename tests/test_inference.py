import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predictor import TheftRiskPredictor

PROCESSED_QUICK = ROOT / "data" / "processed_quick"
MODELS_DIR = ROOT / "artifacts" / "models"
METRICS_DIR = ROOT / "artifacts" / "metrics"
MODEL_TAG = "xgboost"


def _config():
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / f"{MODEL_TAG}.joblib").exists() or not PROCESSED_QUICK.exists(),
    reason="requires quick-mode processed artifacts + a trained baseline checkpoint",
)


def test_predict_by_customer_id_for_train_and_held_out_customer():
    config = _config()
    predictor = TheftRiskPredictor(PROCESSED_QUICK, MODELS_DIR, METRICS_DIR, config, model_tag=MODEL_TAG)

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


def test_unknown_customer_id_returns_none():
    config = _config()
    predictor = TheftRiskPredictor(PROCESSED_QUICK, MODELS_DIR, METRICS_DIR, config, model_tag=MODEL_TAG)
    assert predictor.predict_by_customer_id("NOT_A_REAL_ID") is None
    assert predictor.consumption_series("NOT_A_REAL_ID") is None


def test_consumption_series_returns_dates_and_readings_of_equal_length():
    config = _config()
    predictor = TheftRiskPredictor(PROCESSED_QUICK, MODELS_DIR, METRICS_DIR, config, model_tag=MODEL_TAG)
    cid = predictor.customer_ids[0]
    series = predictor.consumption_series(cid)
    assert series is not None
    assert len(series["dates"]) == len(series["readings"])


def test_predict_rows_batch_matches_per_row_predictions():
    config = _config()
    predictor = TheftRiskPredictor(PROCESSED_QUICK, MODELS_DIR, METRICS_DIR, config, model_tag=MODEL_TAG)
    rows = predictor.splits["test"][:5]
    batch_probs = predictor.predict_rows_batch(rows)
    per_row_probs = [predictor.predict_row(int(r))["theft_probability"] for r in rows]
    assert batch_probs == pytest.approx(per_row_probs, abs=1e-6)
