"""End-to-end smoke test on a small SYNTHETIC fixture (not SGCC data):
load -> preprocess -> features -> split -> baseline train -> evaluation ->
artifact save/load -> inference. Fast, for CI / quick checks.

Never presents these numbers as SGCC results — see data/raw/DATA_SOURCE.md.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.fixtures import make_synthetic_dataframe, make_synthetic_config
from src.data.loader import resolve_schema
from src.data.validation import validate_raw_dataset
from src.data.preprocessing import clean_dataset, impute_missing, split_customers
from src.data.feature_engineering import compute_features, FEATURE_NAMES
from src.models.baselines import get_logistic_regression_grid, build_logistic_regression
from src.evaluation.metrics import full_evaluation_report
from sklearn.preprocessing import StandardScaler
import joblib


def main():
    print("1. Generating synthetic fixture (NOT SGCC data)...")
    df = make_synthetic_dataframe(n_customers=300, n_days=60, theft_rate=0.15)
    config = make_synthetic_config()

    print("2. Resolving schema + validating...")
    schema = resolve_schema(df, config)
    report = validate_raw_dataset(df, schema)
    assert report["label_is_binary"]

    print("3. Cleaning + imputing...")
    ids_labels, matrix_raw, clean_report = clean_dataset(df, schema, config)
    matrix_imputed, missing_frac = impute_missing(matrix_raw)
    assert not np.isnan(matrix_imputed).any()

    print("4. Engineering features...")
    features = compute_features(matrix_raw, matrix_imputed, missing_frac, schema.dates_sorted)
    assert np.isfinite(features).all()

    print("5. Splitting customers...")
    splits = split_customers(ids_labels, schema, config)
    assert set(splits["train"]).isdisjoint(splits["val"])
    assert set(splits["train"]).isdisjoint(splits["test"])

    print("6. Fitting scaler on train only...")
    scaler = StandardScaler().fit(features[splits["train"]])
    features_scaled = scaler.transform(features).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        processed_dir = tmp_path / "processed"
        models_dir = tmp_path / "models"
        metrics_dir = tmp_path / "metrics"
        for d in (processed_dir, models_dir, metrics_dir):
            d.mkdir(parents=True)

        np.save(processed_dir / "features_scaled.npy", features_scaled)
        np.save(processed_dir / "labels.npy", ids_labels[schema.label_col].to_numpy())
        np.savez(processed_dir / "splits.npz", **splits)
        joblib.dump(scaler, models_dir / "feature_scaler.joblib")

        print("7. Training baseline (Logistic Regression)...")
        y_train = ids_labels[schema.label_col].to_numpy()[splits["train"]]
        y_test = ids_labels[schema.label_col].to_numpy()[splits["test"]]
        model = build_logistic_regression(get_logistic_regression_grid(config["seed"])[0])
        model.fit(features_scaled[splits["train"]], y_train)
        test_score = model.predict_proba(features_scaled[splits["test"]])[:, 1]
        baseline_report = full_evaluation_report(y_test, test_score, threshold=0.5)
        assert baseline_report["pr_auc"] is not None

        print("8. Saving checkpoint and reloading for inference...")
        joblib.dump(model, models_dir / "smoke_logistic_regression.joblib")
        reloaded = joblib.load(models_dir / "smoke_logistic_regression.joblib")
        reloaded_score = reloaded.predict_proba(features_scaled[splits["test"]])[:, 1]
        assert np.allclose(test_score, reloaded_score)

    print("\nSMOKE TEST PASSED: full pipeline runs end-to-end on synthetic data with no leakage violations.")


if __name__ == "__main__":
    main()
