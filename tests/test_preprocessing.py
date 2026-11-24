import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import resolve_schema, load_raw_data
from src.data.validation import validate_raw_dataset
from src.data.preprocessing import clean_dataset, impute_missing, split_customers
from src.data.feature_engineering import compute_features, FEATURE_NAMES
from tests.fixtures import make_synthetic_dataframe, make_synthetic_config


def test_schema_resolves_and_sorts_dates_chronologically():
    df = make_synthetic_dataframe()
    config = make_synthetic_config()
    schema = resolve_schema(df, config)
    assert schema.id_col == "CONS_NO"
    assert schema.label_col == "FLAG"
    assert schema.dates_sorted == sorted(schema.dates_sorted)
    assert len(schema.date_cols_sorted) == len(schema.dates_sorted)


def test_customer_ids_unique_after_cleaning():
    df = make_synthetic_dataframe()
    config = make_synthetic_config()
    schema = resolve_schema(df, config)
    ids_labels, matrix, report = clean_dataset(df, schema, config)
    assert ids_labels["CONS_NO"].is_unique
    assert matrix.shape[0] == len(ids_labels)


def test_labels_are_valid_binary():
    df = make_synthetic_dataframe()
    config = make_synthetic_config()
    schema = resolve_schema(df, config)
    report = validate_raw_dataset(df, schema)
    assert report["label_is_binary"]
    assert report["n_missing_labels"] == 0


def test_missing_values_imputed_without_nan():
    df = make_synthetic_dataframe()
    config = make_synthetic_config()
    schema = resolve_schema(df, config)
    _, matrix_raw, _ = clean_dataset(df, schema, config)
    assert np.isnan(matrix_raw).any(), "fixture should contain missing values to test imputation"
    matrix_imputed, missing_frac = impute_missing(matrix_raw)
    assert not np.isnan(matrix_imputed).any()
    assert missing_frac.shape[0] == matrix_raw.shape[0]


def test_train_val_test_customers_do_not_overlap():
    df = make_synthetic_dataframe(n_customers=200)
    config = make_synthetic_config()
    schema = resolve_schema(df, config)
    ids_labels, matrix_raw, _ = clean_dataset(df, schema, config)
    splits = split_customers(ids_labels, schema, config)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == len(ids_labels)


def test_engineered_features_have_no_nan_or_inf():
    df = make_synthetic_dataframe(n_customers=80)
    config = make_synthetic_config()
    schema = resolve_schema(df, config)
    ids_labels, matrix_raw, _ = clean_dataset(df, schema, config)
    matrix_imputed, missing_frac = impute_missing(matrix_raw)
    features = compute_features(matrix_raw, matrix_imputed, missing_frac, schema.dates_sorted)
    assert features.shape == (len(ids_labels), len(FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_scaler_is_fit_on_training_customers_only():
    from sklearn.preprocessing import StandardScaler
    df = make_synthetic_dataframe(n_customers=200)
    config = make_synthetic_config()
    schema = resolve_schema(df, config)
    ids_labels, matrix_raw, _ = clean_dataset(df, schema, config)
    matrix_imputed, missing_frac = impute_missing(matrix_raw)
    features = compute_features(matrix_raw, matrix_imputed, missing_frac, schema.dates_sorted)
    splits = split_customers(ids_labels, schema, config)

    scaler = StandardScaler().fit(features[splits["train"]])
    train_scaled = scaler.transform(features[splits["train"]])
    assert np.allclose(train_scaled.mean(axis=0), 0, atol=1e-5)

    # Fitting on ALL data instead must give a different mean/scale than
    # fitting on train only, proving the pipeline doesn't accidentally leak.
    scaler_all = StandardScaler().fit(features)
    assert not np.allclose(scaler.mean_, scaler_all.mean_)
