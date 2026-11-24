"""Cleaning, missing-value handling, and reproducible customer-level splits.

Operates on the wide (customer x date) reading matrix produced by
`src.data.loader.load_raw_data`. All steps here are row-wise (per customer)
and label-blind, so they can run before splitting without introducing
train/val/test leakage; only normalization (fit on training customers only)
needs the split to already exist.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import DatasetSchema


@dataclass
class CleaningReport:
    n_duplicate_rows_dropped: int
    n_negative_readings_nulled: int
    n_extreme_readings_nulled: int
    n_customers_dropped_low_coverage: int


def clean_dataset(
    df: pd.DataFrame, schema: DatasetSchema, config: dict
) -> tuple[pd.DataFrame, np.ndarray, CleaningReport]:
    """Return (ids_and_labels_df, reading_matrix[float32], report).

    `reading_matrix` has shape (n_customers, n_dates) and uses NaN for
    missing/invalid readings; genuine zero consumption is kept as 0.
    """
    df = df.drop_duplicates(subset=[schema.id_col], keep="first").reset_index(drop=True)
    n_dupes = len(df) - df[schema.id_col].nunique()  # should be 0 after drop; kept for the report
    before = len(df)

    readings = df[schema.date_cols_sorted].apply(pd.to_numeric, errors="coerce")
    matrix = readings.to_numpy(dtype=np.float32)

    n_negative = int(np.nansum(matrix < 0))
    matrix[matrix < 0] = np.nan  # negative consumption is physically impossible -> missing

    z_thresh = config["preprocessing"]["extreme_value_zscore_threshold"]
    finite = matrix[np.isfinite(matrix)]
    n_extreme = 0
    if finite.size:
        mean, std = finite.mean(), finite.std()
        if std > 0:
            z = np.abs((matrix - mean) / std)
            extreme_mask = np.isfinite(matrix) & (z > z_thresh)
            n_extreme = int(extreme_mask.sum())
            # Extreme spikes are most consistent with meter/transmission errors;
            # treated as missing (to be imputed) rather than capped, to avoid
            # silently injecting a biased value.
            matrix[extreme_mask] = np.nan

    coverage = np.isfinite(matrix).mean(axis=1)
    min_coverage = config["preprocessing"]["min_customer_coverage"]
    keep_mask = coverage >= min_coverage
    n_dropped = int((~keep_mask).sum())

    ids_labels = df[[schema.id_col, schema.label_col]].reset_index(drop=True)
    ids_labels = ids_labels.loc[keep_mask].reset_index(drop=True)
    matrix = matrix[keep_mask]

    report = CleaningReport(
        n_duplicate_rows_dropped=before - len(df) + n_dupes,
        n_negative_readings_nulled=n_negative,
        n_extreme_readings_nulled=n_extreme,
        n_customers_dropped_low_coverage=n_dropped,
    )
    return ids_labels, matrix, report


def impute_missing(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-customer median imputation; falls back to the training-wide median
    for any customer whose entire row is missing (kept distinct via the
    returned missing-fraction array, used as a feature rather than discarded).
    """
    missing_frac = np.isnan(matrix).mean(axis=1)
    row_median = np.nanmedian(matrix, axis=1)
    global_median = np.nanmedian(matrix)
    row_median = np.where(np.isnan(row_median), global_median, row_median)

    imputed = matrix.copy()
    nan_mask = np.isnan(imputed)
    row_idx = np.where(nan_mask)[0]
    imputed[nan_mask] = row_median[row_idx]
    return imputed, missing_frac


def split_customers(
    ids_labels: pd.DataFrame, schema: DatasetSchema, config: dict
) -> dict[str, np.ndarray]:
    """Stratified, seeded customer-level split. Returns row-index arrays
    (positional, into `ids_labels`/`matrix`), not customer IDs, so callers
    can directly slice the aligned reading matrix.
    """
    seed = config["seed"]
    train_frac = config["split"]["train_frac"]
    val_frac = config["split"]["val_frac"]
    test_frac = config["split"]["test_frac"]
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9

    idx = np.arange(len(ids_labels))
    labels = ids_labels[schema.label_col].to_numpy()

    idx_train, idx_rest, y_train, y_rest = train_test_split(
        idx, labels, train_size=train_frac, random_state=seed, stratify=labels
    )
    val_ratio_of_rest = val_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(
        idx_rest, train_size=val_ratio_of_rest, random_state=seed, stratify=y_rest
    )

    assert set(idx_train).isdisjoint(idx_val)
    assert set(idx_train).isdisjoint(idx_test)
    assert set(idx_val).isdisjoint(idx_test)

    return {"train": idx_train, "val": idx_val, "test": idx_test}
