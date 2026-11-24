"""Dataset quality checks run before preprocessing.

Produces a structured report documenting data-quality issues (duplicates,
missing values, negative/extreme readings, class imbalance) rather than
silently handling them — Phase 2 of the project spec requires these to be
surfaced and documented, not assumed away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import DatasetSchema


def validate_raw_dataset(df: pd.DataFrame, schema: DatasetSchema) -> dict:
    report: dict = {}

    ids = df[schema.id_col]
    report["n_rows"] = int(len(df))
    report["n_duplicate_ids"] = int(ids.duplicated().sum())
    report["duplicate_id_examples"] = ids[ids.duplicated()].unique()[:5].tolist()

    labels = df[schema.label_col]
    report["label_values"] = sorted(labels.dropna().unique().tolist())
    report["label_is_binary"] = set(report["label_values"]).issubset({0, 1})
    report["n_missing_labels"] = int(labels.isna().sum())
    class_counts = labels.value_counts().to_dict()
    report["class_counts"] = {int(k): int(v) for k, v in class_counts.items()}
    n_total = len(labels)
    report["class_imbalance_ratio"] = (
        max(class_counts.values()) / min(class_counts.values())
        if len(class_counts) == 2 and min(class_counts.values()) > 0
        else None
    )

    readings = df[schema.date_cols_sorted]
    non_numeric_mask = readings.apply(lambda col: pd.to_numeric(col, errors="coerce")).isna() & readings.notna()
    report["n_nonnumeric_readings"] = int(non_numeric_mask.to_numpy().sum())

    numeric_readings = readings.apply(pd.to_numeric, errors="coerce")
    report["n_missing_readings"] = int(numeric_readings.isna().to_numpy().sum())
    report["pct_missing_readings"] = float(numeric_readings.isna().to_numpy().mean())

    report["n_negative_readings"] = int((numeric_readings < 0).to_numpy().sum())

    finite_vals = numeric_readings.to_numpy()[np.isfinite(numeric_readings.to_numpy())]
    if finite_vals.size:
        mean, std = finite_vals.mean(), finite_vals.std()
        z_threshold = 6.0
        if std > 0:
            report["n_extreme_readings_z6"] = int(
                (np.abs((finite_vals - mean) / std) > z_threshold).sum()
            )
        else:
            report["n_extreme_readings_z6"] = 0
        report["reading_mean"] = float(mean)
        report["reading_std"] = float(std)
        report["reading_max"] = float(finite_vals.max())
        report["reading_min"] = float(finite_vals.min())
    else:
        report["n_extreme_readings_z6"] = 0

    report["n_zero_readings"] = int((numeric_readings == 0).to_numpy().sum())
    report["n_date_columns"] = len(schema.date_cols_sorted)
    report["date_range"] = (str(schema.dates_sorted[0]), str(schema.dates_sorted[-1]))

    return report


def format_report(report: dict) -> str:
    lines = ["Dataset validation report", "=" * 40]
    for key, value in report.items():
        lines.append(f"{key}: {value}")
    return "\n".join(lines)
