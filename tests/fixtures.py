"""Small synthetic dataset for fast software tests only.

This is NOT the SGCC dataset and must never be used to report results that
are presented as SGCC results — see data/raw/DATA_SOURCE.md.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

SYNTHETIC_SEED = 123


def make_synthetic_dataframe(n_customers: int = 60, n_days: int = 40, theft_rate: float = 0.2):
    rng = np.random.default_rng(SYNTHETIC_SEED)
    dates = [dt.date(2014, 1, 1) + dt.timedelta(days=i) for i in range(n_days)]
    date_cols = [f"{d.year}/{d.month}/{d.day}" for d in dates]

    n_theft = int(round(n_customers * theft_rate))
    labels = np.array([1] * n_theft + [0] * (n_customers - n_theft))
    rng.shuffle(labels)

    base = rng.normal(loc=20, scale=5, size=(n_customers, n_days))
    base[labels == 1] *= rng.uniform(0.3, 0.6, size=(n_theft, 1)).repeat(1, axis=0) if n_theft else 1
    base = np.clip(base, 0, None)

    nan_mask = rng.random((n_customers, n_days)) < 0.1
    base[nan_mask] = np.nan

    cons_no = [f"CUST_{i:05d}" for i in range(n_customers)]
    df = pd.DataFrame(base, columns=date_cols)
    df.insert(0, "FLAG", labels)
    df.insert(0, "CONS_NO", cons_no)
    # shuffle date columns to mimic the real file's unordered headers
    shuffled_date_cols = list(date_cols)
    rng.shuffle(shuffled_date_cols)
    df = df[["CONS_NO", "FLAG"] + shuffled_date_cols]
    return df


def make_synthetic_config(tmp_path=None):
    return {
        "seed": 42,
        "schema": {
            "customer_id_col": "CONS_NO",
            "label_col": "FLAG",
            "date_column_format": "%Y/%m/%d",
        },
        "split": {"train_frac": 0.7, "val_frac": 0.15, "test_frac": 0.15},
        "preprocessing": {"extreme_value_zscore_threshold": 6.0, "min_customer_coverage": 0.1},
        "graph": {"k": 5, "metric": "cosine", "self_loops": True},
        "model": {"hidden_dim_1": 16, "hidden_dim_2": 8, "dropout": 0.3,
                   "learning_rate": 0.01, "weight_decay": 0.0005},
        "quick_mode": {"enabled": True, "subset_size": 60, "max_epochs": 5, "patience": 3},
        "full_mode": {"max_epochs": 5, "patience": 3},
        "risk_thresholds": {"low_max": 0.33, "medium_max": 0.66},
    }
