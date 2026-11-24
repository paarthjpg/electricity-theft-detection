"""Vectorized customer-level feature engineering.

All functions operate on the full (n_customers, n_dates) matrix at once
(no per-customer Python loops over rows) so this scales to the full 42k x
1034 SGCC matrix. `matrix_raw` (NaN = missing, 0 = genuine zero) is used for
distributional statistics; `matrix_imputed` (no NaN) is used for anything
that needs a complete series, such as rolling windows or trend slope.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

FEATURE_NAMES = [
    "mean_consumption", "median_consumption", "std_consumption",
    "min_consumption", "max_consumption", "coefficient_of_variation",
    "pct_zero_days", "pct_missing", "trend_slope",
    "avg_monthly_consumption", "monthly_variability",
    "weekday_weekend_ratio", "sudden_drop_frequency",
    "max_consecutive_low_run", "first_half_second_half_ratio",
    "rolling_mean_volatility", "rolling_std_mean", "seasonal_variation",
    "consumption_p10", "consumption_p90", "long_term_stability",
]


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    result[~np.isfinite(result)] = 0.0
    return result


def _max_consecutive_run(mask: np.ndarray) -> np.ndarray:
    """Longest run of True values per row, without a per-row Python loop."""
    n_rows, n_cols = mask.shape
    run_len = np.zeros((n_rows, n_cols), dtype=np.int32)
    run_len[:, 0] = mask[:, 0].astype(np.int32)
    for j in range(1, n_cols):
        run_len[:, j] = np.where(mask[:, j], run_len[:, j - 1] + 1, 0)
    return run_len.max(axis=1)


def _rolling_stat(matrix: np.ndarray, window: int, stat: str) -> np.ndarray:
    """Rolling mean/std along axis=1 via a sliding-window view (no loops)."""
    n_rows, n_cols = matrix.shape
    if n_cols < window:
        window = n_cols
    windows = np.lib.stride_tricks.sliding_window_view(matrix, window_shape=window, axis=1)
    if stat == "mean":
        return windows.mean(axis=2)
    if stat == "std":
        return windows.std(axis=2)
    raise ValueError(stat)


def compute_features(
    matrix_raw: np.ndarray,
    matrix_imputed: np.ndarray,
    missing_frac: np.ndarray,
    dates_sorted: list[dt.date],
) -> np.ndarray:
    """Returns an (n_customers, len(FEATURE_NAMES)) float32 feature matrix."""
    n_customers, n_days = matrix_imputed.shape

    mean_c = np.nanmean(matrix_raw, axis=1)
    median_c = np.nanmedian(matrix_raw, axis=1)
    std_c = np.nanstd(matrix_raw, axis=1)
    min_c = np.nanmin(matrix_raw, axis=1)
    max_c = np.nanmax(matrix_raw, axis=1)
    cv = _safe_divide(std_c, mean_c)

    non_missing = np.isfinite(matrix_raw)
    pct_zero = _safe_divide(
        np.sum((matrix_raw == 0) & non_missing, axis=1).astype(np.float64),
        non_missing.sum(axis=1).astype(np.float64),
    )

    t = np.arange(n_days, dtype=np.float64)
    t_centered = t - t.mean()
    denom = np.sum(t_centered ** 2)
    x_centered = matrix_imputed - matrix_imputed.mean(axis=1, keepdims=True)
    trend_slope = (x_centered @ t_centered) / denom

    month_ids = np.array([d.year * 12 + d.month for d in dates_sorted])
    unique_months = np.unique(month_ids)
    monthly_means = np.stack(
        [matrix_imputed[:, month_ids == m].mean(axis=1) for m in unique_months], axis=1
    )
    avg_monthly = monthly_means.mean(axis=1)
    monthly_var = monthly_means.std(axis=1)

    weekday_mask = np.array([d.weekday() < 5 for d in dates_sorted])
    weekday_mean = matrix_imputed[:, weekday_mask].mean(axis=1)
    weekend_mean = matrix_imputed[:, ~weekday_mask].mean(axis=1)
    weekday_weekend_ratio = _safe_divide(weekday_mean, weekend_mean)

    diffs = matrix_imputed[:, 1:] - matrix_imputed[:, :-1]
    pct_change = _safe_divide(diffs, matrix_imputed[:, :-1])
    sudden_drop_freq = (pct_change < -0.5).mean(axis=1)

    low_threshold = np.nanpercentile(matrix_raw, 25, axis=1, method="linear")
    low_mask = matrix_imputed < low_threshold[:, None]
    max_consecutive_low = _max_consecutive_run(low_mask).astype(np.float64)

    half = n_days // 2
    first_half_mean = matrix_imputed[:, :half].mean(axis=1)
    second_half_mean = matrix_imputed[:, half:].mean(axis=1)
    first_second_ratio = _safe_divide(first_half_mean, second_half_mean)

    roll_mean = _rolling_stat(matrix_imputed, window=7, stat="mean")
    roll_std = _rolling_stat(matrix_imputed, window=7, stat="std")
    rolling_mean_volatility = roll_mean.std(axis=1)
    rolling_std_mean = roll_std.mean(axis=1)

    season_of_month = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                        6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    season_ids = np.array([season_of_month[d.month] for d in dates_sorted])
    season_means = np.stack(
        [matrix_imputed[:, season_ids == s].mean(axis=1) for s in range(4)
         if (season_ids == s).any()], axis=1
    )
    seasonal_variation = season_means.std(axis=1)

    p10 = np.nanpercentile(matrix_raw, 10, axis=1)
    p90 = np.nanpercentile(matrix_raw, 90, axis=1)

    long_term_stability = 1.0 / (1.0 + _safe_divide(monthly_var, np.abs(avg_monthly) + 1e-6))

    features = np.stack([
        mean_c, median_c, std_c, min_c, max_c, cv,
        pct_zero, missing_frac, trend_slope,
        avg_monthly, monthly_var,
        weekday_weekend_ratio, sudden_drop_freq,
        max_consecutive_low, first_second_ratio,
        rolling_mean_volatility, rolling_std_mean, seasonal_variation,
        p10, p90, long_term_stability,
    ], axis=1).astype(np.float32)

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features
