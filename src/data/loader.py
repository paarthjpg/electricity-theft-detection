"""Configurable schema loader for the SGCC-style electricity consumption CSV.

The raw file has one row per customer: a customer-id column, a binary label
column, and one column per calendar date holding that day's consumption
reading. Column names/order are not assumed fixed across dataset releases,
so the schema (which columns are id/label vs. dates) is resolved here from
`configs/config.yaml` rather than hardcoded elsewhere.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSchema:
    id_col: str
    label_col: str
    date_cols_sorted: list[str]  # original column names, in chronological order
    dates_sorted: list[dt.date]


def resolve_schema(df: pd.DataFrame, config: dict) -> DatasetSchema:
    """Identify id/label/date columns and sort date columns chronologically.

    Raw date column headers (e.g. "2014/1/9") sort incorrectly as strings, so
    they are parsed to real dates and reordered before any time-series work.
    """
    schema_cfg = config["schema"]
    id_col = schema_cfg["customer_id_col"]
    label_col = schema_cfg["label_col"]
    date_fmt = schema_cfg["date_column_format"]

    if id_col not in df.columns:
        raise ValueError(f"Configured id column {id_col!r} not found in dataset columns")
    if label_col not in df.columns:
        raise ValueError(f"Configured label column {label_col!r} not found in dataset columns")

    candidate_date_cols = [c for c in df.columns if c not in (id_col, label_col)]
    parsed: list[tuple[str, dt.date]] = []
    unparseable: list[str] = []
    for col in candidate_date_cols:
        try:
            parsed.append((col, dt.datetime.strptime(col, date_fmt).date()))
        except ValueError:
            unparseable.append(col)
    if unparseable:
        raise ValueError(
            f"{len(unparseable)} columns did not match date_column_format "
            f"{date_fmt!r}: {unparseable[:5]}..."
        )

    parsed.sort(key=lambda pair: pair[1])
    date_cols_sorted = [c for c, _ in parsed]
    dates_sorted = [d for _, d in parsed]
    return DatasetSchema(id_col=id_col, label_col=label_col,
                          date_cols_sorted=date_cols_sorted, dates_sorted=dates_sorted)


def load_raw_data(csv_path: str, config: dict) -> tuple[pd.DataFrame, DatasetSchema]:
    """Load the raw CSV and return it with date columns in chronological order."""
    df = pd.read_csv(csv_path)
    schema = resolve_schema(df, config)
    ordered_cols = [schema.id_col, schema.label_col] + schema.date_cols_sorted
    df = df[ordered_cols]
    return df, schema
