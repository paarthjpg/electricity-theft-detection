"""End-to-end data preparation: load -> validate -> clean -> impute ->
split -> engineer features -> normalize -> save artifacts.

Usage:
    python scripts/prepare_data.py            # full dataset
    python scripts/prepare_data.py --quick     # small stratified subset (dev/demo)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_raw_data
from src.data.validation import validate_raw_dataset, format_report
from src.data.preprocessing import clean_dataset, impute_missing, split_customers
from src.data.feature_engineering import compute_features, FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use a small stratified subset (QUICK MODE)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    seed = config["seed"]
    np.random.seed(seed)

    raw_path = ROOT / config["paths"]["raw_data"]
    processed_dir = ROOT / config["paths"]["processed_dir"]
    models_dir = ROOT / config["paths"]["models_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw data from {raw_path} ...")
    df, schema = load_raw_data(str(raw_path), config)
    print(f"Loaded {len(df)} customers x {len(schema.date_cols_sorted)} dates "
          f"({schema.dates_sorted[0]} to {schema.dates_sorted[-1]})")

    print("Validating raw dataset ...")
    report = validate_raw_dataset(df, schema)
    print(format_report(report))

    if args.quick:
        subset_size = min(config["quick_mode"]["subset_size"], len(df))
        pos_frac = df[schema.label_col].mean()
        n_pos = max(1, int(round(subset_size * pos_frac)))
        n_neg = subset_size - n_pos
        pos_idx = df.index[df[schema.label_col] == 1].to_numpy()
        neg_idx = df.index[df[schema.label_col] == 0].to_numpy()
        rng = np.random.default_rng(seed)
        chosen = np.concatenate([
            rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False),
            rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False),
        ])
        df = df.loc[sorted(chosen)].reset_index(drop=True)
        print(f"QUICK MODE: subsampled to {len(df)} customers "
              f"(theft rate preserved at {df[schema.label_col].mean():.3f})")

    print("Cleaning dataset ...")
    ids_labels, matrix_raw, clean_report = clean_dataset(df, schema, config)
    print(clean_report)

    print("Imputing missing values (per-customer median) ...")
    matrix_imputed, missing_frac = impute_missing(matrix_raw)

    print("Splitting customers (stratified, seed={}) ...".format(seed))
    splits = split_customers(ids_labels, schema, config)
    for name, idx in splits.items():
        rate = ids_labels[schema.label_col].to_numpy()[idx].mean()
        print(f"  {name}: {len(idx)} customers, theft rate={rate:.4f}")

    print("Engineering features ...")
    features_raw = compute_features(matrix_raw, matrix_imputed, missing_frac, schema.dates_sorted)
    assert np.isfinite(features_raw).all(), "Non-finite values in engineered features"

    print("Fitting scaler on TRAINING customers only ...")
    scaler = StandardScaler()
    scaler.fit(features_raw[splits["train"]])
    features_scaled = scaler.transform(features_raw).astype(np.float32)
    joblib.dump(scaler, models_dir / "feature_scaler.joblib")

    print("Saving processed artifacts ...")
    np.save(processed_dir / "reading_matrix_raw.npy", matrix_raw)
    np.save(processed_dir / "reading_matrix_imputed.npy", matrix_imputed)
    np.save(processed_dir / "features_raw.npy", features_raw)
    np.save(processed_dir / "features_scaled.npy", features_scaled)
    np.save(processed_dir / "labels.npy", ids_labels[schema.label_col].to_numpy())
    np.save(processed_dir / "customer_ids.npy", ids_labels[schema.id_col].to_numpy())
    np.savez(processed_dir / "splits.npz", **splits)

    with open(processed_dir / "feature_names.json", "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    with open(processed_dir / "dates.json", "w") as f:
        json.dump([d.isoformat() for d in schema.dates_sorted], f)
    with open(processed_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(processed_dir / "cleaning_report.json", "w") as f:
        json.dump(vars(clean_report), f, indent=2)
    with open(processed_dir / "mode.json", "w") as f:
        json.dump({"quick_mode": args.quick, "n_customers": len(ids_labels)}, f, indent=2)

    print(f"Done. Processed {len(ids_labels)} customers, {features_raw.shape[1]} features.")


if __name__ == "__main__":
    main()
