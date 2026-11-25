"""Run baseline model training (Logistic Regression, Random Forest, XGBoost)
on whichever processed artifacts are in `data/processed` (full or quick mode).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_baselines import train_and_evaluate_baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    processed_dir = ROOT / (config["paths"]["processed_dir"] + ("_quick" if args.quick else ""))
    models_dir = ROOT / config["paths"]["models_dir"]
    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training baselines (Logistic Regression, Random Forest, XGBoost) ===")
    train_and_evaluate_baselines(processed_dir, models_dir, metrics_dir, config)

    print("Training complete.")


if __name__ == "__main__":
    main()
