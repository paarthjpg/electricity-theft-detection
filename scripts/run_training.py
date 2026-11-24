"""Run baseline + GNN training, then the ablation studies (Phase 7), on
whichever processed artifacts are in `data/processed` (full or quick mode).

Ablations:
  A. Random Forest without graph information         (src/training/train_baselines.py)
  B. GraphSAGE k=3
  C. GraphSAGE k=5
  D. GraphSAGE k=10
  E. GCN under the documented inductive protocol
  F. GraphSAGE with randomized edges (graph-information control)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_baselines import train_and_evaluate_baselines
from src.training.train_gnn import train_gnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    processed_dir = ROOT / (config["paths"]["processed_dir"] + ("_quick" if args.quick else ""))
    models_dir = ROOT / config["paths"]["models_dir"]
    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_baselines:
        print("=== Training baselines ===")
        train_and_evaluate_baselines(processed_dir, models_dir, metrics_dir, config)

    print("=== Training primary GCN (k=5) ===")
    train_gnn("gcn", config, processed_dir, models_dir, metrics_dir, k=5, quick=args.quick, tag="gcn_k5")

    print("=== Training primary GraphSAGE (k=5) ===")
    train_gnn("graphsage", config, processed_dir, models_dir, metrics_dir, k=5, quick=args.quick, tag="graphsage_k5")

    if not args.skip_ablations:
        print("=== Ablation B/D: GraphSAGE k=3, k=10 ===")
        train_gnn("graphsage", config, processed_dir, models_dir, metrics_dir, k=3, quick=args.quick, tag="graphsage_k3")
        train_gnn("graphsage", config, processed_dir, models_dir, metrics_dir, k=10, quick=args.quick, tag="graphsage_k10")

        print("=== Ablation F: GraphSAGE with randomized edges (control) ===")
        train_gnn("graphsage", config, processed_dir, models_dir, metrics_dir, k=5, quick=args.quick,
                   randomize=True, tag="graphsage_k5_randomized")

    print("All training + ablations complete.")


if __name__ == "__main__":
    main()
