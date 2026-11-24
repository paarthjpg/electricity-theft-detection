"""Aggregate per-model metrics JSON files into a single comparison table."""
from __future__ import annotations

import json
from pathlib import Path


def load_all_results(metrics_dir: Path) -> dict[str, dict]:
    results = {}
    for path in sorted(metrics_dir.glob("*_results.json")):
        if path.name == "baseline_results.json":
            continue  # handled separately below: it holds multiple sub-models, not one
        name = path.stem.replace("_results", "")
        with open(path) as f:
            results[name] = json.load(f)
    baseline_path = metrics_dir / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            for name, report in json.load(f).items():
                results[name] = report
    return results


def build_comparison_table(results: dict[str, dict]) -> dict[str, dict]:
    table = {}
    for name, report in results.items():
        table[name] = {
            "accuracy": report.get("accuracy"),
            "balanced_accuracy": report.get("balanced_accuracy"),
            "precision": report.get("precision"),
            "recall": report.get("recall"),
            "f1": report.get("f1"),
            "roc_auc": report.get("roc_auc"),
            "pr_auc": report.get("pr_auc"),
            "false_positive_rate": report.get("false_positive_rate"),
            "false_negative_rate": report.get("false_negative_rate"),
            "precision_at_top_5pct": (report.get("precision_at_top_5pct") or {}).get("precision_at_k"),
            "precision_at_top_10pct": (report.get("precision_at_top_10pct") or {}).get("precision_at_k"),
            "train_time_seconds": report.get("train_time_seconds"),
            "inference_time_seconds": report.get("inference_time_seconds"),
        }
    return table


def rank_models(table: dict[str, dict], metric: str = "pr_auc") -> list[tuple[str, float]]:
    ranked = sorted(((name, row.get(metric) or 0.0) for name, row in table.items()),
                     key=lambda pair: pair[1], reverse=True)
    return ranked
