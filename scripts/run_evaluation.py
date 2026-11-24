"""Aggregate all trained-model results into a comparison table, generate
report figures, and write RESULTS.md from the actual evaluation outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import load_all_results, build_comparison_table, rank_models
from src.evaluation.plots import (
    plot_model_comparison_bars, plot_class_distribution,
    plot_precision_recall_at_k, plot_training_curves,
)


def main():
    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    figures_dir = ROOT / config["paths"]["figures_dir"]
    processed_dir = ROOT / config["paths"]["processed_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(metrics_dir)
    if not results:
        print("No results found in artifacts/metrics. Run training first.")
        return

    table = build_comparison_table(results)
    with open(metrics_dir / "model_comparison.json", "w") as f:
        json.dump(table, f, indent=2, default=str)

    ranked = rank_models(table, metric="pr_auc")
    print("Model ranking by PR-AUC (test set):")
    for name, score in ranked:
        print(f"  {name}: {score:.4f}")

    plot_model_comparison_bars(table, "pr_auc", figures_dir / "model_comparison_pr_auc.png")
    plot_model_comparison_bars(table, "f1", figures_dir / "model_comparison_f1.png")
    plot_precision_recall_at_k(results, figures_dir / "precision_at_k.png")

    labels_path = processed_dir / "labels.npy"
    if labels_path.exists():
        plot_class_distribution(np.load(labels_path), figures_dir / "class_distribution.png")

    for name, report in results.items():
        if "history" in report:
            plot_training_curves(report["history"], name, figures_dir / f"training_curves_{name}.png")

    generate_results_md(results, table, ranked, config)
    print(f"Figures saved to {figures_dir}")


def generate_results_md(results: dict, table: dict, ranked: list, config: dict):
    processed_dir = ROOT / config["paths"]["processed_dir"]
    mode_path = processed_dir / "mode.json"
    mode_info = json.load(open(mode_path)) if mode_path.exists() else {"quick_mode": "unknown", "n_customers": "unknown"}

    lines = [
        "# Results",
        "",
        f"Generated from actual evaluation outputs in `artifacts/metrics/`. "
        f"Dataset mode: {'QUICK MODE (subset)' if mode_info.get('quick_mode') else 'FULL MODE (complete dataset)'}, "
        f"n_customers={mode_info.get('n_customers')}.",
        "",
        "## Model Comparison (test set)",
        "",
        "| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC | P@5% | P@10% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, row in table.items():
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
        lines.append(
            f"| {name} | {fmt(row['accuracy'])} | {fmt(row['balanced_accuracy'])} | "
            f"{fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} | "
            f"{fmt(row['roc_auc'])} | {fmt(row['pr_auc'])} | "
            f"{fmt(row['precision_at_top_5pct'])} | {fmt(row['precision_at_top_10pct'])} |"
        )

    lines += ["", "## Ranking by PR-AUC (primary metric for imbalanced theft detection)", ""]
    for i, (name, score) in enumerate(ranked, 1):
        lines.append(f"{i}. **{name}**: {score:.4f}")

    best_baseline = max(
        ((n, r.get("pr_auc") or 0) for n, r in results.items() if n in ("logistic_regression", "random_forest", "xgboost")),
        key=lambda p: p[1], default=(None, 0),
    )
    best_gnn = max(
        ((n, r.get("pr_auc") or 0) for n, r in results.items() if n.startswith("gcn") or n.startswith("graphsage")),
        key=lambda p: p[1], default=(None, 0),
    )
    lines += ["", "## Honest Comparison", ""]
    if best_baseline[0] and best_gnn[0]:
        if best_baseline[1] >= best_gnn[1]:
            lines.append(
                f"The best conventional baseline (**{best_baseline[0]}**, PR-AUC={best_baseline[1]:.4f}) "
                f"matched or outperformed the best GNN (**{best_gnn[0]}**, PR-AUC={best_gnn[1]:.4f}) on this "
                "dataset and split. This is reported as-is rather than adjusted: possible causes include weak "
                "homophily in the k-NN similarity graph relative to the engineered summary-statistic features, "
                "class imbalance limiting the GNN's effective supervision signal, oversmoothing across only two "
                "message-passing layers being insufficient to exploit relational structure, or the engineered "
                "features already capturing most of the discriminative signal present in the raw consumption series."
            )
        else:
            lines.append(
                f"The best GNN (**{best_gnn[0]}**, PR-AUC={best_gnn[1]:.4f}) outperformed the best conventional "
                f"baseline (**{best_baseline[0]}**, PR-AUC={best_baseline[1]:.4f}), suggesting the customer "
                "similarity graph adds information beyond each customer's own engineered features."
            )

    randomized = results.get("graphsage_k5_randomized")
    non_randomized = results.get("graphsage_k5")
    if randomized and non_randomized:
        lines.append("")
        lines.append(
            f"Randomized-edge control: GraphSAGE with the real similarity graph scored "
            f"PR-AUC={non_randomized.get('pr_auc'):.4f} vs. PR-AUC={randomized.get('pr_auc'):.4f} with edges "
            "randomly shuffled (same degree sequence, same edge count, no label information used). "
            + ("The real graph's specific structure provides a measurable benefit over an arbitrary graph of "
               "the same size." if non_randomized.get('pr_auc', 0) > randomized.get('pr_auc', 0) else
               "The real graph's specific structure did NOT clearly outperform an arbitrary graph of the same "
               "size, suggesting the similarity edges carry limited homophilous signal for this task.")
        )

    with open(ROOT / "RESULTS.md", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
