"""Static matplotlib figures for the written report (artifacts/figures/).
The dashboard uses its own interactive Plotly charts (app/components/charts.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve

sns.set_style("whitegrid")


def plot_roc_curves(results_by_model: dict[str, dict], y_true_by_model: dict[str, np.ndarray],
                     y_score_by_model: dict[str, np.ndarray], out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in results_by_model:
        fpr, tpr, _ = roc_curve(y_true_by_model[name], y_score_by_model[name])
        auc = results_by_model[name].get("roc_auc")
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})" if auc else name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(results_by_model: dict[str, dict], y_true_by_model: dict[str, np.ndarray],
                    y_score_by_model: dict[str, np.ndarray], out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for name in results_by_model:
        precision, recall, _ = precision_recall_curve(y_true_by_model[name], y_score_by_model[name])
        pr_auc = results_by_model[name].get("pr_auc")
        ax.plot(recall, precision, label=f"{name} (PR-AUC={pr_auc:.3f})" if pr_auc else name)
    prevalence = np.mean(list(y_true_by_model.values())[0])
    ax.axhline(prevalence, color="k", linestyle="--", alpha=0.4, label=f"baseline prevalence={prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], model_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Pred Normal", "Pred Theft"], yticklabels=["True Normal", "True Theft"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison_bars(comparison: dict[str, dict], metric: str, out_path: Path):
    names = list(comparison.keys())
    values = [comparison[n].get(metric) or 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=sns.color_palette("viridis", len(names)))
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison — {metric}")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(labels: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = np.bincount(labels.astype(int))
    ax.bar(["Normal (0)", "Theft (1)"], counts, color=["#4C72B0", "#DD8452"])
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom")
    ax.set_title("Class Distribution")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_precision_recall_at_k(k_results: dict[str, dict], out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(k_results.keys())
    p5 = [k_results[n]["precision_at_top_5pct"]["precision_at_k"] for n in names]
    p10 = [k_results[n]["precision_at_top_10pct"]["precision_at_k"] for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, p5, width, label="Top 5%")
    ax.bar(x + width / 2, p10, width, label="Top 10%")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Precision@K")
    ax.set_title("Precision@K by Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
