"""Train and evaluate a GCN or GraphSAGE model under the leakage-safe
inductive protocol described in src/graph/graph_builder.py.

Training graph: train-train kNN edges only (built from training customers'
features only). Validation/test evaluation: training graph + one-directional
edges from training customers to the held-out split being evaluated, so
held-out customers are never visible to each other or back into the
training graph.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.graph.graph_builder import build_training_graph, build_inductive_edges, combine_graphs, randomize_edges
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.training.callbacks import EarlyStopping
from src.evaluation.metrics import full_evaluation_report, select_threshold_from_validation, compute_classification_metrics


def get_device() -> torch.device:
    # MPS is intentionally not used: several PyG scatter/aggregation kernels
    # used by GCNConv/SAGEConv are unsupported or numerically unstable on
    # Apple's MPS backend at the pinned torch/torch_geometric versions.
    # CPU is the documented default; CUDA is used automatically if present.
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_model(model_name: str, in_dim: int, config: dict) -> nn.Module:
    mcfg = config["model"]
    if model_name == "gcn":
        return GCN(in_dim, mcfg["hidden_dim_1"], mcfg["hidden_dim_2"], mcfg["dropout"])
    if model_name == "graphsage":
        return GraphSAGE(in_dim, mcfg["hidden_dim_1"], mcfg["hidden_dim_2"], mcfg["dropout"])
    raise ValueError(model_name)


def _eval_inductive(model, model_name, features_scaled, train_idx, held_out_idx, train_graph, k, metric, device):
    model.eval()
    x_train = torch.tensor(features_scaled[train_idx], dtype=torch.float32, device=device)
    x_held = torch.tensor(features_scaled[held_out_idx], dtype=torch.float32, device=device)
    held_graph = build_inductive_edges(features_scaled[held_out_idx], features_scaled[train_idx], k=k, metric=metric)
    combined = combine_graphs(train_graph, held_graph)

    x_all = torch.cat([x_train, x_held], dim=0)
    edge_index = combined.edge_index.to(device)
    edge_weight = combined.edge_weight.to(device) if model_name == "gcn" else None
    with torch.no_grad():
        logits = model(x_all, edge_index, edge_weight)
    held_logits = logits[len(train_idx):]
    return torch.sigmoid(held_logits).cpu().numpy()


def train_gnn(
    model_name: str,
    config: dict,
    processed_dir: Path,
    models_dir: Path,
    metrics_dir: Path,
    k: int | None = None,
    quick: bool = False,
    randomize: bool = False,
    tag: str | None = None,
) -> dict:
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    k = k if k is not None else config["graph"]["k"]
    metric = config["graph"]["metric"]
    tag = tag or f"{model_name}_k{k}" + ("_randomized" if randomize else "")

    features_scaled = np.load(processed_dir / "features_scaled.npy")
    labels = np.load(processed_dir / "labels.npy")
    splits = np.load(processed_dir / "splits.npz")
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]

    mode_cfg = config["quick_mode"] if quick else config["full_mode"]
    max_epochs = mode_cfg["max_epochs"]
    patience = mode_cfg["patience"]

    device = get_device()
    in_dim = features_scaled.shape[1]
    model = _build_model(model_name, in_dim, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"],
                                  weight_decay=config["model"]["weight_decay"])

    y_train = labels[train_idx]
    n_pos, n_neg = (y_train == 1).sum(), (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_graph = build_training_graph(features_scaled[train_idx], k=k, metric=metric)
    if randomize:
        train_graph = randomize_edges(train_graph, seed=seed)

    x_train = torch.tensor(features_scaled[train_idx], dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    edge_index_train = train_graph.edge_index.to(device)
    edge_weight_train = train_graph.edge_weight.to(device) if model_name == "gcn" else None

    history = {"train_loss": [], "val_pr_auc": [], "val_loss": []}
    early_stopper = EarlyStopping(patience=patience, mode="max")
    best_state = None

    t0 = time.time()
    try:
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x_train, edge_index_train, edge_weight_train)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()
            history["train_loss"].append(float(loss.item()))

            val_scores = _eval_inductive(model, model_name, features_scaled, train_idx, val_idx,
                                          train_graph, k, metric, device)
            val_metrics = compute_classification_metrics(labels[val_idx], val_scores, threshold=0.5)
            val_pr_auc = val_metrics["pr_auc"] or 0.0
            history["val_pr_auc"].append(val_pr_auc)

            improved = early_stopper.step(val_pr_auc, epoch)
            if improved:
                best_state = {k_: v_.detach().cpu().clone() for k_, v_ in model.state_dict().items()}
            if early_stopper.should_stop:
                print(f"[{tag}] early stopping at epoch {epoch} (best epoch {early_stopper.best_epoch})")
                break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                f"[{tag}] Out of memory during GNN training on device={device}. "
                "Try reducing batch size / hidden dims, or use QUICK MODE."
            ) from exc
        raise
    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"model_name": model_name, "state_dict": model.state_dict(), "in_dim": in_dim,
                "model_config": config["model"], "k": k}, models_dir / f"{tag}.pt")

    t0 = time.time()
    val_scores = _eval_inductive(model, model_name, features_scaled, train_idx, val_idx, train_graph, k, metric, device)
    threshold = select_threshold_from_validation(labels[val_idx], val_scores, strategy="f1")
    test_scores = _eval_inductive(model, model_name, features_scaled, train_idx, test_idx, train_graph, k, metric, device)
    inference_time = time.time() - t0

    report = full_evaluation_report(labels[test_idx], test_scores, threshold)
    report["threshold_default_0.5"] = full_evaluation_report(labels[test_idx], test_scores, 0.5)
    report["train_time_seconds"] = train_time
    report["inference_time_seconds"] = inference_time
    report["best_epoch"] = early_stopper.best_epoch
    report["val_pr_auc_at_best_epoch"] = early_stopper.best_score
    report["history"] = history
    report["k"] = k
    report["model_name"] = model_name
    report["protocol"] = "inductive"
    report["randomized_edges_control"] = randomize

    with open(metrics_dir / f"{tag}_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[{tag}] test PR-AUC={report['pr_auc']:.4f} F1={report['f1']:.4f} "
          f"(best val PR-AUC={early_stopper.best_score:.4f} @ epoch {early_stopper.best_epoch})")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gcn", "graphsage"], required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    with open(ROOT / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)
    processed_dir = ROOT / config["paths"]["processed_dir"]
    models_dir = ROOT / config["paths"]["models_dir"]
    metrics_dir = ROOT / config["paths"]["metrics_dir"]
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_gnn(args.model, config, processed_dir, models_dir, metrics_dir,
              k=args.k, quick=args.quick, randomize=args.randomize, tag=args.tag)


if __name__ == "__main__":
    main()
