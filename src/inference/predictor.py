"""Loads saved artifacts and serves theft-risk predictions for individual
customers or batches, respecting the inductive leakage-safe protocol:
a validation/test customer is always scored by connecting it only to
TRAINING reference customers, never to other held-out customers.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from src.graph.graph_builder import build_training_graph, build_inductive_edges, combine_graphs
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE


def _build_model(model_name: str, in_dim: int, model_config: dict):
    if model_name == "gcn":
        return GCN(in_dim, model_config["hidden_dim_1"], model_config["hidden_dim_2"], model_config["dropout"])
    if model_name == "graphsage":
        return GraphSAGE(in_dim, model_config["hidden_dim_1"], model_config["hidden_dim_2"], model_config["dropout"])
    raise ValueError(model_name)


class TheftRiskPredictor:
    def __init__(self, processed_dir: Path, models_dir: Path, metrics_dir: Path, config: dict,
                 model_tag: str = "graphsage_k5"):
        self.config = config
        self.processed_dir = Path(processed_dir)
        self.metric = config["graph"]["metric"]

        self.customer_ids = np.load(self.processed_dir / "customer_ids.npy", allow_pickle=True)
        self.labels = np.load(self.processed_dir / "labels.npy")
        self.features_scaled = np.load(self.processed_dir / "features_scaled.npy")
        self.features_raw = np.load(self.processed_dir / "features_raw.npy")
        self.reading_matrix_raw = np.load(self.processed_dir / "reading_matrix_raw.npy")
        with open(self.processed_dir / "dates.json") as f:
            self.dates = json.load(f)
        with open(self.processed_dir / "feature_names.json") as f:
            self.feature_names = json.load(f)
        splits = np.load(self.processed_dir / "splits.npz")
        self.splits = {k: splits[k] for k in splits.files}

        self.id_to_row = {cid: i for i, cid in enumerate(self.customer_ids)}
        self.row_to_split = {}
        for split_name, idx in self.splits.items():
            for i in idx:
                self.row_to_split[int(i)] = split_name

        checkpoint = torch.load(models_dir / f"{model_tag}.pt", map_location="cpu", weights_only=False)
        self.model_name = checkpoint["model_name"]
        self.k = checkpoint["k"]
        self.model = _build_model(self.model_name, checkpoint["in_dim"], checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        results_path = metrics_dir / f"{model_tag}_results.json"
        self.threshold = 0.5
        if results_path.exists():
            with open(results_path) as f:
                self.threshold = json.load(f).get("threshold", 0.5)

        self._train_graph = build_training_graph(
            self.features_scaled[self.splits["train"]], k=self.k, metric=self.metric
        )
        self._train_logits_cache = None

    def _train_logits(self) -> torch.Tensor:
        if self._train_logits_cache is None:
            x_train = torch.tensor(self.features_scaled[self.splits["train"]], dtype=torch.float32)
            edge_weight = self._train_graph.edge_weight if self.model_name == "gcn" else None
            with torch.no_grad():
                self._train_logits_cache = self.model(x_train, self._train_graph.edge_index, edge_weight)
        return self._train_logits_cache

    def risk_category(self, probability: float) -> str:
        thresholds = self.config["risk_thresholds"]
        if probability <= thresholds["low_max"]:
            return "Low"
        if probability <= thresholds["medium_max"]:
            return "Medium"
        return "High"

    def predict_row(self, row: int) -> dict:
        split = self.row_to_split.get(row, "train")
        if split == "train":
            train_rows = self.splits["train"]
            local_idx = int(np.where(train_rows == row)[0][0])
            logits = self._train_logits()
            logit = logits[local_idx].item()
        else:
            query_feat = self.features_scaled[row:row + 1]
            train_feats = self.features_scaled[self.splits["train"]]
            held_graph = build_inductive_edges(query_feat, train_feats, k=self.k, metric=self.metric)
            combined = combine_graphs(self._train_graph, held_graph)
            x_all = torch.cat([
                torch.tensor(train_feats, dtype=torch.float32),
                torch.tensor(query_feat, dtype=torch.float32),
            ], dim=0)
            edge_weight = combined.edge_weight if self.model_name == "gcn" else None
            with torch.no_grad():
                logits = self.model(x_all, combined.edge_index, edge_weight)
            logit = logits[-1].item()

        probability = float(torch.sigmoid(torch.tensor(logit)).item())
        predicted_class = int(probability >= self.threshold)
        return {
            "customer_id": self.customer_ids[row],
            "split": split,
            "ground_truth_label": int(self.labels[row]),
            "theft_probability": probability,
            "predicted_class": predicted_class,
            "risk_category": self.risk_category(probability),
            "classification_threshold": self.threshold,
        }

    def predict_rows_batch(self, rows: np.ndarray) -> np.ndarray:
        """Vectorized probability prediction for many rows at once — one kNN
        fit and one forward pass for the whole batch, instead of one each
        per customer. Mixed train/held-out rows are handled correctly but
        held-out rows are batched together for efficiency.
        """
        rows = np.asarray(rows)
        probs = np.zeros(len(rows), dtype=np.float64)

        train_rows = self.splits["train"]
        is_train = np.array([self.row_to_split.get(int(r), "train") == "train" for r in rows])

        if is_train.any():
            train_row_to_local = {int(r): i for i, r in enumerate(train_rows)}
            train_logits = self._train_logits()
            for i in np.where(is_train)[0]:
                local_idx = train_row_to_local[int(rows[i])]
                probs[i] = torch.sigmoid(train_logits[local_idx]).item()

        held_mask = ~is_train
        if held_mask.any():
            held_rows = rows[held_mask]
            query_feat = self.features_scaled[held_rows]
            train_feats = self.features_scaled[train_rows]
            held_graph = build_inductive_edges(query_feat, train_feats, k=self.k, metric=self.metric)
            combined = combine_graphs(self._train_graph, held_graph)
            x_all = torch.cat([
                torch.tensor(train_feats, dtype=torch.float32),
                torch.tensor(query_feat, dtype=torch.float32),
            ], dim=0)
            edge_weight = combined.edge_weight if self.model_name == "gcn" else None
            with torch.no_grad():
                logits = self.model(x_all, combined.edge_index, edge_weight)
            held_logits = logits[len(train_rows):]
            probs[held_mask] = torch.sigmoid(held_logits).numpy()

        return probs

    def predict_by_customer_id(self, customer_id: str) -> dict | None:
        row = self.id_to_row.get(customer_id)
        if row is None:
            return None
        return self.predict_row(row)

    def consumption_series(self, customer_id: str) -> dict | None:
        row = self.id_to_row.get(customer_id)
        if row is None:
            return None
        return {"dates": self.dates, "readings": self.reading_matrix_raw[row].tolist()}
