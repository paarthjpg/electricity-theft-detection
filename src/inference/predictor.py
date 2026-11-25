"""Loads saved baseline-model artifacts (Logistic Regression, Random Forest,
XGBoost) and serves theft-risk predictions for individual customers or
batches. Predictions are read from saved checkpoints only — nothing here
retrains on a dashboard refresh."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np


class TheftRiskPredictor:
    def __init__(self, processed_dir: Path, models_dir: Path, metrics_dir: Path, config: dict,
                 model_tag: str = "xgboost"):
        self.config = config
        self.processed_dir = Path(processed_dir)
        self.model_tag = model_tag

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

        self.model = joblib.load(Path(models_dir) / f"{model_tag}.joblib")

        self.threshold = 0.5
        results_path = Path(metrics_dir) / "baseline_results.json"
        if results_path.exists():
            with open(results_path) as f:
                baseline_results = json.load(f)
            self.threshold = baseline_results.get(model_tag, {}).get("threshold", 0.5)

    def risk_category(self, probability: float) -> str:
        thresholds = self.config["risk_thresholds"]
        if probability <= thresholds["low_max"]:
            return "Low"
        if probability <= thresholds["medium_max"]:
            return "Medium"
        return "High"

    def predict_row(self, row: int) -> dict:
        probability = float(self.model.predict_proba(self.features_scaled[row:row + 1])[0, 1])
        predicted_class = int(probability >= self.threshold)
        return {
            "customer_id": self.customer_ids[row],
            "split": self.row_to_split.get(row, "train"),
            "ground_truth_label": int(self.labels[row]),
            "theft_probability": probability,
            "predicted_class": predicted_class,
            "risk_category": self.risk_category(probability),
            "classification_threshold": self.threshold,
        }

    def predict_rows_batch(self, rows: np.ndarray) -> np.ndarray:
        """Vectorized probability prediction for many rows at once."""
        rows = np.asarray(rows)
        return self.model.predict_proba(self.features_scaled[rows])[:, 1]

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
