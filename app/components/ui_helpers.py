"""Shared loading/caching helpers for the dashboard. Loads saved artifacts
only — never retrains on page refresh."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@st.cache_data
def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def _processed_dir(config: dict) -> Path:
    mode = st.session_state.get("data_mode", "full")
    suffix = "_quick" if mode == "quick" else ""
    return ROOT / (config["paths"]["processed_dir"] + suffix)


@st.cache_data
def load_processed_artifacts(processed_dir_str: str) -> dict | None:
    processed_dir = Path(processed_dir_str)
    required = ["features_scaled.npy", "features_raw.npy", "labels.npy",
                "customer_ids.npy", "splits.npz", "feature_names.json", "dates.json"]
    if not processed_dir.exists() or not all((processed_dir / r).exists() for r in required):
        return None

    data = {
        "features_scaled": np.load(processed_dir / "features_scaled.npy"),
        "features_raw": np.load(processed_dir / "features_raw.npy"),
        "labels": np.load(processed_dir / "labels.npy"),
        "customer_ids": np.load(processed_dir / "customer_ids.npy", allow_pickle=True),
        "reading_matrix_raw": np.load(processed_dir / "reading_matrix_raw.npy"),
    }
    splits = np.load(processed_dir / "splits.npz")
    data["splits"] = {k: splits[k] for k in splits.files}
    with open(processed_dir / "feature_names.json") as f:
        data["feature_names"] = json.load(f)
    with open(processed_dir / "dates.json") as f:
        data["dates"] = json.load(f)
    mode_path = processed_dir / "mode.json"
    data["mode_info"] = json.load(open(mode_path)) if mode_path.exists() else {}
    validation_path = processed_dir / "validation_report.json"
    data["validation_report"] = json.load(open(validation_path)) if validation_path.exists() else {}
    return data


@st.cache_data
def load_metrics(metrics_dir_str: str) -> dict:
    sys.path.insert(0, str(ROOT))
    from src.evaluation.evaluate import load_all_results, build_comparison_table
    metrics_dir = Path(metrics_dir_str)
    if not metrics_dir.exists():
        return {"results": {}, "table": {}}
    results = load_all_results(metrics_dir)
    table = build_comparison_table(results) if results else {}
    return {"results": results, "table": table}


@st.cache_resource
def load_predictor(processed_dir_str: str, models_dir_str: str, metrics_dir_str: str,
                    _config: dict, model_tag: str):
    from src.inference.predictor import TheftRiskPredictor
    checkpoint_path = Path(models_dir_str) / f"{model_tag}.joblib"
    if not checkpoint_path.exists():
        return None
    return TheftRiskPredictor(Path(processed_dir_str), Path(models_dir_str), Path(metrics_dir_str),
                               _config, model_tag=model_tag)


def get_paths(config: dict) -> dict:
    return {
        "processed_dir": str(_processed_dir(config)),
        "models_dir": str(ROOT / config["paths"]["models_dir"]),
        "metrics_dir": str(ROOT / config["paths"]["metrics_dir"]),
        "figures_dir": str(ROOT / config["paths"]["figures_dir"]),
    }


def missing_artifacts_notice(what: str):
    st.warning(
        f"{what} not found. Run the corresponding script "
        f"(`scripts/prepare_data.py`, `scripts/run_training.py`, or `scripts/run_evaluation.py`) first. "
        "No fabricated data is shown here."
    )
