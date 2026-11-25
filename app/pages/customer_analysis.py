from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui_helpers import load_config, load_processed_artifacts, load_predictor, get_paths, missing_artifacts_notice
from app.components.charts import consumption_history_chart, risk_badge

config = load_config()
paths = get_paths(config)
data = load_processed_artifacts(paths["processed_dir"])

st.header("Customer Risk Analysis")

if data is None:
    missing_artifacts_notice("Processed dataset")
    st.stop()

selected_tag = st.session_state.get("selected_model_tag")
if not selected_tag:
    missing_artifacts_notice("A trained model checkpoint")
    st.stop()

predictor = load_predictor(paths["processed_dir"], paths["models_dir"], paths["metrics_dir"], config, selected_tag)
if predictor is None:
    missing_artifacts_notice(f"Checkpoint for model '{selected_tag}'")
    st.stop()

customer_ids = data["customer_ids"]
customer_id = st.selectbox("Select a Customer ID", customer_ids, index=0)

try:
    result = predictor.predict_by_customer_id(customer_id)
except Exception as exc:
    st.error("Could not compute a prediction for this customer. Please try another customer or model.")
    st.stop()

if result is None:
    st.error("Customer not found in the processed dataset.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Customer ID", str(result["customer_id"]))
col2.metric("Ground-Truth Label", "Theft" if result["ground_truth_label"] == 1 else "Normal")
col3.metric("Predicted Theft Probability", f"{result['theft_probability']:.3f}")
col4.metric("Predicted Class", "Theft" if result["predicted_class"] == 1 else "Normal")

st.markdown(f"**Risk Category:** {risk_badge(result['risk_category'])}", unsafe_allow_html=True)
st.caption(
    f"Split: {result['split']} · Classification threshold (validation-optimized F1): "
    f"{result['classification_threshold']:.3f} · Risk category thresholds: "
    f"Low ≤ {config['risk_thresholds']['low_max']}, Medium ≤ {config['risk_thresholds']['medium_max']}, High above that."
)
st.warning("A high risk score is an indicator for investigation, not proof of electricity theft.")

st.markdown("---")
st.subheader("Historical Electricity Consumption")
series = predictor.consumption_series(customer_id)
st.plotly_chart(consumption_history_chart(series["dates"], series["readings"], customer_id), use_container_width=True)

st.subheader("Consumption Statistics")
row = predictor.id_to_row[customer_id]
feat_row = data["features_raw"][row]
stats_df = pd.DataFrame({"Feature": data["feature_names"], "Value": feat_row})
st.dataframe(stats_df, use_container_width=True, hide_index=True)

readings = np.array(series["readings"], dtype=float)
valid = readings[~np.isnan(readings)]
if len(valid) > 1:
    diffs = np.diff(valid)
    pct_changes = np.divide(diffs, valid[:-1], out=np.zeros_like(diffs), where=valid[:-1] != 0)
    n_drops = int((pct_changes < -0.5).sum())
    st.caption(f"Unusual consumption changes: {n_drops} day-to-day drops greater than 50% detected in the raw series.")

st.subheader("Comparison Against Similar Consumers")
train_rows = predictor.splits["train"]
train_feats = data["features_scaled"][train_rows]
from sklearn.neighbors import NearestNeighbors
nn_model = NearestNeighbors(n_neighbors=min(5, len(train_rows)), metric="cosine").fit(train_feats)
_, neighbor_idx = nn_model.kneighbors(data["features_scaled"][row:row + 1])
neighbor_rows = train_rows[neighbor_idx[0]]
neighbor_df = pd.DataFrame({
    "Customer ID": data["customer_ids"][neighbor_rows],
    "Ground Truth": np.where(data["labels"][neighbor_rows] == 1, "Theft", "Normal"),
    "Mean Consumption": data["features_raw"][neighbor_rows, data["feature_names"].index("mean_consumption")],
})
st.dataframe(neighbor_df, use_container_width=True, hide_index=True)
st.caption("Nearest training-set neighbors in engineered-feature space (cosine similarity) — for interpretability only; "
           "no graph structure is used by the model.")
