from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui_helpers import load_config, load_processed_artifacts, load_metrics, load_predictor, get_paths, missing_artifacts_notice
from app.components.charts import class_distribution_chart, consumption_distribution_chart, risk_score_distribution_chart, model_comparison_bar_chart

config = load_config()
paths = get_paths(config)
data = load_processed_artifacts(paths["processed_dir"])

st.header("Overview")

if data is None:
    missing_artifacts_notice("Processed dataset")
    st.stop()

mode_info = data["mode_info"]
badge = "FULL MODE — real SGCC data" if not mode_info.get("quick_mode") else "QUICK MODE — stratified subset (dev/demo)"
st.caption(badge)

labels = data["labels"]
n_total = len(labels)
n_theft = int(labels.sum())
theft_prevalence = n_theft / n_total if n_total else 0.0

metrics_bundle = load_metrics(paths["metrics_dir"])
table = metrics_bundle["table"]
results = metrics_bundle["results"]

selected_tag = st.session_state.get("selected_model_tag")
selected_report = results.get(selected_tag) if selected_tag else None
if selected_tag:
    predictor = load_predictor(paths["processed_dir"], paths["models_dir"], paths["metrics_dir"], config, selected_tag)
else:
    predictor = None

held_out_idx = np.concatenate([data["splits"]["val"], data["splits"]["test"]])
held_out_scores = predictor.predict_rows_batch(held_out_idx) if predictor is not None else None
n_high_risk = int((held_out_scores >= config["risk_thresholds"]["medium_max"]).sum()) if held_out_scores is not None else None

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Consumers Analyzed", f"{n_total:,}")
col2.metric("Labeled Theft Consumers", f"{n_theft:,}")
col3.metric("Theft Prevalence", f"{theft_prevalence:.2%}")
col4.metric("High-Risk Predictions (held-out)", f"{n_high_risk:,}" if n_high_risk is not None else "N/A")

col5, col6, col7 = st.columns(3)
col5.metric("Selected Model", selected_tag or "N/A")
col6.metric("Model PR-AUC (test)", f"{selected_report['pr_auc']:.4f}" if selected_report and selected_report.get("pr_auc") else "N/A")
col7.metric("Model F1-score (test)", f"{selected_report['f1']:.4f}" if selected_report else "N/A")

st.markdown("---")
st.subheader("Dataset Information")
st.markdown(
    f"""
- **Source:** SGCC Electricity Theft Detection Dataset (real data; see `data/raw/DATA_SOURCE.md` for citation and checksum), unless QUICK MODE subset is selected above
- **Customers in this view:** {n_total:,}
- **Features engineered per customer:** {len(data['feature_names'])}
- **Date range:** {data['dates'][0]} to {data['dates'][-1]}
"""
)

st.subheader("Charts")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(class_distribution_chart(labels), use_container_width=True)
with c2:
    st.plotly_chart(consumption_distribution_chart(data["features_raw"], data["feature_names"]), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    if table:
        st.plotly_chart(model_comparison_bar_chart(table, "pr_auc"), use_container_width=True)
    else:
        missing_artifacts_notice("Model comparison results")
with c4:
    if held_out_scores is not None:
        st.plotly_chart(risk_score_distribution_chart(held_out_scores), use_container_width=True)
    else:
        missing_artifacts_notice("Trained model checkpoint")
