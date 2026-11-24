from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui_helpers import load_config, load_metrics, load_processed_artifacts, load_predictor, get_paths, missing_artifacts_notice
from app.components.charts import (
    model_comparison_bar_chart, roc_curve_chart, pr_curve_chart,
    confusion_matrix_chart, precision_at_k_chart, training_curve_chart,
)

config = load_config()
paths = get_paths(config)
data = load_processed_artifacts(paths["processed_dir"])
metrics_bundle = load_metrics(paths["metrics_dir"])
table, results = metrics_bundle["table"], metrics_bundle["results"]

st.header("Model Performance")

if not table:
    missing_artifacts_notice("Model evaluation results")
    st.stop()

st.subheader("Why PR-AUC and Recall Matter Here")
st.markdown(
    "Theft is rare (roughly 8-10% of customers in this dataset), so a model predicting "
    "'normal' for everyone would score ~90% accuracy while catching zero theft cases. "
    "**PR-AUC** (area under the precision-recall curve) and **recall** are far more informative "
    "under this imbalance than accuracy or even ROC-AUC, because they are sensitive to how well "
    "the model ranks the rare positive class rather than being dominated by the large negative class. "
    "**Precision@K** matters operationally: utilities can only physically inspect a limited number "
    "of flagged customers, so precision among the top-K highest-risk customers tells us how many "
    "inspections would actually find theft."
)

st.subheader("Model Comparison Table")
comparison_df = pd.DataFrame(table).T
st.dataframe(comparison_df, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(model_comparison_bar_chart(table, "pr_auc"), use_container_width=True)
with c2:
    st.plotly_chart(model_comparison_bar_chart(table, "f1"), use_container_width=True)

st.plotly_chart(precision_at_k_chart(results), use_container_width=True)

st.subheader("ROC / Precision-Recall Curves")
model_for_curves = st.selectbox("Model", list(results.keys()))
if data is not None:
    labels = data["labels"]
    splits = data["splits"]
    test_idx = splits["test"]
    predictor = load_predictor(paths["processed_dir"], paths["models_dir"], paths["metrics_dir"], config, model_for_curves) \
        if (Path(paths["models_dir"]) / f"{model_for_curves}.pt").exists() else None
    if predictor is not None:
        test_scores = predictor.predict_rows_batch(test_idx)
        test_labels = labels[test_idx]
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(roc_curve_chart(test_labels, test_scores, model_for_curves), use_container_width=True)
        with cc2:
            st.plotly_chart(pr_curve_chart(test_labels, test_scores, model_for_curves), use_container_width=True)
    else:
        st.caption("ROC/PR curves require a GNN checkpoint (baseline sklearn models store the test score "
                   "distribution in their results JSON instead; see the confusion matrix below).")

st.subheader("Confusion Matrix")
cm_report = results.get(model_for_curves, {})
if cm_report.get("confusion_matrix"):
    st.plotly_chart(confusion_matrix_chart(cm_report["confusion_matrix"]), use_container_width=True)

st.subheader("Training History")
if "history" in cm_report:
    st.plotly_chart(training_curve_chart(cm_report["history"]), use_container_width=True)
else:
    st.caption(f"'{model_for_curves}' has no training-curve history (not a GNN, or history was not recorded).")
