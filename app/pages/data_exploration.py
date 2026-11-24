from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.ui_helpers import load_config, load_processed_artifacts, get_paths, missing_artifacts_notice
from app.components.charts import correlation_heatmap_chart

config = load_config()
paths = get_paths(config)
data = load_processed_artifacts(paths["processed_dir"])

st.header("Data Exploration")

if data is None:
    missing_artifacts_notice("Processed dataset")
    st.stop()

feature_names = data["feature_names"]
features_raw = data["features_raw"]
labels = data["labels"]
customer_ids = data["customer_ids"]

st.subheader("Dataset Preview")
preview_df = pd.DataFrame(features_raw, columns=feature_names)
preview_df.insert(0, "CONS_NO", customer_ids)
preview_df.insert(1, "FLAG", labels)
st.dataframe(preview_df.head(50), use_container_width=True)

st.subheader("Missing Data Analysis")
report = data["validation_report"]
if report:
    c1, c2, c3 = st.columns(3)
    c1.metric("% Missing Readings", f"{report.get('pct_missing_readings', 0):.2%}")
    c2.metric("Negative Readings (pre-clean)", f"{report.get('n_negative_readings', 0):,}")
    c3.metric("Extreme Readings (z>6, pre-clean)", f"{report.get('n_extreme_readings_z6', 0):,}")
else:
    st.caption("Validation report not found for this dataset mode.")

st.subheader("Normal vs. Theft Consumer Comparison")
compare_feature = st.selectbox("Feature to compare", feature_names, index=feature_names.index("mean_consumption"))
df_cmp = pd.DataFrame({compare_feature: features_raw[:, feature_names.index(compare_feature)],
                       "Class": np.where(labels == 1, "Theft", "Normal")})
import plotly.express as px
fig = px.box(df_cmp, x="Class", y=compare_feature, color="Class", template="plotly_white",
             color_discrete_map={"Normal": "#4C72B0", "Theft": "#DD8452"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Monthly Consumption Trend (sample of customers)")
dates = pd.to_datetime(data["dates"])
rng = np.random.default_rng(0)
sample_rows = rng.choice(len(customer_ids), size=min(25, len(customer_ids)), replace=False)
monthly_df = pd.DataFrame(data["reading_matrix_raw"][sample_rows].T, index=dates)
monthly_mean = monthly_df.resample("MS").mean().mean(axis=1)
fig2 = px.line(x=monthly_mean.index, y=monthly_mean.values, template="plotly_white",
               labels={"x": "Month", "y": "Mean Consumption (sampled customers)"})
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Feature Correlation Heatmap")
st.plotly_chart(correlation_heatmap_chart(features_raw, feature_names), use_container_width=True)

st.subheader("Outlier Analysis")
outlier_feature = st.selectbox("Feature for outlier view", feature_names, index=feature_names.index("max_consumption"), key="outlier_feat")
vals = features_raw[:, feature_names.index(outlier_feature)]
q1, q3 = np.percentile(vals, [25, 75])
iqr = q3 - q1
n_outliers = int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum())
st.metric(f"IQR Outliers in '{outlier_feature}'", f"{n_outliers:,} ({n_outliers / len(vals):.2%})")
fig3 = px.histogram(x=vals, nbins=60, template="plotly_white", labels={"x": outlier_feature})
st.plotly_chart(fig3, use_container_width=True)
