"""Reusable Plotly chart builders for the dashboard pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve

TEMPLATE = "plotly_white"
RISK_COLORS = {"Low": "#2E8B57", "Medium": "#E8A33D", "High": "#C0392B"}


def class_distribution_chart(labels: np.ndarray) -> go.Figure:
    counts = pd.Series(labels).value_counts().sort_index()
    df = pd.DataFrame({"Class": ["Normal", "Theft"], "Count": [counts.get(0, 0), counts.get(1, 0)]})
    fig = px.bar(df, x="Class", y="Count", color="Class", text="Count",
                 color_discrete_map={"Normal": "#4C72B0", "Theft": "#DD8452"}, template=TEMPLATE)
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, title="Class Distribution")
    return fig


def consumption_distribution_chart(features_raw: np.ndarray, feature_names: list[str]) -> go.Figure:
    idx = feature_names.index("mean_consumption")
    fig = px.histogram(x=features_raw[:, idx], nbins=60, template=TEMPLATE,
                        labels={"x": "Mean Daily Consumption"})
    fig.update_layout(title="Consumption Distribution (Mean Daily Reading)")
    return fig


def risk_score_distribution_chart(scores: np.ndarray) -> go.Figure:
    fig = px.histogram(x=scores, nbins=50, template=TEMPLATE, labels={"x": "Predicted Theft Probability"})
    fig.update_layout(title="Risk Score Distribution")
    return fig


def model_comparison_bar_chart(table: dict, metric: str) -> go.Figure:
    names = list(table.keys())
    values = [table[n].get(metric) or 0 for n in names]
    fig = px.bar(x=names, y=values, template=TEMPLATE, text=[f"{v:.3f}" for v in values])
    fig.update_traces(textposition="outside")
    fig.update_layout(title=f"Model Comparison — {metric}", xaxis_title="Model", yaxis_title=metric)
    return fig


def roc_curve_chart(y_true: np.ndarray, y_score: np.ndarray, label: str = "") -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=label))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), name="Chance"))
    fig.update_layout(template=TEMPLATE, title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig


def pr_curve_chart(y_true: np.ndarray, y_score: np.ndarray, label: str = "") -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    prevalence = float(np.mean(y_true))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=label))
    fig.add_hline(y=prevalence, line_dash="dash", line_color="gray",
                  annotation_text=f"baseline prevalence={prevalence:.3f}")
    fig.update_layout(template=TEMPLATE, title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    return fig


def confusion_matrix_chart(cm: list[list[int]]) -> go.Figure:
    z = np.array(cm)
    fig = px.imshow(z, text_auto=True, color_continuous_scale="Blues",
                     x=["Pred Normal", "Pred Theft"], y=["True Normal", "True Theft"], template=TEMPLATE)
    fig.update_layout(title="Confusion Matrix", coloraxis_showscale=False)
    return fig


def precision_at_k_chart(results: dict) -> go.Figure:
    rows = []
    for name, report in results.items():
        for k_label, field in [("Top 5%", "precision_at_top_5pct"), ("Top 10%", "precision_at_top_10pct")]:
            info = report.get(field)
            if info:
                rows.append({"Model": name, "K": k_label, "Precision@K": info["precision_at_k"]})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Model", y="Precision@K", color="K", barmode="group", template=TEMPLATE)
    fig.update_layout(title="Precision@K by Model")
    return fig


def consumption_history_chart(dates: list[str], readings: list[float], customer_id: str) -> go.Figure:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "consumption": readings})
    fig = px.line(df, x="date", y="consumption", template=TEMPLATE)
    fig.update_layout(title=f"Consumption History — {customer_id}", xaxis_title="Date", yaxis_title="Reading")
    return fig


def correlation_heatmap_chart(features_raw: np.ndarray, feature_names: list[str]) -> go.Figure:
    df = pd.DataFrame(features_raw, columns=feature_names)
    corr = df.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, template=TEMPLATE,
                     x=feature_names, y=feature_names)
    fig.update_layout(title="Feature Correlation Heatmap", height=700)
    return fig


def risk_badge(risk_category: str) -> str:
    color = RISK_COLORS.get(risk_category, "#888")
    return f"<span style='background-color:{color}; color:white; padding:4px 12px; border-radius:12px; font-weight:600;'>{risk_category}</span>"
