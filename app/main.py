"""Electricity Theft Detection — analytics dashboard entry point.

Defines shared chrome (page config, CSS, sidebar dataset/model selectors)
and explicit page navigation order via st.navigation, so page files keep
the exact names required by the project structure regardless of
alphabetical sort order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Electricity Theft Detection — Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2rem; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.app-badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 0.75rem; font-weight: 600; margin-left: 8px;
}
.app-badge-real { background-color: #DCFCE7; color: #166534; }
.app-badge-quick { background-color: #FEF3C7; color: #92400E; }
.risk-note { color: #6B7280; font-size: 0.85rem; }
hr { margin: 0.6rem 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

from app.components.ui_helpers import load_config, get_paths

config = load_config()

with st.sidebar:
    st.markdown("### ⚡ Theft Detection")
    st.caption("Machine Learning for Electricity Theft Detection")
    st.markdown("---")

    processed_dir_full = ROOT / config["paths"]["processed_dir"]
    processed_dir_quick = ROOT / (config["paths"]["processed_dir"] + "_quick")
    mode_options = []
    if processed_dir_full.exists():
        mode_options.append("full")
    if processed_dir_quick.exists():
        mode_options.append("quick")
    if not mode_options:
        mode_options = ["full"]

    chosen_mode = st.selectbox(
        "Dataset mode", mode_options,
        index=0,
        format_func=lambda m: "FULL MODE (complete dataset)" if m == "full" else "QUICK MODE (subset, for demo/dev)",
        key="data_mode_select",
    )
    st.session_state["data_mode"] = chosen_mode

    model_tags = []
    models_dir = ROOT / config["paths"]["models_dir"]
    if models_dir.exists():
        model_tags = sorted(p.stem for p in models_dir.glob("*.joblib"))
    default_tag = "xgboost" if "xgboost" in model_tags else (model_tags[0] if model_tags else None)
    chosen_model = (
        st.selectbox("Model for Risk Analysis", model_tags,
                     index=model_tags.index(default_tag) if default_tag in model_tags else 0,
                     key="model_tag_select")
        if model_tags else None
    )
    st.session_state["selected_model_tag"] = chosen_model

    st.markdown("---")
    st.markdown(
        "<p class='risk-note'>Predictions are indicators for investigation, not proof of "
        "electricity theft.</p>",
        unsafe_allow_html=True,
    )

PAGES_DIR = ROOT / "app" / "pages"
pg = st.navigation([
    st.Page(str(PAGES_DIR / "overview.py"), title="Overview", icon="🏠", default=True),
    st.Page(str(PAGES_DIR / "data_exploration.py"), title="Data Exploration", icon="🔎"),
    st.Page(str(PAGES_DIR / "customer_analysis.py"), title="Customer Risk Analysis", icon="🧑‍💼"),
    st.Page(str(PAGES_DIR / "model_performance.py"), title="Model Performance", icon="📈"),
    st.Page(str(PAGES_DIR / "research_insights.py"), title="Research Insights", icon="🧪"),
])
pg.run()
