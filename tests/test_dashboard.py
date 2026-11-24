"""Dashboard smoke tests using Streamlit's headless AppTest runner.
Skipped automatically if processed artifacts for QUICK MODE aren't present
(run `python scripts/prepare_data.py --quick` first)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROCESSED_QUICK = ROOT / "data" / "processed_quick"
PAGES_DIR = ROOT / "app" / "pages"

pytestmark = pytest.mark.skipif(
    not PROCESSED_QUICK.exists(), reason="requires `python scripts/prepare_data.py --quick`"
)


def _run_page(filename: str):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(PAGES_DIR / filename), default_timeout=60)
    at.session_state["data_mode"] = "quick"
    at.session_state["selected_model_tag"] = None
    at.run()
    return at


def test_overview_page_starts_without_exception():
    at = _run_page("overview.py")
    assert not at.exception


def test_data_exploration_page_starts_without_exception():
    at = _run_page("data_exploration.py")
    assert not at.exception


def test_graph_visualization_page_starts_without_exception():
    at = _run_page("graph_visualization.py")
    assert not at.exception


def test_model_performance_page_handles_missing_metrics_gracefully():
    at = _run_page("model_performance.py")
    assert not at.exception


def test_customer_analysis_page_handles_missing_model_gracefully():
    at = _run_page("customer_analysis.py")
    assert not at.exception


def test_research_insights_page_starts_without_exception():
    at = _run_page("research_insights.py")
    assert not at.exception
