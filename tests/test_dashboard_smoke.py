"""Run actual Streamlit dashboard pages with the official AppTest harness.

This is a CPU/UI-smoke test, NOT a deployed website availability check.
"""
from pathlib import Path
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("page", [
    "Research overview",
    "Dataset explorer",
    "Evaluation runs",
    "Output guard experiment",
    "Try the output guard",
    "Judge verification",
    "Protocol & next steps",
])
def test_dashboard_page_runs_without_streamlit_exception(page):
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    assert not app.exception, str(app.exception)
    assert app.title[0].value == "PolicyProof AI"
    if page != "Research overview":
        app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, str(app.exception)
    assert app.sidebar.radio[0].value == page
