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
    "Safety benchmark",
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



def test_safety_benchmark_exposes_per_case_evidence_trace():
    """The judge UI must expose preserved raw output, source text and gold disclaimer."""
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Safety benchmark").run()
    assert not app.exception, str(app.exception)
    assert any(
        "Inspect an original scored case" in item.label for item in app.expander
    ), "The case-level evidence trace is missing"
    assert any(item.label == "Scored case language" for item in app.selectbox)
    assert any(item.label == "Scored case to inspect" for item in app.selectbox)


def test_guard_playground_includes_duplicate_json_attack_example():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Try the output guard").run()
    assert not app.exception, str(app.exception)
    example = next(item for item in app.selectbox if item.label == "Try an example")
    assert "Duplicate JSON member" in example.options
