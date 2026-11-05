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
    "Paired model evidence",
    "Safety benchmark",
    "Output guard experiment",
    "Try the output guard",
    "Final benchmark review",
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



def test_review_workbench_is_unsigned_and_can_export_blank_packet():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Final benchmark review").run()
    assert not app.exception, str(app.exception)
    assert any("No reviewer has signed it off" in w.value for w in app.warning)
    assert any("Independent reviewers needed" in m.label for m in app.metric)
    # AppTest does not expose Streamlit download buttons. Chromium E2E
    # verifies the real downloadable control; AppTest checks state and status.
    assert app.text_input[0].value == "reviewer_01"


def test_review_workbench_rejects_invalid_reviewer_identity():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Final benchmark review").run()
    assert not app.exception, str(app.exception)
    app.text_input[0].set_value("not an anonymous code").run()
    assert not app.exception, str(app.exception)
    assert any("Reviewer code" in e.value for e in app.error)



def test_judge_page_discloses_final_paired_evaluation_blockers():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Judge verification").run()
    assert not app.exception, str(app.exception)
    assert any("Frozen base vs adapter evaluation readiness" in x.value for x in app.markdown)
    assert any("Paired GPU evaluation: BLOCKED" in warning.value for warning in app.warning)



def test_judge_page_links_the_unpublished_native_evidence_handoff():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Judge verification").run()
    assert not app.exception, str(app.exception)
    assert any(
        "Reproducible, unpublished Track 2a evidence handoff" in item.value
        for item in app.markdown
    )
    assert any(
        "checksum-verified ZIP" in item.value
        for item in app.info
    )



def test_judge_page_honestly_labels_owner_publication_path():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Judge verification").run()
    assert not app.exception, str(app.exception)
    assert any(
        "No publication has been performed" in item.value for item in app.markdown
    )
    assert any(
        "V2 dataset publication instructions" in item.value for item in app.markdown
    )



def test_judge_page_requires_source_linked_real_human_review_not_just_pseudonyms():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Judge verification").run()
    assert not app.exception, str(app.exception)
    assert any(
        "completed-signoff boolean" in item.value
        for item in app.markdown
    )
    assert any(
        "Private review-source and signoff instructions" in item.value
        for item in app.markdown
    )
    assert any(
        "Paired GPU evaluation: BLOCKED" in item.value
        for item in app.warning
    )



def test_paired_model_evidence_page_rejects_semantic_overclaim():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Paired model evidence").run()
    assert not app.exception, str(app.exception)
    assert any("Do NOT treat valid JSON" in w.value for w in app.warning)
    assert any("Required JSON schema" == m.label for m in app.metric)
    assert any("Expected source ID" == m.label for m in app.metric)
    assert any("exact supported-answer text metric is 0/39" in i.value for i in app.info)
    assert any("blinded multilingual reviewer protocol" in m.value for m in app.markdown)



def test_guard_playground_flags_numeric_claim_not_in_cited_policy():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Try the output guard").run()
    example = next(item for item in app.selectbox if item.label == "Try an example")
    example.set_value("Invented numeric fee (needs review)").run()
    validate = next(item for item in app.button if item.label == "Validate response")
    validate.click().run()
    assert not app.exception, str(app.exception)
    assert any("Numeric evidence review needed" in item.value for item in app.error)
    assert any("risk flag, not a human factual verdict" in item.value for item in app.error)


def test_output_guard_results_include_honest_numeric_triage():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Output guard experiment").run()
    assert not app.exception, str(app.exception)
    assert any("Cases needing numeric evidence review" in item.label for item in app.metric)
    assert any("No model inference was rerun" in item.value for item in app.caption)



def test_judge_release_ledger_shows_both_tracks_and_unfinished_steps():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Judge verification").run()
    assert not app.exception, str(app.exception)
    assert any("Track 1 checks passing" == m.label for m in app.metric)
    assert any("Track 2a checks passing" == m.label for m in app.metric)
    assert any("Neither track has passed its full local submission gate" in warning.value
               for warning in app.warning)
    assert any("Release gate verification protocol" in item.value
               for item in app.markdown)



def test_guard_playground_flags_wrong_product_even_when_number_is_cited():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Try the output guard").run()
    example = next(x for x in app.selectbox if x.label == "Try an example")
    assert "Wrong banking product despite cited amount" in example.options
    example.set_value("Wrong banking product despite cited amount").run()
    assert any("Fictional user question" in x.label for x in app.text_input)
    question = next(x for x in app.text_input if "Fictional user question" in x.label)
    assert "overdraft" in question.value
    validate = next(x for x in app.button if x.label == "Validate response")
    validate.click().run()
    assert not app.exception, str(app.exception)
    assert any("Source-topic mismatch requires HUMAN review" in x.value
               for x in app.error)
    assert not any("Numeric evidence review needed" in x.value
                   for x in app.error)


def test_saved_topic_relevance_is_present_and_not_mislabeled_factual_accuracy():
    dashboard = Path(__file__).resolve().parents[1] / "research_dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Output guard experiment").run()
    assert not app.exception, str(app.exception)
    assert any(x.label == "Cited-topic mismatch review flags" for x in app.metric)
    assert any("A matching fee number may refer to the wrong product" in x.value
               for x in app.warning)
    assert any("reference answers and gold source IDs are not used" in x.value
               for x in app.caption)
