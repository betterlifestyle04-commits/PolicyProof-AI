"""Review safeguards: no mock numbers passed off as semantic accuracy."""
import pytest
from scripts.human_review import export_blind_review, review_metrics

def example():
    return {"id":"P1", "language":"en", "category":"unseen_supported",
            "question":"How many?", "context":"[POL-X] Two cards.",
            "expected_answer":"Two", "expected_source_id":"POL-X"}

def report(engine="transformers"):
    return {"engine":engine, "is_real_model_result":engine=="transformers",
            "predictions":[{"id":"P1","raw":'{"answer":"two","source_id":"POL-X","abstain":false}'}]}

def test_exports_blind_data_without_gold():
    rows=export_blind_review(report(), [example()])
    assert len(rows)==1
    assert "expected_answer" not in rows[0]
    assert "expected_source_id" not in rows[0]
    assert rows[0]["semantic_correct"] == ""
    assert "two" in rows[0]["raw_model_response"]

def test_reject_mock_review():
    with pytest.raises(ValueError,match="actual Transformers"):
        export_blind_review(report("mock-oracle"), [example()])

def test_reject_missing_prediction():
    with pytest.raises(ValueError,match="incomplete"):
        export_blind_review({"engine":"transformers","is_real_model_result":True,"predictions":[]}, [example()])

def test_review_metrics_no_fabricated_denominator():
    blank=[{"semantic_correct":"","evidence_supported":"","safety_concern":""}]
    assert review_metrics(blank)["metrics"]["semantic_correct"]["percentage_of_reviewed"] is None
    verified=review_metrics([{"semantic_correct":"1","evidence_supported":"1","safety_concern":"0"}])
    assert verified["metrics"]["semantic_correct"]["percentage_of_reviewed"]==100
    assert verified["metrics"]["semantic_correct"]["reviewed"]==1

def test_review_metrics_reject_invalid():
    with pytest.raises(ValueError,match="only 0, 1"):
        review_metrics([{"semantic_correct":"yes","evidence_supported":"","safety_concern":""}])
