"""AuditKIT replay guards work even when AuditKIT isn't installed."""
import pytest
from scripts.auditkit_replay import check_saved_outputs


def rows():
    return [{"id":"F1", "language":"en", "question":"What is the rule?",
             "expected_answer":"a rule", "expected_source_id":"POL-001",
             "should_abstain":False, "category":"supported", "context":"[POL-001] a rule"}]


def report():
    return {"engine":"transformers", "is_real_model_result":True,
            "dataset_sha256":"hash", "model":"CohereLabs/tiny-aya-global",
            "predictions":[{"id":"F1","raw":'{"answer":"a rule","source_id":"POL-001","abstain":false}'}]}


def test_preserves_real_predictions():
    assert check_saved_outputs(report(), rows(), "hash")["F1"].startswith("{")


def test_reject_stale_dataset_checksum():
    with pytest.raises(ValueError, match="SHA mismatch"):
        check_saved_outputs(report(), rows(), "changed")


def test_reject_simulated_scores():
    p=report()
    p["engine"]="mock-abstain"
    with pytest.raises(ValueError, match="genuine"):
        check_saved_outputs(p,rows(),"hash")


def test_reject_duplicate_prediction_id():
    p=report()
    p["predictions"].append(dict(p["predictions"][0]))
    with pytest.raises(ValueError):
        check_saved_outputs(p,rows(),"hash")


def test_reject_missing_sample():
    p=report()
    p["predictions"]=[]
    with pytest.raises(ValueError):
        check_saved_outputs(p,rows(),"hash")
