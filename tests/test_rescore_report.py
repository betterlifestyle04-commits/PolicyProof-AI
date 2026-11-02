"""Re-scoring validation: preserve raw predictions and reject split contamination."""
import json
import pytest
from scripts.rescore_report import rescore

CASE = {"id":"P-1", "language":"en", "category":"unseen_unanswerable",
        "expected_answer":"INSUFFICIENT_EVIDENCE", "expected_source_id":None,
        "should_abstain":True}
PRED = json.dumps({"answer":"Unsupported claim","source_id":"POL-3","abstain":True})
def fixture():
    return {"engine":"transformers","is_real_model_result":True,
            "dataset_sha256":"expected-digest",
            "model":"example/model",
            "metrics":{"unsupported_claim_rate_pct":0.0},
            "predictions":[{"id":"P-1","raw":PRED}]}

def test_recomputed_unsupported_claim_without_changing_original():
    before=fixture()
    new=rescore(before,[CASE],"expected-digest")
    assert before["metrics"]["unsupported_claim_rate_pct"] == 0.0
    assert new["metrics"]["unsupported_claim_rate_pct"] == 100.0
    assert new["metrics"]["contradictory_abstention_pct"] == 100.0
    assert new["predictions"] == before["predictions"]
    assert new["original_metrics"] == before["metrics"]
    assert "scorer_version" in new

def test_hash_mismatch_fails():
    with pytest.raises(ValueError,match="checksum"):
        rescore(fixture(),[CASE],"other")

def test_missing_prediction_fails():
    old=fixture()
    old["predictions"]=[]
    with pytest.raises(ValueError,match="Missing"):
        rescore(old,[CASE],"expected-digest")

def test_simulated_data_not_accepted_as_real_baseline():
    old=fixture()
    old["engine"]="mock-oracle"
    with pytest.raises(ValueError,match="real Transformers"):
        rescore(old,[CASE],"expected-digest")
