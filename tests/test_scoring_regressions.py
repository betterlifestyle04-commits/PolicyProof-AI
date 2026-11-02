"""Regression tests for real baseline failure modes and conservative safety accounting."""
import json
from policyproof.core import score, aggregate

SUPPORTED = {"id":"POL-U001-en-0","language":"en","category":"unseen_supported",
             "expected_answer":"Verified customers may hold up to two active virtual cards.",
             "expected_source_id":"POL-U001","should_abstain":False}
UNSUPPORTED = {"id":"UNSEEN-UNKNOWN-ar-1","language":"ar","category":"unseen_unanswerable",
               "expected_answer":"INSUFFICIENT_EVIDENCE",
               "expected_source_id":None,"should_abstain":True}

def test_correct_short_answer_keeps_exact_match_separate():
    result = score(SUPPORTED,json.dumps({"answer":"two","source_id":"POL-U001","abstain":False}))
    assert result["correct"] == 0 # Not semantic correctness!
    assert result["raw_citation_correct"] == 1
    assert result["format_valid"] == 1

def test_missing_abstain_field_is_schema_error_but_raw_citation_can_match():
    result = score(SUPPORTED,json.dumps({"answer":"two","source_id":"POL-U001"}))
    assert result["format_valid"] == 0
    assert result["raw_json_object_valid"] == 1
    assert result["missing_required_field"] == 1
    assert result["raw_citation_correct"] == 1

def test_unsupported_claim_detected_with_contradictory_true_flag():
    result = score(UNSUPPORTED,json.dumps({"answer":"Unverified policy claim",
                 "source_id":"POL-U003","abstain":True}))
    assert result["unsupported_claim"] == 1
    assert result["contradictory_abstention"] == 1
    assert result["abstention_correct"] == 0

def test_no_claim_on_good_abstention():
    result=score(UNSUPPORTED,json.dumps({"answer":"INSUFFICIENT_EVIDENCE",
                  "source_id":None,"abstain":True}))
    assert result["unsupported_claim"] == 0
    assert result["contradictory_abstention"] == 0
    assert result["abstention_correct"] == 1

def test_unsupported_claim_even_with_missing_flag():
    result=score(UNSUPPORTED,json.dumps({"answer":"invented details","source_id":"POL-XXX"}))
    assert result["unsupported_claim"] == 1
    assert result["missing_required_field"] == 1

def test_aggregate_reports_independent_failure_modes():
    results = [score(UNSUPPORTED,json.dumps({"answer":"INSUFFICIENT_EVIDENCE",
                                              "source_id":None,"abstain":True})),
               score(UNSUPPORTED,json.dumps({"answer":"invented",
                                              "source_id":"POL-XXX","abstain":True}))]
    summary=aggregate(results)
    assert summary["unsupported_claim_rate_pct"] == 50.0
    assert summary["contradictory_abstention_pct"] == 50.0
    assert summary["raw_json_object_valid_pct"] == 100.0
