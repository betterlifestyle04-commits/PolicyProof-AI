import json
import pytest
from policyproof.core import ROOT, prepare_rows, validate_no_leakage, make_prompt, make_gold, score, aggregate, parse_prediction

@pytest.fixture
def cases():
    fixture = json.loads((ROOT / "data/policies.json").read_text(encoding="utf-8"))
    return prepare_rows(fixture)

def test_expected_counts(cases):
    train, para, unseen = cases
    assert (len(train), len(para), len(unseen)) == (78, 42, 48)

def test_no_policy_id_overlap(cases):
    train, _, unseen = cases
    validate_no_leakage(train, unseen)
    x = dict(unseen[0], context=unseen[0]["context"] + "\n[POL-001] leaked")
    with pytest.raises(ValueError, match="leakage"):
        validate_no_leakage(train, [x])

def test_languages(cases):
    _, _, unseen = cases
    assert {x["language"] for x in unseen} == {"en", "no", "ar"}
    assert len([x for x in unseen if x["category"] == "context_injection"]) == 3

def test_perfect_mock_gold_is_software_only(cases):
    _, _, unseen = cases
    scores = [score(x, make_gold(x["expected_answer"], x["expected_source_id"])) for x in unseen]
    metrics = aggregate(scores)
    assert metrics["strict_accuracy_pct"] == 100
    assert metrics["abstention_recall_pct"] == 100
    assert metrics["unsupported_claim_rate_pct"] == 0

def test_abstain_on_supported_is_wrong(cases):
    _, _, unseen = cases
    x = next(x for x in unseen if not x["should_abstain"])
    s = score(x, make_gold("INSUFFICIENT_EVIDENCE", None))
    assert s["correct"] == 0

def test_unsafe_unsupported_answer(cases):
    _, _, unseen = cases
    x = next(x for x in unseen if x["should_abstain"])
    assert score(x, make_gold("Invented rate", "POL-U001"))["unsupported_claim"] == 1

def test_bad_citation(cases):
    _, _, unseen = cases
    x = next(x for x in unseen if not x["should_abstain"])
    assert score(x, make_gold(x["expected_answer"], "POL-404"))["correct"] == 0

def test_invalid_format(cases):
    _, _, unseen = cases
    assert parse_prediction("not JSON") is None
    assert score(unseen[0], "not JSON")["format_valid"] == 0

def test_prompt_instructions(cases):
    _, _, unseen = cases
    p = make_prompt(unseen[0])
    assert "INSUFFICIENT_EVIDENCE" in p and "source_id" in p
