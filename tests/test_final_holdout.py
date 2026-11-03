"""Candidate final-holdout data-integrity tests (no model calls)."""
import json
from pathlib import Path
import pytest
from scripts.validate_final_holdout import validate_final_holdout
from policyproof.core import ROOT, read_jsonl, write_jsonl


def test_new_candidate_is_disjoint_and_balanced():
    r=validate_final_holdout()
    assert r["cases"] == 60
    assert r["languages"] == {"en":20, "no":20, "ar":20}
    assert r["categories"] == {
        "final_supported":48, "final_unknown":9, "final_injection":3
    }
    assert r["translations_human_reviewed"] is False
    assert r["final_results_measured"] is False


def test_reject_policy_leakage(tmp_path):
    rows=read_jsonl(ROOT/"data/final_holdout_v1.jsonl")
    leak=[dict(rows[0])]
    leak[0]["id"]="LEAK-TRAIN"
    leak[0]["context"]="[POL-F101] Bad leaked source in train"
    leak[0]["expected_source_id"]="POL-F101"
    train=tmp_path/"leak.jsonl"
    write_jsonl(train,leak)
    with pytest.raises(ValueError,match="leaked"):
        validate_final_holdout(existing_train=train)


def test_reject_bad_abstention_gold(tmp_path):
    rows=read_jsonl(ROOT/"data/final_holdout_v1.jsonl")
    sample=next(x for x in rows if x["should_abstain"])
    sample["expected_answer"]="invented answer"
    changed=tmp_path/"holdout.jsonl"
    write_jsonl(changed,rows)
    with pytest.raises(ValueError,match="Abstention"):
        validate_final_holdout(holdout_file=changed)


def test_new_policies_not_original_fixture():
    old=json.loads((ROOT/"data/policies.json").read_text(encoding="utf-8"))
    orig={p["id"] for p in old["policies"]}
    held=set(validate_final_holdout()["new_source_policies"])
    assert not orig.intersection(held)
