"""Track 2a real CuratorKIT-AuditKIT chain preflight (no toolkit install needed)."""
import copy
import json
import pytest
from policyproof.core import ROOT, read_jsonl
from scripts.track2a_chain import (
    curator_alpaca_rows, match_curated, validate_benchmark, aggregate_slice,
)

def load_cases():
    return read_jsonl(ROOT/"data/unseen_policy_test.jsonl")


def test_all_48_preserved_under_curator_format():
    rows=load_cases()
    validate_benchmark(rows)
    alpac=curator_alpaca_rows(rows)
    assert len(alpac)==48
    assert all("BENCH_CASE_ID:"+r["id"] in a["instruction"]
               for r,a in zip(rows,alpac))
    assert all(a["input"]=="" for a in alpac)
    match_curated(rows,alpac)

def test_curator_validation_fails_on_label_mutation():
    rows=load_cases()
    alpac=curator_alpaca_rows(rows)
    modified=copy.deepcopy(alpac)
    modified[0]["output"]=json.dumps({"answer":"FABRICATED","source_id":None,"abstain":True})
    with pytest.raises(ValueError,match="changed reference"):
        match_curated(rows,modified)

def test_curator_validation_fails_on_missing_or_repeated_case():
    rows=load_cases()
    alpac=curator_alpaca_rows(rows)
    with pytest.raises(ValueError,match="dropped or added"):
        match_curated(rows,alpac[:47])
    duplicated=alpac[:47]+[alpac[0]]
    with pytest.raises(ValueError,match="duplicated"):
        match_curated(rows,duplicated)

def test_benchmark_mix_cannot_silently_change():
    rows=load_cases()
    bad=copy.deepcopy(rows)
    bad[0]["category"]="context_injection"
    with pytest.raises(ValueError,match="case/category"):
        validate_benchmark(bad)
    bad=copy.deepcopy(rows)
    bad[0]["human_reviewed"]=True
    with pytest.raises(ValueError,match="unreviewed"):
        validate_benchmark(bad)

def test_reports_do_not_claim_semantic_safety():
    rows=load_cases()
    fake={r["id"]:json.dumps({"answer":"INSUFFICIENT_EVIDENCE",
       "source_id":None,"abstain":True}) for r in rows}
    result=aggregate_slice(rows,fake)
    assert result["samples"]==48
    assert result["supported_cases"]==39
    assert result["unsupported_cases"]==9
    assert result["strict_exact"]==9
    assert result["exact_supported_answers"]==0
