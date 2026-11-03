"""Offline-only tests; make_gold outputs here are SYNTHETIC TEST FIXTURES."""
from __future__ import annotations
import copy
import hashlib
import json
from policyproof.core import make_gold
from scripts.generate_track2a_challenge import build_challenge
from scripts.score_track2a_72 import (
    KNOWN_CHALLENGE_SHA256, validate_model_report, summarize
)
import pytest


def rows_and_report():
    rows=build_challenge()
    raw="".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows).encode()
    digest=hashlib.sha256(raw).hexdigest()
    report={
        "engine":"transformers",
        "is_real_model_result":True,  # SIMULATED METADATA FOR VALIDATOR UNIT TEST ONLY
        "is_finetuned":False,
        "model":"CohereLabs/tiny-aya-global",
        "model_revision":"a"*40,
        "dataset_sha256":digest,
        "created_utc":"2026-11-03T00:00:00Z",
        "generation_config":{"do_sample":False,"max_new_tokens":128},
        "predictions":[{"id":r["id"],
                        "raw":make_gold(r["expected_answer"],r["expected_source_id"])}
                       for r in rows],
    }
    return rows,report,digest


def test_frozen_challenge_sha_and_full_72_response_coverage():
    rows,report,digest=rows_and_report()
    assert digest==KNOWN_CHALLENGE_SHA256
    mapping=validate_model_report(report,rows,digest)
    assert len(mapping)==72


def test_missing_prediction_never_generates_fabricated_scores():
    rows,report,digest=rows_and_report()
    report["predictions"].pop()
    with pytest.raises(ValueError,match="exactly 72"):
        validate_model_report(report,rows,digest)


def test_duplicate_prediction_id_fails():
    rows,report,digest=rows_and_report()
    report["predictions"][-1]["id"]=report["predictions"][0]["id"]
    with pytest.raises(ValueError,match="Duplicate"):
        validate_model_report(report,rows,digest)


def test_wrong_hash_and_fake_metadata_are_rejected():
    rows,report,digest=rows_and_report()
    with pytest.raises(ValueError,match="dataset"):
        validate_model_report(report,rows,"0"*64)
    report["is_real_model_result"]=False
    with pytest.raises(ValueError,match="actual model-inference"):
        validate_model_report(report,rows,digest)


def test_unpinned_or_tuned_checkpoint_is_rejected():
    rows,report,digest=rows_and_report()
    report["model_revision"]="not-committed"
    with pytest.raises(ValueError,match="40-character"):
        validate_model_report(report,rows,digest)
    report["model_revision"]="a"*40
    report["is_finetuned"]=True
    with pytest.raises(ValueError,match="unchanged base"):
        validate_model_report(report,rows,digest)


def test_matched_control_attack_analysis_on_synthetic_perfect_oracle():
    rows,report,digest=rows_and_report()
    mapping=validate_model_report(report,rows,digest)
    stats=summarize(rows,mapping,digest,report)
    assert stats["samples"]==72
    assert stats["overall"]["strict_accuracy_pct"]==100.0
    assert stats["human_semantic_review_completed"] is False
    assert stats["trained_adapter_evaluated"] is False
    assert stats["output_claims_from_user_report_not_independently_reinferred"] is True
    assert sum(v["n"] for v in stats["by_attack_family"].values())==72
    for result in stats["control_vs_attack_exact_pair_counts"].values():
        assert result["control_correct_attack_correct"]==12
        assert result["control_correct_attack_wrong"]==0
        assert sum(result.values())==12
    assert stats["explicit_canonical_abstention_on_answerable_cases"]["count"]==0


def test_controls_and_attacks_are_pairwise_matched_not_72_independent_policies():
    rows,report,digest=rows_and_report()
    names={x["id"] for x in rows}
    for r in rows:
        if r["category"] in ("role_spoof","source_spoof","encoded_instruction","language_override"):
            assert r["id"].rsplit("-",1)[0]+"-control" in names
