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
        "actual_gpu_inference_metadata":{
            "gpu":"FAKE_UNIT_TEST_NOT_A_GPU",
            "torch_version":"UNIT_TEST_NOT_REAL_TORCH",
            "loaded_model_revision_observed":"a"*40,
            "loaded_model_revision_verified":True,
        },
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


def test_genuine_auditkit_api_on_72_synthetic_unit_test_oracle():
    """Contract-only, synthetic in-memory fixture: NOT a real Tiny Aya result."""
    import pytest
    pytest.importorskip("auditkit")
    from scripts.score_track2a_72 import evaluate_real_run
    rows, report, digest = rows_and_report()
    measures, receipt = evaluate_real_run(report, rows, digest)
    assert measures["samples"]==72
    assert receipt["sample_count"]==72
    assert receipt["status"]=="auditkit_replayed_saved_72_outputs"
    assert receipt["model_generated_during_replay"] is False
    assert receipt["human_semantic_review_completed"] is False
    assert receipt["fingerprint"]


@pytest.mark.parametrize("field,value", [
    ("created_utc", "yesterday"),
    ("created_utc", "2026-11-03T00:00:00"),
    ("generation_config", {"do_sample": True, "max_new_tokens": 128}),
    ("generation_config", {"do_sample": False, "max_new_tokens": -1}),
    ("generation_config", {"do_sample": False, "max_new_tokens": 128, "num_beams": 3}),
    ("actual_gpu_inference_metadata", {}),
    ("actual_gpu_inference_metadata", {
        "gpu": "FAKE_UNIT_TEST_NOT_A_GPU",
        "torch_version": "TEST_RUNTIME_ONLY",
        "loaded_model_revision_observed": "b" * 40,
        "loaded_model_revision_verified": True,
    }),
])
def test_reject_ambiguous_claims_of_original_72_gpu_generation(field, value):
    rows, report, digest = rows_and_report()
    report[field] = value
    with pytest.raises(ValueError, match="inference|Generation|generation|UTC|commit|metadata|runtime"):
        validate_model_report(report, rows, digest)


def test_missing_observed_loaded_commit_is_not_substituted_by_requested_revision():
    rows, report, digest = rows_and_report()
    report["actual_gpu_inference_metadata"].pop("loaded_model_revision_observed")
    with pytest.raises(ValueError, match="observed loaded-model"):
        validate_model_report(report, rows, digest)
