"""Synthetic CPU regression: independently re-score, never fake real results."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyproof.core import sha256
from scripts.rescore_original_native40_dev48 import (
    ORIGINAL_NATIVE40_ZIP, rescore_original_bundle,
)
from tests.test_export_review_from_bundle import make_fake_bundle


def native_fixture(tmp_path, edits=None):
    def adjust(base, tuned, paired):
        paired["training_backend_declared"] = "aligntune_wrapper"
        if edits is not None:
            edits(base, tuned, paired)
    return make_fake_bundle(tmp_path, edits=adjust)


def test_96_raw_predictions_recalculated_against_frozen_gold(tmp_path):
    root, archive, digest = native_fixture(tmp_path)
    result = rescore_original_bundle(root, archive, digest)
    assert result["status"] == "FROZEN_GOLD_AND_96_RAW_PREDICTIONS_STRICTLY_RESCORED"
    assert result["original_raw_response_count"] == 96
    assert result["all_original_saved_case_scores_exactly_reproduced"] is True
    assert result["metrics"]["schema_valid"]["raw_before"] == 48
    assert result["metrics"]["schema_valid"]["raw_after"] == 48
    assert result["metrics"]["unsupported_abstention_correct"]["raw_before"] == 9
    assert result["metrics"]["unsupported_abstention_correct"]["raw_after"] == 9
    assert result["metrics"]["supported_answer_exact"]["raw_before"] == 39
    assert result["metrics"]["supported_answer_exact"]["raw_after"] == 39
    assert result["metrics"]["schema_valid"]["exploratory_exact_mcnemar_p_unrounded"] == 1.0
    assert result["independent_human_semantic_review_completed"] is False
    assert result["original_GPU_execution_independently_attested"] is False
    assert result["raw_private_model_predictions_included"] is False
    assert [result["languages"][lang]["case_count"] for lang in ("en","no","ar")] == [16]*3
    blob = json.dumps(result, ensure_ascii=False)
    assert "model.layers" not in blob
    assert "raw_response" not in blob


@pytest.mark.parametrize("change", [
    lambda b,t,p: b["scores"][0].update(correct=1 - b["scores"][0]["correct"]),
    lambda b,t,p: t["predictions"][0].update(raw='{"answer":"forged", "source_id":null,"abstain":true}'),
    lambda b,t,p: t["scores"][0].update(format_valid=0),
])
def test_forged_saved_scores_or_raw_outputs_fail_even_if_bundle_resigned(tmp_path,change):
    root, archive, digest = native_fixture(tmp_path, edits=change)
    with pytest.raises(ValueError, match="raw predictions disagree"):
        rescore_original_bundle(root, archive, digest)


def test_tampered_bundle_digest_fails_before_any_report_file(tmp_path):
    root, archive, digest = native_fixture(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rescore_original_bundle(root, archive, "0"*64)
    assert not (root/"artifacts").exists()


def test_fake_manual_fallback_cannot_claim_native_result(tmp_path):
    root, archive, digest = make_fake_bundle(tmp_path)
    with pytest.raises(ValueError, match="native40 paired"):
        rescore_original_bundle(root, archive, digest)


def test_production_default_is_pinned_original_owner_archive_not_test_fixture():
    assert ORIGINAL_NATIVE40_ZIP == (
        "e0319c83fbb87ef8570ca0c42caf68bea4a4fef1f8ae320bbe11e9a7c1761c68"
    )


def test_original_gold_data_not_modifiable_by_aggregate_rescore(tmp_path):
    root, archive, digest = native_fixture(tmp_path)
    gold = root / "data/unseen_policy_test.jsonl"
    before = sha256(gold)
    rescore_original_bundle(root, archive, digest)
    assert sha256(gold) == before
    assert archive.exists()
