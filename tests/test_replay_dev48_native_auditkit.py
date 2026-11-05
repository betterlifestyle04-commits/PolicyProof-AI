"""Synthetic-only tests of native AuditKIT replay; no real model is run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from policyproof.core import ROOT, prepare_rows, sha256
from scripts.replay_dev48_native_auditkit import (
    _matched_counts, validated_dev48_source, replay,
)
from tests.test_export_review_from_bundle import make_fake_bundle


def fake_pair_bundle(tmp_path: Path, *, altered=None):
    fixture = json.loads((ROOT / "data/policies.json").read_text(encoding="utf-8"))
    _, _, rows = prepare_rows(fixture)

    def decorate(base, tuned, paired):
        paired.update({
            "split": "unseen-policy",
            "not_comparable_to_historical_unpinned_48_base": True,
            "baseline_model_revision": base["model_revision"],
            "generation_config": base["generation_config"],
            "case_matched_deterministic_counts": _matched_counts(rows, base, tuned),
            "verified_human_semantic_improvement": False,
        })
        if altered:
            altered(base, tuned, paired)

    return make_fake_bundle(tmp_path, edits=decorate)


def test_preflight_uses_original_saved_reports_only(tmp_path):
    root, path, digest = fake_pair_bundle(tmp_path)
    rows, base, tuned, original, meta = validated_dev48_source(root, path, digest)
    assert len(rows) == 48
    assert len(base["predictions"]) == len(tuned["predictions"]) == 48
    assert original["case_matched_deterministic_counts"]["correct"]["count_delta"] == 0
    assert meta["source_zip_sha256"] == digest
    assert not (root / "reviews").exists()


def test_real_native_auditkit_replays_synthetic_test_fixture(tmp_path):
    """Official AuditKIT evaluates fake text only; not a measured model result."""
    pytest.importorskip("auditkit")
    root, path, digest = fake_pair_bundle(tmp_path)
    outcome = replay(root, path, digest)
    assert outcome["paired_cases"] == 48
    assert outcome["model_inference"] is False
    assert outcome["human_semantic_review"] is False
    assert outcome["comparison_matches_saved_counts"] is True
    output = root / "reviews/dev48_native_auditkit_cpu_owner_only.json"
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["auditkit_executed_in_this_process"] is True
    assert receipt["replayed_model_outputs_total"] == 96
    assert receipt["final_blind_benchmark"] is False
    assert receipt["semantic_correctness_human_review_completed"] is False
    assert receipt["safe_for_blinded_reviewers_before_voting"] is False
    assert receipt["independent_gpu_generation_proven"] is False
    assert receipt["original_base_report_member_sha256"]
    assert receipt["original_tuned_report_member_sha256"]
    assert receipt["original_paired_report_member_sha256"]
    assert receipt["case_matched_deterministic_counts"]["correct"]["count_delta"] == 0
    with pytest.raises(FileExistsError, match="overwrite"):
        replay(root, path, digest)
    assert path.exists()


def test_sha_mismatch_refuses_to_create_native_receipt(tmp_path):
    root, path, digest = fake_pair_bundle(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        replay(root, path, "0" * 64)
    assert not (root / "reviews").exists()


@pytest.mark.parametrize("alter,match", [
    (
        lambda base, tuned, paired: paired.update(
            case_matched_deterministic_counts={"correct": {"count_delta": 100}}),
        "Archived paired counts conflict",
    ),
    (
        lambda base, tuned, paired: paired.update(
            verified_human_semantic_improvement=True),
        "strict development provenance",
    ),
    (
        lambda base, tuned, paired: paired.update(
            baseline_model_revision="f" * 40),
        "strict development provenance",
    ),
    (
        lambda base, tuned, paired: paired.update(
            not_comparable_to_historical_unpinned_48_base=False),
        "strict development provenance",
    ),
])
def test_rejects_forged_pair_metadata_before_native_runner(tmp_path, alter, match):
    root, path, digest = fake_pair_bundle(tmp_path, altered=alter)
    with pytest.raises(ValueError, match=match):
        replay(root, path, digest)
    assert not (root / "reviews").exists()


def test_tuned_raw_output_mutation_cannot_keep_old_scores(tmp_path):
    def bad(base, tuned, paired):
        tuned["predictions"][0]["raw"] = '{"answer":"invented","source_id":"POL-U003","abstain":false}'
    root, path, digest = fake_pair_bundle(tmp_path, altered=bad)
    with pytest.raises(ValueError, match="Saved output metrics do not match"):
        replay(root, path, digest)
    assert not (root / "reviews").exists()


def test_source_and_result_files_are_never_overwritten(tmp_path):
    root, path, digest = fake_pair_bundle(tmp_path)
    from scripts.paired_semantic_review import _private_path
    with pytest.raises(ValueError, match="Private evidence path"):
        _private_path(root, "artifacts/paid_receipt.json", ".json", new=True)
    assert path.is_file()
