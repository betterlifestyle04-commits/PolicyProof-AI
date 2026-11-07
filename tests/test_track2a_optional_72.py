"""Source-bound optional 72-case transition tests use FAKE oracle GPU fixtures.

No actual Tiny Aya, CUDA, human annotation or external publication is run.
"""
from __future__ import annotations

import json

import pytest

from policyproof.core import sha256
from scripts.score_track2a_72 import score_saved_72
from scripts.track2a_optional_72 import optional_72_evidence
from scripts.track2a_readiness import readiness
from tests.test_score_track2a_72_handoff import evidence_sources

DATASET = "artifacts/track2a_challenge_unscored/challenge_v1.jsonl"
SOURCE = "artifacts/track2a_72_unchanged_base_predictions.json"
FOLDER = "artifacts/actual_72_case_evaluation"


def simulate_original_72_receipt(tmp_path):
    """SYNTHETIC original-response fixture exclusively for CPU validator tests."""
    pytest.importorskip("auditkit")
    rows, original, dataset, source = evidence_sources(tmp_path)
    desired = tmp_path / DATASET
    desired.parent.mkdir(parents=True)
    dataset.rename(desired)
    original_file = tmp_path / SOURCE
    source.rename(original_file)
    score_saved_72(desired, original_file, FOLDER, root=tmp_path)
    return desired, original_file, tmp_path / FOLDER


def test_default_frozen_72_has_no_gpu_claim_or_fake_evidence(tmp_path):
    out = optional_72_evidence(tmp_path)
    assert out["status"] == "NO_SAVED_72_MODEL_EVALUATION"
    assert out["evidence_present"] is False
    assert out["cases_source_replayed"] == 0
    assert out["original_gpu_execution_independently_authenticated"] is False
    state = readiness(tmp_path)
    assert state["checks"]["optional_72_saved_model_source_replay"]["passed"] is False
    assert state["checks"]["optional_72_saved_model_source_replay"]["required_for_submission"] is False
    assert state["new_model_evaluation_claimed_for_72_cases"] is False
    assert state["conflicting_optional_evidence_blocks_release"] is False
    assert state["saved_72_source_replay"]["status"] == "NO_SAVED_72_MODEL_EVALUATION"
    assert not (tmp_path / "artifacts").exists()


def test_saved_72_replays_original_files_without_modifying_any_source(tmp_path):
    dataset, original, folder = simulate_original_72_receipt(tmp_path)
    paths = [dataset, original] + sorted(folder.iterdir())
    bytes_before = {str(path): path.read_bytes() for path in paths}
    outcome = optional_72_evidence(tmp_path)
    assert outcome["status"] == "SAVED_72_CPU_SOURCE_REPLAY_VERIFIED_NOT_GPU_ATTESTED"
    assert outcome["verified_source_replay"] is True
    assert outcome["cases_source_replayed"] == 72
    assert outcome["source_replay"]["dataset_sha256"] == sha256(dataset)
    assert outcome["source_replay"]["original_model_report_sha256"] == sha256(original)
    assert outcome["original_gpu_execution_independently_authenticated"] is False
    assert outcome["human_semantic_review_completed"] is False
    readiness_report = readiness(tmp_path)
    assert readiness_report["new_model_evaluation_claimed_for_72_cases"] is True
    assert readiness_report["checks"]["optional_72_saved_model_source_replay"]["passed"] is True
    assert readiness_report["saved_72_source_replay"]["cases"] == 72
    assert readiness_report["saved_72_source_replay"]["original_gpu_provenance_independently_verified"] is False
    assert readiness_report["ready_to_submit"] is False
    assert readiness_report["official_submission_confirmed"] is False
    assert all(path.read_bytes() == bytes_before[str(path)] for path in paths)


def test_partial_source_report_does_not_count_as_model_evaluation(tmp_path):
    (tmp_path / SOURCE).parent.mkdir(parents=True)
    (tmp_path / SOURCE).write_text('{"is_real_model_result":true}')
    report = optional_72_evidence(tmp_path)
    assert report["status"] == "INVALID_OR_INCOMPLETE_SAVED_72_EVIDENCE"
    assert report["verified_source_replay"] is False
    assert "source" in report["warning"].lower() or "directory" in report["warning"].lower()
    state = readiness(tmp_path)
    assert state["status"] == "SOURCE_EVIDENCE_CONFLICT_BLOCKED"
    assert state["conflicting_optional_evidence_blocks_release"] is True
    assert state["new_model_evaluation_claimed_for_72_cases"] is False


def test_fake_marker_never_marks_saved_72_completed(tmp_path):
    (tmp_path / FOLDER).mkdir(parents=True)
    (tmp_path / FOLDER / "EVIDENCE_COMPLETE.json").write_text(
        json.dumps({"status":"SAVED_72_CPU_AUDIT_COMPLETED"})
    )
    report = optional_72_evidence(tmp_path)
    assert report["status"] == "INVALID_OR_INCOMPLETE_SAVED_72_EVIDENCE"
    assert report["cases_source_replayed"] == 0
    state = readiness(tmp_path)
    assert state["status"] == "SOURCE_EVIDENCE_CONFLICT_BLOCKED"
    assert state["ready_to_submit"] is False


def test_swapped_saved_model_raw_response_rejected_even_if_marker_exists(tmp_path):
    dataset, original, folder = simulate_original_72_receipt(tmp_path)
    report = json.loads(original.read_text())
    report["predictions"][0]["raw"] = "SIMULATED_TAMPERING_WITH_SAVED_GPU_OUTPUT"
    original.write_text(json.dumps(report), encoding="utf-8")
    out = optional_72_evidence(tmp_path)
    assert out["status"] == "INVALID_OR_INCOMPLETE_SAVED_72_EVIDENCE"
    assert out["verified_source_replay"] is False
    assert readiness(tmp_path)["new_model_evaluation_claimed_for_72_cases"] is False


def test_rehashed_forged_72_score_is_still_blocked(tmp_path):
    dataset, original, folder = simulate_original_72_receipt(tmp_path)
    score_file = folder / "measured_72_report.json"
    result = json.loads(score_file.read_text())
    result["overall"]["strict_accuracy_pct"] = -100
    score_file.write_text(json.dumps(result), encoding="utf-8")
    marker_file = folder / "EVIDENCE_COMPLETE.json"
    marker = json.loads(marker_file.read_text())
    marker["measured_report_sha256"] = sha256(score_file)
    marker_file.write_text(json.dumps(marker), encoding="utf-8")
    out = optional_72_evidence(tmp_path)
    assert out["status"] == "INVALID_OR_INCOMPLETE_SAVED_72_EVIDENCE"
    assert readiness(tmp_path)["conflicting_optional_evidence_blocks_release"] is True


def test_source_evidence_symlink_cannot_bypass_validation(tmp_path):
    shared = tmp_path / "external"
    shared.mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / FOLDER).symlink_to(shared, target_is_directory=True)
    result = optional_72_evidence(tmp_path)
    assert result["evidence_present"] is True
    assert result["status"] == "INVALID_OR_INCOMPLETE_SAVED_72_EVIDENCE"
    assert not list(shared.iterdir())


def test_local_readiness_does_not_mark_original_gpu_or_semantic_grade_verified(tmp_path):
    dataset, original, folder = simulate_original_72_receipt(tmp_path)
    state = readiness(tmp_path)
    result = state["saved_72_source_replay"]
    assert result["original_gpu_provenance_independently_verified"] is False
    assert result["human_semantic_accuracy_verified"] is False
    assert result["original_model_inference_executed_by_readiness"] is False
    assert state["training_claimed"] is False
