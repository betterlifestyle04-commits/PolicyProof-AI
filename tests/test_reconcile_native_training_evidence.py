"""No-GPU, read-only crash/reconciliation tests using FAKE tiny PEFT files."""
import json
from pathlib import Path
import struct
from zipfile import ZipFile

import pytest

from scripts.reconcile_native_training_evidence import (
    EXPECTED_MEMBERS, inspect_native_training_evidence,
)
from scripts.verify_training_archive import CURATOR_TRAIN_SHA256, digest


def _fixture(tmp_path):
    work = tmp_path / "native40"
    adapter = work / "adapter"
    adapter.mkdir(parents=True)
    tensors = {
        "base_model.model.a.lora_A.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
        },
        "base_model.model.a.lora_B.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [4, 8],
        },
    }
    hdr = json.dumps(tensors).encode("utf-8")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(struct.pack("<Q", len(hdr)) + hdr + bytes(range(8)))
    cfg = adapter / "adapter_config.json"
    cfg.write_text('{"peft_type":"LORA","r":16}', encoding="utf-8")
    info = {
        "status": "completed",
        "model": "CohereLabs/tiny-aya-global",
        "optimizer_steps": 40,
        "optimizer_steps_completed": 40,
        "aligntune_train_result_steps": 40,
        "requested_optimizer_steps": 40,
        "examples": 78,
        "dataset_sha256": CURATOR_TRAIN_SHA256,
        "kind": "40-step native AlignTune QLoRA trial",
        "trainer_backend": "aligntune_wrapper",
        "training_engine": "AlignTune TRL wrapper.train()",
        "aligntune_trainer_completed": True,
        "aligntune_cached_train_dataset_reused": True,
        "second_data_manager_load_requested": False,
        "portable_adapter_members": list(EXPECTED_MEMBERS),
        "artifact_publish_method": "validated_temporary_zip_then_atomic_link",
        "final_benchmark_or_challenge_used_for_training": False,
        "independent_gpu_execution_attested": False,
        "loaded_revision_evidence": {
            "loaded_model_revision_observed": None,
            "loaded_model_revision_verified": False,
        },
        "base_revision": "a" * 40,
        "code_commit": "b" * 40,
        "adapter_sha256": {weights.name: digest(weights)},
        "adapter_config_sha256": digest(cfg),
        "seconds": 20.0,
    }
    (work / "run_info.json").write_text(json.dumps(info))
    archive = tmp_path / "native40.zip"
    with ZipFile(archive, "w") as z:
        for member in EXPECTED_MEMBERS:
            z.write(work / member, member)
    return work, archive


def _edit_local(directory, transform):
    path = directory / "run_info.json"
    info = json.loads(path.read_text())
    transform(info)
    path.write_text(json.dumps(info))
    return info


def test_completed_native40_consistency_is_still_only_self_reported(tmp_path):
    directory, archive = _fixture(tmp_path)
    original_bytes = {
        "local": (directory / "run_info.json").read_bytes(),
        "zip": archive.read_bytes(),
    }
    report = inspect_native_training_evidence(directory, archive)
    assert report["status"] == "MATCHED_SELF_REPORTED_COMPLETION"
    assert report["local_and_archived_receipts_identical"] is True
    assert report["archive_structural_integrity_checked"] is True
    assert report["archive_tiny_aya_geometry_matches"] is False
    assert report["owner_review_required"] is False
    assert report["independent_gpu_training_verified"] is False
    assert report["independent_semantic_quality_verified"] is False
    assert report["safe_for_unreviewed_published_model_claim"] is False
    assert report["archive_sha256"] == digest(archive)
    assert (directory / "run_info.json").read_bytes() == original_bytes["local"]
    assert archive.read_bytes() == original_bytes["zip"]


@pytest.mark.parametrize("status", [
    "in_progress", "packaging", "failed",
    "archive_published_local_receipt_failed", "completed",
])
def test_missing_zip_cannot_be_called_successful(tmp_path, status):
    directory, archive = _fixture(tmp_path)
    _edit_local(directory, lambda x: x.update(status=status))
    report = inspect_native_training_evidence(directory)
    expected = {
        "in_progress": "INTERRUPTED_OR_RUNNING_NO_ZIP",
        "packaging": "PACKAGING_UNPUBLISHED_NO_ZIP",
        "failed": "FAILED_NO_ZIP",
        "archive_published_local_receipt_failed": "PUBLISHED_ZIP_NOT_SUPPLIED",
        "completed": "CLAIMED_COMPLETE_ZIP_NOT_SUPPLIED",
    }
    assert report["status"] == expected[status]
    assert report["owner_review_required"] is True
    assert report["archive_structural_integrity_checked"] is False
    assert report["independent_gpu_training_verified"] is False


@pytest.mark.parametrize("status", [
    "packaging", "archive_published_local_receipt_failed", "failed",
])
def test_published_zip_with_interrupted_local_receipt_requires_owner_review(
    tmp_path, status,
):
    directory, archive = _fixture(tmp_path)
    def mutate(info):
        info["status"] = status
        if status != "packaging":
            info["error_type"] = "OSError"
            info["error"] = "simulated local receipt write failure"
            info["optimizer_steps_observed"] = 40
            info["seconds"] = 22.0
    _edit_local(directory, mutate)
    report = inspect_native_training_evidence(directory, archive)
    assert report["status"] == "ARCHIVE_VALID_LOCAL_RECEIPT_UNRECONCILED"
    assert report["local_and_archived_receipts_identical"] is False
    assert report["archive_structural_integrity_checked"] is True
    assert report["local_adapter_hashes_match_archive"] is True
    assert report["owner_review_required"] is True
    assert report["independent_gpu_training_verified"] is False


def test_tampered_local_evidence_does_not_match_complete_zip(tmp_path):
    directory, archive = _fixture(tmp_path)
    _edit_local(directory, lambda x: x.update(dataset_sha256="0" * 64))
    report = inspect_native_training_evidence(directory, archive)
    assert report["status"] == "RECEIPT_CONTRADICTION"
    assert report["owner_review_required"] is True


def test_local_adapter_mutated_after_publish_is_rejected(tmp_path):
    directory, archive = _fixture(tmp_path)
    (directory / "adapter/adapter_model.safetensors").write_bytes(
        (directory / "adapter/adapter_model.safetensors").read_bytes() + b"x"
    )
    report = inspect_native_training_evidence(directory, archive)
    assert report["status"] == "LOCAL_ADAPTER_HASH_CONFLICT_OR_MISSING"
    assert report["local_adapter_hashes_match_archive"] is False


def test_missing_local_adapter_is_not_replaced_from_zip(tmp_path):
    directory, archive = _fixture(tmp_path)
    (directory / "adapter/adapter_config.json").unlink()
    report = inspect_native_training_evidence(directory, archive)
    assert report["status"] == "LOCAL_ADAPTER_HASH_CONFLICT_OR_MISSING"
    assert not (directory / "adapter/adapter_config.json").exists()


def test_valid_archive_without_local_receipt_cannot_establish_provenance(tmp_path):
    directory, archive = _fixture(tmp_path)
    (directory / "run_info.json").unlink()
    report = inspect_native_training_evidence(directory, archive)
    assert report["status"] == "ARCHIVE_ONLY_NO_LOCAL_RECEIPT"
    assert report["owner_review_required"] is True
    assert report["archive_structural_integrity_checked"] is True


def test_unverified_archive_with_extra_member_is_rejected(tmp_path):
    directory, archive = _fixture(tmp_path)
    with ZipFile(archive, "a") as z:
        z.writestr("adapter/model.bin", "unsafe extra artifact")
    with pytest.raises(ValueError, match="three members"):
        inspect_native_training_evidence(directory, archive)


def test_local_or_archive_symlink_disallowed(tmp_path):
    directory, archive = _fixture(tmp_path)
    alias = tmp_path / "linked.zip"
    alias.symlink_to(archive)
    with pytest.raises(ValueError, match="ZIP symlink"):
        inspect_native_training_evidence(directory, alias)
    local_link = directory / "run_info.json"
    local_link.unlink()
    local_link.symlink_to(archive)
    with pytest.raises(ValueError, match="receipt symlink"):
        inspect_native_training_evidence(directory)


def test_duplicate_keys_in_local_receipt_rejected(tmp_path):
    directory, archive = _fixture(tmp_path)
    (directory / "run_info.json").write_text('{"status":"completed","status":"failed"}')
    with pytest.raises(ValueError, match="invalid JSON"):
        inspect_native_training_evidence(directory, archive)


def test_archive_cannot_self_prove_real_t4_training_with_fake_tensors(tmp_path):
    directory, archive = _fixture(tmp_path)
    report = inspect_native_training_evidence(directory, archive)
    assert report["archive_tiny_aya_geometry_matches"] is False
    assert report["safe_for_unreviewed_published_model_claim"] is False
    assert report["independent_gpu_training_verified"] is False


def test_no_receipts_or_zip_is_a_clear_blocker(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    report = inspect_native_training_evidence(directory)
    assert report["status"] == "NO_LOCAL_OR_ARCHIVED_EVIDENCE"
    assert report["owner_review_required"] is True
