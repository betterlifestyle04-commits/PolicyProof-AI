"""Synthetic CPU tests for original CuratorKIT owner transfer ZIP (NO TRAINING)."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import build_native_train_owner_zip as target


@pytest.fixture
def fake_curated(tmp_path, monkeypatch):
    src = tmp_path / "artifacts" / "curator_train_dual_export"
    src.mkdir(parents=True)
    source = {
        "instruction": "Answer in English. [POL-001] Test",
        "input": "",
        "output": '{"answer":"Test","source_id":"POL-001","abstain":false}',
    }
    shares = {
        "conversations": [
            {"from": "human", "value": source["instruction"]},
            {"from": "gpt", "value": source["output"]},
        ],
    }
    (src / "sft_alpaca.jsonl").write_text(json.dumps(source) + "\n")
    (src / "sft_sharegpt.jsonl").write_text(json.dumps(shares) + "\n")
    for name in ("manifest.json", "lexsi_provenance.json", "checksums.txt",
                 "rejected.jsonl"):
        (src / name).write_text("{}\n")
    monkeypatch.setattr(target, "check_train_data", lambda path: {
        "dataset_sha256": target.EXPECTED_SHA, "row_count": 78
    })
    monkeypatch.setattr(target, "check_dual_export_consistency", lambda a, b: {
        "all_sft_pairs_match": True, "paired_examples": len(a)
    })
    monkeypatch.setattr(target, "check_zip", lambda path: {
        "status": "pass", "row_count": 78
    })
    return tmp_path, src


def test_owner_zip_packages_only_original_training_files_without_gpu(fake_curated):
    root, src = fake_curated
    receipt = target.build_owner_training_zip(root=root)
    path = root / "artifacts/policyproof_curatorkit_train_v1.zip"
    assert receipt["archive"] == str(path)
    assert receipt["status"] == "ORIGINAL_78_CURATORKIT_TRAIN_ZIP_VERIFIED_NO_MODEL_TRAINING"
    assert receipt["original_training_sha256"] == target.EXPECTED_SHA
    assert receipt["actual_gpu_training_executed"] is False
    assert receipt["model_weights_included"] is False
    assert receipt["held_out_48_60_72_examples_included"] is False
    assert receipt["license_and_model_download_approved_by_this_script"] is False
    with ZipFile(path) as z:
        assert sorted(z.namelist()) == sorted(target.REQUIRED)
        assert not any("adapter" in x or "challenge" in x or "holdout" in x
                       for x in z.namelist())
        assert z.read("sft_alpaca.jsonl") == (src / "sft_alpaca.jsonl").read_bytes()
    assert target._file_sha(path) == receipt["archive_sha256"]
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        target.build_owner_training_zip(root=root)
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad", [
    "artifacts/../secrets.zip",
    "data/output.zip",
    "/tmp/out.zip",
    "artifacts/nested/training.zip",
    "artifacts/.hidden.zip",
    "artifacts/unsafe\\test.zip",
])
def test_output_scope_refuses_unsafe_paths(fake_curated, bad):
    root, _ = fake_curated
    with pytest.raises(ValueError):
        target.build_owner_training_zip(root=root, output=bad)
    assert not (root / "artifacts/policyproof_curatorkit_train_v1.zip").exists()


def test_missing_file_fails_before_output(fake_curated):
    root, src = fake_curated
    (src / "checksums.txt").unlink()
    with pytest.raises(ValueError, match="Missing required"):
        target.build_owner_training_zip(root=root)
    assert not (root / "artifacts/policyproof_curatorkit_train_v1.zip").exists()


def test_modified_training_dataset_hash_never_creates_zip(fake_curated, monkeypatch):
    root, _ = fake_curated
    monkeypatch.setattr(target, "check_train_data", lambda path: {
        "dataset_sha256": "0" * 64, "row_count": 78,
    })
    with pytest.raises(ValueError, match="original pinned"):
        target.build_owner_training_zip(root=root)
    assert not (root / "artifacts/policyproof_curatorkit_train_v1.zip").exists()


def test_symlinked_curatorkit_file_fails_closed(fake_curated):
    root, src = fake_curated
    (src / "sft_alpaca.jsonl").unlink()
    outside = root / "private_source"
    outside.write_text("unsafe")
    (src / "sft_alpaca.jsonl").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlinked"):
        target.build_owner_training_zip(root=root)
    assert not (root / "artifacts/policyproof_curatorkit_train_v1.zip").exists()


def test_forged_zip_verification_does_not_publish(fake_curated, monkeypatch):
    root, src = fake_curated
    monkeypatch.setattr(target, "check_zip", lambda _: {
        "status": "failed", "row_count": 78
    })
    with pytest.raises(ValueError, match="strict verifier"):
        target.build_owner_training_zip(root=root)
    assert not (root / "artifacts/policyproof_curatorkit_train_v1.zip").exists()


def test_source_changes_during_zip_verification_abort_without_final_artifact(
    fake_curated, monkeypatch
):
    root, src = fake_curated
    def sabotaged_check(path):
        (src / "sft_alpaca.jsonl").write_text('{"mutated":"after-check"}\n')
        return {"status": "pass", "row_count": 78}
    monkeypatch.setattr(target, "check_zip", sabotaged_check)
    with pytest.raises(ValueError, match="Source changed"):
        target.build_owner_training_zip(root=root)
    assert not (root / "artifacts/policyproof_curatorkit_train_v1.zip").exists()
