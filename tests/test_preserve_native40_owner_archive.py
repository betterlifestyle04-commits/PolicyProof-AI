"""Private backup transfer tests use FAKE archive bytes and a stubbed verifier.

No real native AlignTune run or GPU optimization happens in this test module.
The production helper always calls the real CPU archive verifier.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import preserve_native40_owner_archive as backup


def fake_native_archive(tmp_path, monkeypatch):
    source = tmp_path / "policyproof_align_native40.zip"
    source.write_bytes(b"FAKE_NATIVE40_ARCHIVE_FOR_TRANSFER_UNIT_TEST_ONLY" * 4096)
    original = source.read_bytes()
    calls = []
    def verify(path):
        calls.append(Path(path))
        if Path(path).read_bytes() != original:
            raise ValueError("Unit-test synthetic artifact changed")
        return {
            "native_aligntune_40step_receipt_checked": True,
            "optimizer_steps_self_reported": 40,
            "backend": "aligntune_wrapper",
            "expected_tiny_aya_lora_geometry": True,
            "file_hashes_match": True,
            "training_independently_verified": False,
            "model_improvement_verified": False,
            "archive_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        }
    monkeypatch.setattr(backup, "inspect_training_archive", verify)
    return source, original, calls


def test_backup_is_exact_write_once_native40_and_reuse_is_read_only(tmp_path, monkeypatch):
    source, original, calls = fake_native_archive(tmp_path, monkeypatch)
    private = tmp_path / "private_owner_drive"
    first = backup.preserve_native40_archive(source, private)
    target = Path(first["backup_path"])
    assert target.parent == private
    assert first["status"] == "PRIVATE_NATIVE40_BACKUP_BYTE_VERIFIED_NOT_GPU_ATTESTED"
    assert first["archive_sha256"] == hashlib.sha256(original).hexdigest()
    assert first["reused_existing_exact_backup"] is False
    assert target.read_bytes() == original
    assert source.read_bytes() == original
    assert all(first[k] is False for k in (
        "original_training_gpu_independently_verified",
        "model_improvement_verified", "official_submission_completed",
        "published_to_huggingface",
    ))
    saved = target.stat().st_mtime_ns
    second = backup.preserve_native40_archive(source, private)
    assert second["reused_existing_exact_backup"] is True
    assert target.stat().st_mtime_ns == saved
    assert target.read_bytes() == original
    assert len(calls) == 4  # source + saved each time; no GPU/training/network


def test_existing_corrupt_or_partial_backup_never_overwritten(tmp_path, monkeypatch):
    source, original, calls = fake_native_archive(tmp_path, monkeypatch)
    private = tmp_path / "private"
    private.mkdir()
    digest = hashlib.sha256(original).hexdigest()
    target = private / ("policyproof_align_native40_" + digest[:16] + ".zip")
    target.write_bytes(b"INCOMPLETE_BACKUP_DO_NOT_REPLACE")
    with pytest.raises(FileExistsError, match="never replace"):
        backup.preserve_native40_archive(source, private)
    assert target.read_bytes() == b"INCOMPLETE_BACKUP_DO_NOT_REPLACE"
    assert source.read_bytes() == original


def test_no_copy_when_adapter_provenance_is_invalid(tmp_path, monkeypatch):
    source, original, calls = fake_native_archive(tmp_path, monkeypatch)
    private = tmp_path / "private"
    def invalid(path):
        return {
            "native_aligntune_40step_receipt_checked": False,
            "optimizer_steps_self_reported": 40,
            "backend": "manual_torch_extended_fallback",
        }
    monkeypatch.setattr(backup, "inspect_training_archive", invalid)
    with pytest.raises(ValueError, match="not a checked original native"):
        backup.preserve_native40_archive(source, private)
    assert not private.exists()
    assert source.read_bytes() == original


@pytest.mark.parametrize("which", ["source", "folder", "parent"])
def test_symlinks_refused_before_any_backup_write(tmp_path, monkeypatch, which):
    source, original, _ = fake_native_archive(tmp_path, monkeypatch)
    real_private = tmp_path / "real_private"
    real_private.mkdir()
    if which == "source":
        renamed = tmp_path / "original"
        source.rename(renamed)
        source.symlink_to(renamed)
        private = real_private
    elif which == "folder":
        private = tmp_path / "symlinked_private"
        private.symlink_to(real_private, target_is_directory=True)
    else:
        alias = tmp_path / "alias_parent"
        alias.symlink_to(real_private, target_is_directory=True)
        private = alias / "nested"
    with pytest.raises(ValueError, match="symlink"):
        backup.preserve_native40_archive(source, private)
    assert not (real_private / "nested").exists()


def test_missing_original_and_wrong_filename_do_not_create_backup(tmp_path):
    private = tmp_path / "private"
    with pytest.raises(ValueError, match="Expected original native40"):
        backup.preserve_native40_archive(tmp_path / "fake_other.zip", private)
    with pytest.raises(ValueError, match="missing"):
        backup.preserve_native40_archive(
            tmp_path / "policyproof_align_native40.zip", private
        )
    assert not private.exists()


def test_changed_archive_after_initial_verification_is_blocked(tmp_path, monkeypatch):
    source, original, calls = fake_native_archive(tmp_path, monkeypatch)
    def mutate_on_hash(path):
        if Path(path) == source:
            source.write_bytes(source.read_bytes() + b"CHANGED_POST_CHECK")
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    monkeypatch.setattr(backup, "_digest", mutate_on_hash)
    with pytest.raises(ValueError, match="Original archive hash mismatches"):
        backup.preserve_native40_archive(source, tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_native40_notebook_never_downloads_inside_optimizer_cell():
    notebook = json.loads(
        (Path(__file__).parents[1] / "notebooks/11_native_aligntune_40step_t4.ipynb"
        ).read_text(encoding="utf-8")
    )
    cells = ["".join(c["source"]) for c in notebook["cells"]
             if c["cell_type"] == "code"]
    train = [c for c in cells if "recover_train_save(" in c]
    assert len(train) == 1
    assert "RUN_NATIVE_TRAINING = False" in train[0]
    assert 'if existing_zip.exists():' in train[0]
    assert "SKIPPING ALL training" in train[0]
    assert "assert not native_dir.exists()" in train[0]
    assert "files.download" not in train[0]
    assert train[0].count("recover_train_save(") == 1
    exports = [c for c in cells if "SAVE_PRIVATE_DRIVE_BACKUP = False" in c]
    assert len(exports) == 1
    owner = exports[0]
    assert "DOWNLOAD_NATIVE40_ZIP = False" in owner
    assert "preserve_native40_archive(" in owner
    assert "files.download(str(ORIGINAL_NATIVE40_ZIP))" in owner
    assert "if SAVE_PRIVATE_DRIVE_BACKUP or DOWNLOAD_NATIVE40_ZIP:" in owner
    assert "inspect_training_archive(ORIGINAL_NATIVE40_ZIP)" in owner
    assert "recover_train_save(" not in owner
    assert "trainer.train(" not in owner
    assert "torch.optim" not in owner
    assert "drive.mount" in owner
    assert 'training_independently_verified"] is False' in owner
    for code in cells:
        compile(code, "native40_cpu_only_notebook_syntax", "exec")
