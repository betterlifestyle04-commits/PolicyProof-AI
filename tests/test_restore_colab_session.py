"""CPU-only tests for Colab fresh-runtime restoration safeguards."""
from pathlib import Path
from zipfile import ZipFile
import json
import pytest

from scripts.restore_colab_session import (
    restore_curated_archive, EXPECTED_SHA, ALLOWLIST
)


def test_verified_curator_zip_extracts_allowlisted_files_only(tmp_path, monkeypatch):
    archive=tmp_path/"curated.zip"
    with ZipFile(archive,"w") as z:
        for name in sorted(ALLOWLIST):
            z.writestr(name,"allowed file")
        z.writestr("arbitrary_new_file.txt", "must not be extracted")
    monkeypatch.setattr(
        "scripts.verify_curator_archive.check_zip",
        lambda a: {"status":"pass","row_count":78}
    )
    monkeypatch.setattr(
        "scripts.check_align_config.check_train_data",
        lambda p: {"dataset_sha256":EXPECTED_SHA, "row_count":78}
    )
    dest=tmp_path/"restored"
    result=restore_curated_archive(archive,dest)
    assert result["row_count"]==78
    assert not (dest/"arbitrary_new_file.txt").exists()
    assert set(p.name for p in dest.iterdir())==ALLOWLIST


def test_refuse_wrong_training_sha_even_if_archive_passes(tmp_path, monkeypatch):
    archive=tmp_path/"curated.zip"
    with ZipFile(archive,"w") as z:
        z.writestr("sft_alpaca.jsonl", "incorrect")
    monkeypatch.setattr(
        "scripts.verify_curator_archive.check_zip",
        lambda a: {"status":"pass","row_count":78}
    )
    monkeypatch.setattr(
        "scripts.check_align_config.check_train_data",
        lambda p: {"dataset_sha256":"bad-digest","row_count":78}
    )
    with pytest.raises(ValueError,match="differs"):
        restore_curated_archive(archive,tmp_path/"restored")


def test_refuse_wrong_archive_case_count(tmp_path, monkeypatch):
    archive=tmp_path/"curated.zip"
    with ZipFile(archive,"w") as z:
        z.writestr("sft_alpaca.jsonl","some data")
    monkeypatch.setattr(
        "scripts.verify_curator_archive.check_zip",
        lambda a: {"status":"pass","row_count":70}
    )
    with pytest.raises(ValueError,match="78-example"):
        restore_curated_archive(archive,tmp_path/"restored")



def _real_byte_accurate_curator_fixture(tmp_path, monkeypatch):
    """Simulate ZIP structural checks but truly check staged contents/SHA."""
    import hashlib
    import scripts.restore_colab_session as restore
    archive = tmp_path / "verified-synthetic.zip"
    training = b"synthetic_training_source_bytes_only"
    with ZipFile(archive, "w") as z:
        z.writestr("sft_alpaca.jsonl", training)
        z.writestr("manifest.json", "trusted research metadata")
    digest = hashlib.sha256(training).hexdigest()
    monkeypatch.setattr(restore, "EXPECTED_SHA", digest)
    monkeypatch.setattr(
        "scripts.verify_curator_archive.check_zip",
        lambda _a: {"status": "pass", "row_count": 78}
    )
    monkeypatch.setattr(
        "scripts.check_align_config.check_train_data",
        lambda p: {
            "dataset_sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(),
            "row_count": 78,
        }
    )
    return archive, training


def test_idempotent_curator_restore_preserves_exact_original_bytes(tmp_path, monkeypatch):
    archive, training = _real_byte_accurate_curator_fixture(tmp_path, monkeypatch)
    dest = tmp_path / "curated"
    first = restore_curated_archive(archive, dest)
    before = {p.name: p.read_bytes() for p in dest.iterdir()}
    second = restore_curated_archive(archive, dest)
    after = {p.name: p.read_bytes() for p in dest.iterdir()}
    assert before == after
    assert (dest / "sft_alpaca.jsonl").read_bytes() == training
    assert first["existing_files_overwritten"] is False
    assert second["restore_verified_before_writes"] is True
    assert second["native_training_executed"] is False


def test_conflicting_existing_training_is_never_overwritten(tmp_path, monkeypatch):
    archive, training = _real_byte_accurate_curator_fixture(tmp_path, monkeypatch)
    dest = tmp_path / "curated"
    dest.mkdir()
    protected = dest / "sft_alpaca.jsonl"
    protected.write_bytes(b"ORIGINAL_DO_NOT_REPLACE")
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        restore_curated_archive(archive, dest)
    assert protected.read_bytes() == b"ORIGINAL_DO_NOT_REPLACE"
    assert not (dest / "manifest.json").exists()


def test_non_training_conflict_aborts_without_any_writes(tmp_path, monkeypatch):
    archive, _training = _real_byte_accurate_curator_fixture(tmp_path, monkeypatch)
    dest = tmp_path / "curated"
    dest.mkdir()
    (dest / "manifest.json").write_text("existing owner metadata", encoding="utf-8")
    with pytest.raises(FileExistsError, match="manifest.json"):
        restore_curated_archive(archive, dest)
    assert (dest / "manifest.json").read_text() == "existing owner metadata"
    assert not (dest / "sft_alpaca.jsonl").exists()


def test_bad_training_checksum_never_creates_target(tmp_path, monkeypatch):
    archive, training = _real_byte_accurate_curator_fixture(tmp_path, monkeypatch)
    import scripts.restore_colab_session as restore
    monkeypatch.setattr(restore, "EXPECTED_SHA", "0" * 64)
    dest = tmp_path / "never_written"
    with pytest.raises(ValueError, match="differs"):
        restore_curated_archive(archive, dest)
    assert not dest.exists()


def test_preexisting_symlink_target_is_not_followed(tmp_path, monkeypatch):
    archive, _training = _real_byte_accurate_curator_fixture(tmp_path, monkeypatch)
    dest = tmp_path / "curated"
    dest.mkdir()
    sentinel = tmp_path / "sensitive-owner.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    (dest / "sft_alpaca.jsonl").symlink_to(sentinel)
    with pytest.raises(ValueError, match="symlink"):
        restore_curated_archive(archive, dest)
    assert sentinel.read_text() == "preserve"
    assert (dest / "sft_alpaca.jsonl").is_symlink()


def test_symlink_destination_is_rejected(tmp_path, monkeypatch):
    archive, _training = _real_byte_accurate_curator_fixture(tmp_path, monkeypatch)
    private = tmp_path / "owner-space"
    private.mkdir()
    link = tmp_path / "curated-link"
    link.symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="safe directory"):
        restore_curated_archive(archive, link)
    assert list(private.iterdir()) == []


def test_unexpected_archive_member_is_not_copied(tmp_path, monkeypatch):
    archive, _training = _real_byte_accurate_curator_fixture(tmp_path, monkeypatch)
    with ZipFile(archive, "a") as z:
        z.writestr("private-key.txt", "SHOULD_NOT_BE_EXTRACTED")
    dest = tmp_path / "curated"
    restore_curated_archive(archive, dest)
    assert not (dest / "private-key.txt").exists()


def test_requested_model_revision_check_is_pure_and_strict():
    from scripts.restore_colab_session import validate_requested_revision
    assert validate_requested_revision(None) is None
    assert validate_requested_revision("a" * 40) == "a" * 40
    for bad in ("a" * 39, "a" * 41, "abc123", "G" * 40, True, 42):
        with pytest.raises(ValueError, match="complete pinned 40-hex"):
            validate_requested_revision(bad)


def test_bad_revision_aborts_before_heavy_gpu_imports(tmp_path):
    from scripts.restore_colab_session import restore_model_and_data
    with pytest.raises(ValueError, match="complete pinned 40-hex"):
        restore_model_and_data(tmp_path / "missing", expected_revision="not-a-SHA")


def test_requested_base_revision_is_distinct_from_loaded_model_provenance():
    from types import SimpleNamespace
    from scripts.t4_torch_fallback import check_loaded_revision

    class Model:
        def __init__(self, revision=None):
            self.config = SimpleNamespace(_commit_hash=revision)

    valid = "a" * 40
    assert check_loaded_revision(Model(valid), valid)["loaded_model_revision_verified"] is True
    missing = check_loaded_revision(Model(None), valid)
    assert missing["hub_revision_claim"] == valid
    assert missing["loaded_model_revision_verified"] is False
    assert missing["loaded_model_revision_observed"] is None
    with pytest.raises(RuntimeError, match="differs"):
        check_loaded_revision(Model("b" * 40), valid)
