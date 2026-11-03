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
