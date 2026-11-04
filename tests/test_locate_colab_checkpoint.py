"""Test Colab checkpoint upload resolution; no Colab, GPU or model used."""
from __future__ import annotations

from hashlib import sha256 as hashlib_sha256
import json
from pathlib import Path

import pytest

from policyproof.core import ROOT
from scripts.locate_colab_checkpoint import locate_verified_checkpoint


FILENAME = "policyproof_t4_torch_extended.zip"


def _fake_bytes(directory: Path, payload: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FILENAME
    path.write_bytes(payload)
    return path


def test_uploaded_in_project_cwd_is_found_without_reupload(tmp_path):
    repo = tmp_path / "PolicyProof-AI"
    location = _fake_bytes(repo, b"NOT_ACTUAL_ADAPTER_TEST_ONLY")
    expected = hashlib_sha256(location.read_bytes()).hexdigest()
    assert locate_verified_checkpoint(FILENAME, expected, [
        repo, tmp_path / "content", repo
    ]) == location.resolve()


def test_original_content_location_still_works(tmp_path):
    content = tmp_path / "content"
    location = _fake_bytes(content, b"TEST_ONLY_ORIGINAL_CONTENT")
    expected = hashlib_sha256(location.read_bytes()).hexdigest()
    assert locate_verified_checkpoint(FILENAME, expected, [
        tmp_path / "working", content
    ]) == location.resolve()


def test_finds_exact_original_if_an_older_copy_has_bad_checksum(tmp_path):
    project = tmp_path / "project"
    content = tmp_path / "content"
    _fake_bytes(project, b"incorrect stale test contents")
    exact = _fake_bytes(content, b"correct test checkpoint contents")
    expected = hashlib_sha256(exact.read_bytes()).hexdigest()
    assert locate_verified_checkpoint(FILENAME, expected, [project, content]) == exact


def test_changed_uploaded_zip_must_fail_sha_without_fallback(tmp_path):
    path = _fake_bytes(tmp_path, b"wrong test payload")
    assert path.is_file()
    with pytest.raises(ValueError, match="SHA256 differs"):
        locate_verified_checkpoint(FILENAME, "0" * 64, [tmp_path])


def test_not_uploaded_is_explicit_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No uploaded"):
        locate_verified_checkpoint(FILENAME, "a" * 64, [tmp_path])


@pytest.mark.parametrize("name", [
    "../policyproof_t4_torch_extended.zip", "/content/model.zip",
    r"other\file.zip", "model.bin", "",
])
def test_untrusted_filename_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="simple .zip"):
        locate_verified_checkpoint(name, "a" * 64, [tmp_path])


def test_symlink_is_not_treated_as_direct_uploaded_file(tmp_path):
    origin = _fake_bytes(tmp_path / "original", b"TRAINING_TEST_NOT_REAL")
    linked = tmp_path / "alias"
    linked.mkdir()
    (linked / FILENAME).symlink_to(origin)
    expected = hashlib_sha256(origin.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unsafe"):
        locate_verified_checkpoint(FILENAME, expected, [linked])


def test_new_notebook_uses_location_resolver_and_keeps_gpu_disabled():
    notebook = json.loads(
        (ROOT / "notebooks/06_dev48_pair_eval_t4.ipynb").read_text(encoding="utf-8")
    )
    upload = notebook["cells"][6]["source"]
    if isinstance(upload, list):
        upload = "".join(upload)
    assert "locate_verified_checkpoint" in upload
    assert "Path.cwd()" in upload
    assert "FILENAME in uploaded" in upload
    assert "EXPECTED_ZIP_SHA" in upload
    assert 'ARCHIVE=Path("/content/policyproof_t4_torch_extended.zip")' not in upload
    assert "RUN_DEV48_GPU_EVAL = False" in json.dumps(notebook)
    assert all(
        c.get("execution_count") is None and not c.get("outputs", [])
        for c in notebook["cells"] if c.get("cell_type") == "code"
    )
