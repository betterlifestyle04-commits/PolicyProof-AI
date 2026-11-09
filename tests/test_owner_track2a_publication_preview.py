"""Synthetic offline owner HF dataset preview tests — NO public upload or license approval."""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from zipfile import ZipFile, ZIP_STORED

import pytest

from policyproof.core import ROOT
from scripts import owner_track2a_publication_preview as owner
from tests.test_publish_track2a_candidate import candidate


def test_exact_preview_rebuild_and_no_publication(tmp_path):
    original, digest = candidate(tmp_path)
    before = original.read_bytes()
    zipped, report = owner.build_preview(original, digest, "cc-by-4.0")
    assert report["status"] == "OWNER_PUBLIC_FILES_PREVIEW_REBUILT_EXACTLY_NO_PUBLICATION"
    assert report["owner_preview_archive_sha256"] == hashlib.sha256(zipped).hexdigest()
    assert report["public_payload_file_count"] == 15
    assert report["all_public_file_bytes_compared_to_source_rebuild"] is True
    for key in ("rights_approval_obtained_by_script", "license_rights_independently_verified",
                "huggingface_token_used", "huggingface_upload_performed",
                "dataset_collection_created", "model_collection_created",
                "official_lexsi_submission_completed"):
        assert report[key] is False
    with ZipFile(BytesIO(zipped)) as arc:
        assert set(arc.namelist()) == owner.PUBLIC_FILES
        assert arc.testzip() is None
        assert arc.read("README.md").splitlines()[1] == b"license: cc-by-4.0"
        scope = json.loads(arc.read("provenance/PUBLICATION_SCOPE.json"))
        assert scope["approved_source_archive_sha256"] == digest
        assert scope["independent_human_review_completed"] is False
        assert "provenance/PACKAGE_STATUS.json" not in arc.namelist()
    assert original.read_bytes() == before
    assert not (tmp_path / "artifacts").exists()


def test_write_once_and_recheck_exact_original(tmp_path, monkeypatch):
    source, digest = candidate(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = owner.write_preview(source, digest, "cc-by-4.0")
    file = tmp_path / "artifacts" / owner.ZIP_NAME
    receipt = tmp_path / "artifacts" / owner.RECEIPT_NAME
    assert file.is_file() and receipt.is_file()
    original = file.read_bytes()
    assert result["official_lexsi_submission_completed"] is False
    assert owner.inspect_preview_bytes(source, digest, "cc-by-4.0", original)[
        "public_payload_file_count"
    ] == 15
    with pytest.raises(ValueError, match="never overwrite"):
        owner.write_preview(source, digest, "cc-by-4.0")
    assert file.read_bytes() == original


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "tampered", "rehashed"])
def test_mutated_owner_preview_rejected_even_after_crc_rewrite(tmp_path, change):
    source, digest = candidate(tmp_path)
    blob, _ = owner.build_preview(source, digest, "cc-by-4.0")
    with ZipFile(BytesIO(blob)) as arc:
        files = {name: arc.read(name) for name in arc.namelist()}
    if change == "missing":
        files.pop("challenge_72/challenge_v1_UNSCORED.jsonl")
    if change == "tampered":
        files["README.md"] += b"FAKE RIGHTS"
    if change == "rehashed":
        scope = json.loads(files["provenance/PUBLICATION_SCOPE.json"])
        scope["independent_human_review_completed"] = True
        files["provenance/PUBLICATION_SCOPE.json"] = json.dumps(scope).encode()
        manifest = "provenance/SHA256SUMS.txt"
        files[manifest] = "".join(
            hashlib.sha256(files[name]).hexdigest() + "  " + name + "\\n"
            for name in sorted(files) if name != manifest
        ).encode()
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_STORED) as arc:
        for name, data in files.items():
            arc.writestr(name, data)
        if change == "extra":
            arc.writestr("../private.txt", "dummy")
        if change == "duplicate":
            arc.writestr("README.md", files["README.md"])
    with pytest.raises(ValueError):
        owner.inspect_preview_bytes(source, digest, "cc-by-4.0", output.getvalue())


def test_sha_and_preview_license_mandatory(tmp_path):
    source, digest = candidate(tmp_path)
    with pytest.raises(ValueError):
        owner.build_preview(source, "0" * 64, "cc-by-4.0")
    with pytest.raises(ValueError):
        owner.build_preview(source, digest, "undecided")


def test_owner_workflow_preview_is_manual_and_independent_of_publish_token():
    yml = (ROOT / ".github/workflows/owner-track2a-hf-dataset.yml").read_text()
    assert "workflow_dispatch:" in yml
    assert "preview_public_dataset" in yml
    assert "OFFLINE PREVIEW" in yml
    assert "policyproof-track2a-owner-HF-PREVIEW-NOT-PUBLISHED" in yml
    assert yml.count("HF_TOKEN: ${{ secrets.HF_TOKEN }}") == 1
    assert yml.index("Construct and VERIFY exact public dataset ZIP offline") < yml.index(
        "PUBLIC owner-approved dataset upload"
    )
    assert "--confirm-rights --confirm-publication" in yml
