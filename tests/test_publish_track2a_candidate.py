"""Publisher v2 safety coverage. All API calls are synthetic local stubs.

The tests do NOT publish to Hugging Face or select a license for the owner.
All mock receipts remain inside pytest's temporary directory.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from policyproof.core import ROOT, sha256
from scripts.build_track2a_release_candidate import write_bundle
from scripts.publish_track2a_candidate import (
    inspect_source, stage_approved_candidate, publish_after_approval,
)
from tests.test_build_track2a_release_candidate import original_fixture


def candidate(tmp_path):
    work = tmp_path / "research"
    original_fixture(work)
    path = tmp_path / "candidate_v2.zip"
    receipt = write_bundle(work, path)
    return path, receipt["archive_sha256"]


def test_review_only_dry_run_does_not_stage_or_upload(tmp_path):
    archive, digest = candidate(tmp_path)
    r = subprocess.run(
        [sys.executable, "-m", "scripts.publish_track2a_candidate",
         "--archive", str(archive), "--expected-archive-sha256", digest],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(r.stdout)
    assert report["status"] == "SOURCE_ZIP_VERIFIED_WAITING_OWNER_LICENSE"
    assert report["huggingface_upload_performed"] is False
    assert report["staged_for_upload"] is False
    assert report["file_count"] == 15


def test_local_license_staging_pins_source_and_never_starts_network(tmp_path):
    archive, digest = candidate(tmp_path)
    stage = tmp_path / "preview"
    receipt = stage_approved_candidate(
        archive, stage, license_id="cc-by-4.0", expected_digest=digest,
    )
    assert receipt["status"] == "OFFLINE_PUBLICATION_PREVIEW_ONLY"
    assert receipt["huggingface_upload_performed"] is False
    assert receipt["staged_public_files"] == 15
    assert not (stage / "provenance/PACKAGE_STATUS.json").exists()
    card = (stage / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\nlicense: cc-by-4.0\n")
    assert "NOT YET PUBLISHED" not in card
    assert "No HF publication has yet been verified" not in card
    assert "## Rights and publishing" in card
    scope = json.loads((stage/"provenance/PUBLICATION_SCOPE.json").read_text())
    assert scope["approved_source_archive_sha256"] == digest
    assert scope["new_model_inference_performed"] is False
    assert scope["independent_human_review_completed"] is False
    from scripts.verify_hf_collections import REQUIRED_DATASET_FILES
    assert set(REQUIRED_DATASET_FILES).issubset(receipt["file_names"])
    with pytest.raises(FileExistsError):
        stage_approved_candidate(archive, stage, license_id="cc0-1.0", expected_digest=digest)


def test_wrong_source_sha_cannot_stage(tmp_path):
    archive, digest = candidate(tmp_path)
    target = tmp_path / "preview"
    with pytest.raises(ValueError, match="approved SHA256"):
        stage_approved_candidate(
            archive, target, license_id="cc-by-4.0", expected_digest="0"*64
        )
    assert not target.exists()
    with pytest.raises(ValueError, match="complete 64-character"):
        inspect_source(archive, "not-a-cryptographic-sha")
    assert inspect_source(archive, digest)["file_count"] == 15


def test_historical_publisher_remains_pinned_to_its_other_archive(tmp_path):
    archive, digest = candidate(tmp_path)
    from scripts.publish_track2a_dataset import validate_archive
    with pytest.raises(ValueError, match="ZIP hash differs"):
        validate_archive(archive)


def test_authorized_publish_cli_fails_closed_without_user_consent_and_token(tmp_path, monkeypatch):
    archive, digest = candidate(tmp_path)
    common = [sys.executable, "-m", "scripts.publish_track2a_candidate",
              "--archive", str(archive), "--expected-archive-sha256", digest,
              "--license", "cc-by-4.0", "--repo-id", "owner/research",
              "--create-collection", "--publish"]
    monkeypatch.delenv("HF_TOKEN", raising=False)
    p = subprocess.run(common, cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert p.returncode != 0 and "confirm-rights" in p.stderr
    p2 = subprocess.run(
        common + ["--confirm-rights", "--confirm-publication"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert p2.returncode != 0 and "HF_TOKEN" in p2.stderr
    assert not (ROOT / "track2a_v2_publication_receipt.json").exists()


def test_published_mutation_flags_rejected_during_offline_dry_run(tmp_path):
    archive, digest = candidate(tmp_path)
    for flag in ("--allow-existing-repo", "--create-collection"):
        p = subprocess.run(
            [sys.executable, "-m", "scripts.publish_track2a_candidate",
             "--archive", str(archive), "--expected-archive-sha256", digest,
             flag], cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        assert p.returncode != 0
        assert "require --publish" in p.stderr


class LocalOnlyHub:
    """Fake mutable/repository state, never connects to actual Hugging Face."""
    def __init__(self):
        self.files = {}
        self.items = []
        self.private = False
        self.gated = False
        self.tamper = None
        self.calls = []

    def create_repo(self, **kwargs):
        assert kwargs["private"] is False
        self.calls.append("create_repo")

    def upload_folder(self, **kwargs):
        root = Path(kwargs["folder_path"])
        self.files = {
            str(file.relative_to(root)): file
            for file in root.rglob("*") if file.is_file()
        }
        self.calls.append("upload")
        return SimpleNamespace(oid="a"*40)

    def repo_info(self, **kwargs):
        self.calls.append(("repo_info", kwargs.get("token")))
        return SimpleNamespace(
            private=self.private, gated=self.gated, sha="a"*40
        )

    def list_repo_files(self, **kwargs):
        self.calls.append(("list_repo_files", kwargs.get("token")))
        return list(self.files)

    def hf_hub_download(self, **kwargs):
        assert kwargs["token"] is False
        name = kwargs["filename"]
        self.calls.append(("download", name, kwargs["token"]))
        if name == self.tamper:
            tmp = self.files[name].parent / "TAMPERED_PUBLIC_READ.txt"
            tmp.write_bytes(b"CORRUPTED_AFTER_UPLOAD")
            return str(tmp)
        return str(self.files[name])

    def create_collection(self, **kwargs):
        assert kwargs["private"] is False
        self.calls.append("create_collection")
        return SimpleNamespace(slug="owner/policyproof-track2a-datasets-0000")

    def add_collection_item(self, **kwargs):
        self.items.append(SimpleNamespace(
            item_id=kwargs["item_id"], item_type=kwargs["item_type"]
        ))

    def get_collection(self, slug, *, token=False):
        assert token is False
        self.calls.append(("get_collection", token))
        return SimpleNamespace(private=self.private, items=self.items)


def test_strong_post_upload_anonymous_read_with_stubbed_api(tmp_path):
    archive, digest = candidate(tmp_path)
    preview_dir = tmp_path / "preview"
    preview = stage_approved_candidate(
        archive, preview_dir, license_id="cc0-1.0", expected_digest=digest,
    )
    fake = LocalOnlyHub()
    result = publish_after_approval(
        preview_dir, preview["file_names"], source_digest=digest,
        license_id="cc0-1.0", repo_id="owner/research", collection_slug=None,
        new_collection=True, token="SYNTHETIC_TEST_TOKEN_NOT_REAL",
        api=fake, anonymous_api=fake,
    )
    assert result["anonymous_dataset_file_hashes_verified"] is True
    assert result["source_zip_sha256"] == digest
    assert result["dataset_commit_verified"] == "a"*40
    assert result["model_collection_requirement_verified"] is False
    assert result["official_lexsi_submission_completed"] is False
    assert "create_repo" in fake.calls
    assert "create_collection" in fake.calls
    assert any(x[0] == "download" and x[-1] is False
               for x in fake.calls if isinstance(x, tuple))


def test_no_success_receipt_if_public_files_corrupted_after_stub_upload(tmp_path):
    archive, digest = candidate(tmp_path)
    preview_dir = tmp_path/"preview"
    preview = stage_approved_candidate(
        archive, preview_dir, license_id="cc-by-sa-4.0", expected_digest=digest,
    )
    fake = LocalOnlyHub()
    fake.tamper = "evaluated_48/original_saved_tiny_aya_base_model_predictions.json"
    with pytest.raises(ValueError, match="SHA256"):
        publish_after_approval(
            preview_dir, preview["file_names"], source_digest=digest,
            license_id="cc-by-sa-4.0", repo_id="owner/research", collection_slug=None,
            new_collection=True, token="FAKE", api=fake, anonymous_api=fake,
        )


def test_anonymous_private_or_gated_repo_fails_publication(tmp_path):
    archive, digest = candidate(tmp_path)
    stage = tmp_path/"preview"
    preview = stage_approved_candidate(
        archive, stage, license_id="cc0-1.0", expected_digest=digest,
    )
    for field, value in (("private",True), ("gated","auto")):
        fake = LocalOnlyHub()
        setattr(fake, field, value)
        with pytest.raises((ValueError, RuntimeError)):
            publish_after_approval(
                stage, preview["file_names"], source_digest=digest,
                license_id="cc0-1.0", repo_id="owner/research", collection_slug=None,
                new_collection=True, token="FAKE", api=fake, anonymous_api=fake,
            )


def test_two_mutually_exclusive_collection_modes_are_required(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        publish_after_approval(
            tmp_path, [], source_digest="a"*64, license_id="cc0-1.0", repo_id="owner/research",
            collection_slug=None, new_collection=False, token="FAKE",
        )



def test_preupload_stage_mutation_is_rejected_before_any_api_call(tmp_path):
    from scripts.publish_track2a_candidate import validate_public_staging
    archive, digest = candidate(tmp_path)
    directory = tmp_path / "preview"
    preview = stage_approved_candidate(
        archive, directory, license_id="cc-by-4.0", expected_digest=digest,
    )
    assert validate_public_staging(
        directory, preview["file_names"], source_digest=digest,
        license_id="cc-by-4.0",
    )["checksums_verified"] is True
    file = directory / "challenge_72/challenge_v1_UNSCORED.jsonl"
    file.write_text("MUTATED BEFORE UPLOAD", encoding="utf-8")
    fake = LocalOnlyHub()
    with pytest.raises(ValueError, match="contents changed"):
        publish_after_approval(
            directory, preview["file_names"], source_digest=digest,
            license_id="cc-by-4.0", repo_id="owner/research",
            new_collection=True, collection_slug=None, token="FAKE",
            api=fake, anonymous_api=fake,
        )
    assert "create_repo" not in fake.calls


def test_publication_refuses_license_changed_after_stage(tmp_path):
    archive, digest = candidate(tmp_path)
    directory = tmp_path / "preview"
    preview = stage_approved_candidate(
        archive, directory, license_id="cc-by-4.0", expected_digest=digest,
    )
    fake = LocalOnlyHub()
    with pytest.raises(ValueError, match="Stage README license"):
        publish_after_approval(
            directory, preview["file_names"], source_digest=digest,
            license_id="cc0-1.0", repo_id="owner/research",
            new_collection=True, collection_slug=None, token="FAKE",
            api=fake, anonymous_api=fake,
        )
    assert fake.calls == []
