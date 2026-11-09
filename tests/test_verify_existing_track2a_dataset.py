"""Read-only recovery tests with fake Hugging Face state. NO real uploads."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.verify_existing_track2a_dataset import inspect_existing_publication
from tests.test_publish_track2a_candidate import (
    candidate, LocalOnlyHub, stage_approved_candidate,
)


def published_fake(tmp_path):
    archive, digest = candidate(tmp_path)
    stage = tmp_path / "public-stage"
    report = stage_approved_candidate(
        archive, stage, license_id="cc-by-4.0", expected_digest=digest
    )
    repo = "owner/policyproof-track2a-benchmark"
    collection = "owner/policyproof-ai-multilingual-safety-datasets"
    fake = LocalOnlyHub()
    fake.files = {
        item.relative_to(stage).as_posix(): item
        for item in stage.rglob("*") if item.is_file()
    }
    assert len(fake.files) == 15
    fake.items = [SimpleNamespace(item_id=repo, item_type="dataset")]
    return fake, repo, collection, digest


def test_read_only_verified_existing_release_with_exact_original_zip_pin(tmp_path):
    api, repo, collection, digest = published_fake(tmp_path)
    result = inspect_existing_publication(
        repo_id=repo, collection_slug=collection,
        expected_source_sha256=digest, license_id="cc-by-4.0",
        expected_revision="a" * 40, api=api,
    )
    assert result["status"] == "EXISTING_PUBLIC_DATASET_ANONYMOUSLY_VERIFIED_ONLY_NOT_SUBMITTED"
    assert result["all_content_sha256_verified"] is True
    assert result["dataset_public_commit_sha"] == "a"*40
    assert result["published_public_files"] == 15
    assert result["source_prediction_cases_verified"] == 48
    assert result["additional_unscored_challenge_cases_verified"] == 72
    assert result["huggingface_token_used"] is False
    assert result["huggingface_write_performed"] is False
    assert result["public_model_collection_requirement_verified"] is False
    assert result["official_lexsi_submission_completed"] is False
    assert not any(item in api.calls for item in (
        "upload", "create_repo", "create_collection"
    ))
    assert all(call[1] is False for call in api.calls if
               isinstance(call, tuple) and call[0] in ("download", "get_collection"))


@pytest.mark.parametrize("change", [
    "wrong_digest", "wrong_license", "wrong_revision", "unrelated_collection",
    "missing_collection", "bad_metadata", "tampered_public_file", "wrong_owner",
])
def test_existing_release_recovery_rejects_forged_claim_or_mutated_bytes(tmp_path,change):
    api, repo, collection, digest = published_fake(tmp_path)
    kwargs = {
        "repo_id": repo, "collection_slug": collection,
        "expected_source_sha256": digest, "license_id": "cc-by-4.0",
        "expected_revision": "a"*40, "api": api,
    }
    if change == "wrong_digest":
        kwargs["expected_source_sha256"] = "f"*64
    elif change == "wrong_license":
        kwargs["license_id"] = "cc0-1.0"
    elif change == "wrong_revision":
        kwargs["expected_revision"] = "b"*40
    elif change == "unrelated_collection":
        api.items = [SimpleNamespace(item_id="owner/other", item_type="dataset")]
    elif change == "missing_collection":
        api.items = []
    elif change == "bad_metadata":
        api.gated = "auto"
    elif change == "tampered_public_file":
        api.tamper = "provenance/PUBLICATION_SCOPE.json"
    elif change == "wrong_owner":
        kwargs["collection_slug"] = "outsider/synthetic-data"
    with pytest.raises((ValueError, RuntimeError)):
        inspect_existing_publication(**kwargs)
    assert "upload" not in api.calls and "create_collection" not in api.calls


def test_read_only_recovery_source_has_no_remote_write_or_token_read():
    source = (Path(__file__).resolve().parents[1] /
              "scripts/verify_existing_track2a_dataset.py").read_text()
    for forbidden in (
        "HF_TOKEN", "create_repo(", "upload_folder(", "add_collection_item(",
        "create_collection(", "delete_repo(", "publish_after_approval(",
        "--publish", "requests.post(", "token=True",
    ):
        # Descriptive docstrings can mention HF_TOKEN; runtime never needs it.
        if forbidden == "HF_TOKEN":
            assert "os.environ" not in source
        else:
            assert forbidden not in source
    assert "verify_anonymous_access" in source
    assert "with target.open(\"x\"" in source
