"""Read-only recovery for a PARTIALLY SUCCESSFUL owner-authorized HF upload.

The public dataset and its dataset collection can exist even if the original
workflow fails after uploading while reading back cached Hugging Face JSON.
This module NEVER creates repos, collections, pushes files, reads HF_TOKEN, or
submits an organizer form. It independently re-downloads the public dataset
through HfApi(token=False), verifies SHA256SUMS and source-claim provenance,
then saves a non-overwriting dataset-only receipt.

THIS CANNOT SATISFY the separate public MODEL collection requirement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import tempfile

from scripts.publish_track2a_dataset import LICENSES
from scripts.verify_hf_collections import HF_REPO, HF_COLLECTION, verify_anonymous_access

SHA = re.compile(r"^[a-f0-9]{64}$")
GIT_SHA = re.compile(r"^[a-f0-9]{40}$")
STATUS = "EXISTING_PUBLIC_DATASET_ANONYMOUSLY_VERIFIED_ONLY_NOT_SUBMITTED"


def inspect_existing_publication(*, repo_id: str, collection_slug: str,
                                 expected_source_sha256: str, license_id: str,
                                 expected_revision: str | None = None,
                                 api=None) -> dict:
    """Read-only pinned remote verification, including ALL published hashes."""
    if not isinstance(repo_id, str) or not HF_REPO.fullmatch(repo_id):
        raise ValueError("Expected full owner/dataset repository")
    if not isinstance(collection_slug, str) or not HF_COLLECTION.fullmatch(collection_slug):
        raise ValueError("Expected owner/existing-public-collection slug")
    if collection_slug.split("/", 1)[0] != repo_id.split("/", 1)[0]:
        raise ValueError("Dataset collection must belong to the dataset owner")
    if not isinstance(expected_source_sha256, str) or not SHA.fullmatch(expected_source_sha256):
        raise ValueError("Owner-approved original ZIP SHA256 must be exact 64-hex")
    if license_id not in LICENSES:
        raise ValueError("Owner-selected license not recognized")
    if expected_revision is not None and (
        not isinstance(expected_revision, str) or not GIT_SHA.fullmatch(expected_revision)
    ):
        raise ValueError("Optional owner-pinned HF revision must be 40-hex")

    with tempfile.TemporaryDirectory(prefix="policyproof_hf_existing_readback_") as t:
        manifest = Path(t) / "readonly_existing.json"
        manifest.write_text(json.dumps({
            "dataset_url": "https://huggingface.co/datasets/" + repo_id,
            "dataset_collection_url": "https://huggingface.co/collections/" + collection_slug,
            "model_collection_url": None,
        }), encoding="utf-8")
        verified = verify_anonymous_access(manifest, api=api, dataset_only=True)

    dataset = next(
        (entry for entry in verified["dataset_repositories"] if entry["id"] == repo_id),
        None,
    )
    revision = dataset["revision"] if dataset else None
    checks = {
        "read_only_dataset_status": verified.get("status") == "ANONYMOUS_DATASET_ACCESS_VERIFIED_ONLY",
        "repository": verified.get("dataset_id") == repo_id,
        "collection": verified.get("dataset_collection_slug") == collection_slug,
        "published_cc_license": verified.get("declared_dataset_license") == license_id,
        "owner_approved_original_zip": (
            verified.get("approved_source_archive_sha256_claim") == expected_source_sha256
        ),
        "all_hashes_verified": verified.get("all_declared_public_files_sha256_verified") is True,
        "sha256_manifest_checked": verified.get("public_internal_checksums_verified") is True,
        "exact_public_v2_file_count": verified.get("public_verified_file_count") == 14,
        "original_saved_predictions": verified.get("public_baseline_cases") == 48,
        "new_uns cored challenge": verified.get("public_unscored_challenge_cases") == 72,
        "public_commit_pinned": (
            isinstance(revision, str) and GIT_SHA.fullmatch(revision) is not None
        ),
        "optional_expected_remote_revision": (
            expected_revision is None or revision == expected_revision
        ),
    }
    if not all(checks.values()):
        raise ValueError(
            "Existing public dataset does not match approved source and study: "
            + ", ".join(name for name, passed in checks.items() if not passed)
        )
    return {
        "schema": "policyproof_existing_dataset_readonly_recovery_v1",
        "status": STATUS,
        "verified_utc": verified["checked_utc"],
        "public_dataset_repo": verified["dataset_url"],
        "public_dataset_collection": verified["dataset_collection_url"],
        "dataset_public_commit_sha": revision,
        "approved_source_zip_sha256": expected_source_sha256,
        "verified_public_license": license_id,
        "published_public_files": 15,
        "source_prediction_cases_verified": 48,
        "additional_unscored_challenge_cases_verified": 72,
        "all_content_sha256_verified": True,
        "external_verification_used_anonymous_client": True,
        "huggingface_token_used": False,
        "huggingface_write_performed": False,
        "dataset_created_by_this_check": False,
        "dataset_collection_created_by_this_check": False,
        "public_model_collection_requirement_verified": False,
        "official_lexsi_submission_completed": False,
        "warning": (
            "This is a read-only dataset-only recovery receipt. It does NOT "
            "verify rights, owner authorization, an actual model collection, "
            "a trained model, human semantic review or Lexsi acceptance."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--collection-slug", required=True)
    p.add_argument("--expected-source-sha256", required=True)
    p.add_argument("--license", required=True, choices=sorted(LICENSES))
    p.add_argument("--expected-revision", default=None)
    p.add_argument("--output", type=Path,
                   default=Path("artifacts/track2a_existing_dataset_readonly_receipt.json"))
    args = p.parse_args()
    target = args.output
    if target.exists() or target.is_symlink():
        p.error("Receipt already exists; refusing to overwrite evidence")
    result = inspect_existing_publication(
        repo_id=args.repo_id,
        collection_slug=args.collection_slug,
        expected_source_sha256=args.expected_source_sha256,
        license_id=args.license,
        expected_revision=args.expected_revision,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
