"""Owner-approval-gated Hugging Face publisher for the NEW 15-member Track 2a ZIP.

This is a SEPARATE publication path from the historically pinned 20-file
publisher; it cannot silently override that hash or borrow its consent.

Default runs are OFFLINE / read-only or staging dry runs. A real upload
requires an exact 64-hex source ZIP digest explicitly supplied by the owner,
a chosen license, TWO distinct explicit opt-ins, a namespace/repository ID,
collection intent and an HF_TOKEN in the private runtime. Post-upload, the
published dataset and collection must pass independent *anonymous* Hub reads
including full required-file SHA256 verification, or no success receipt is
created. No model repo, model weights, GPU, reviewer signoff or Lexsi form.
"""
from __future__ import annotations

import argparse
from hashlib import sha256 as hashlib_sha256
import json
import os
from pathlib import Path
import re
import tempfile
from zipfile import ZipFile

from scripts.build_track2a_release_candidate import (
    HASH_FILE, STATUS_FILE, inspect_zip,
)
from scripts.publish_track2a_dataset import LICENSES, HF_REPO, publish
from scripts.verify_hf_collections import verify_anonymous_access

SHA = re.compile(r"^[0-9a-f]{64}$")
PREAMBLE = (
    "**Draft dataset card for a future public Hugging Face dataset. "
    "No HF publication has yet been verified.**"
)
LICENSE_HEAD = "## Rights and publishing\n"


def sha256_bytes(blob: bytes) -> str:
    return hashlib_sha256(blob).hexdigest()


def inspect_source(archive: Path, expected_digest: str) -> dict:
    """Require explicit trusted CI digest plus strict structural and score checks."""
    if not isinstance(expected_digest, str) or not SHA.fullmatch(expected_digest):
        raise ValueError("Owner must pin the complete 64-character SHA256 of the selected release ZIP")
    inspected = inspect_zip(archive)
    if inspected["archive_sha256"] != expected_digest:
        raise ValueError("Selected research ZIP differs from the owner-approved SHA256")
    return inspected


def stage_approved_candidate(
    archive: Path, folder: Path, *,
    license_id: str, expected_digest: str,
) -> dict:
    """Create PUBLICATION PREVIEW only; no external access or upload performed.

    This function requires an owner's explicit SHA and chosen license, but
    calling it is NOT confirmation that owner has publication rights.
    """
    if license_id not in LICENSES:
        raise ValueError("An explicit supported dataset license is required")
    audit = inspect_source(archive, expected_digest)
    folder = Path(folder)
    if folder.exists() and (folder.is_symlink() or not folder.is_dir()
                            or any(folder.iterdir())):
        raise FileExistsError("Stage folder must be an empty real directory")
    with ZipFile(archive, "r") as z:
        files = {
            name: z.read(name) for name in z.namelist()
            if name not in (STATUS_FILE, HASH_FILE)
        }
        original_checksums = sha256_bytes(z.read(HASH_FILE))

    card = files["README.md"].decode("utf-8")
    if card.count(PREAMBLE) != 1 or not card.startswith("---\n") or (
        "license:" in card.split("---", 2)[1]
    ) or card.count(LICENSE_HEAD) != 1:
        raise ValueError("Unrecognized or already licensed draft README")
    card = card.replace("---\n", "---\nlicense: " + license_id + "\n", 1)
    card = card.replace(
        PREAMBLE,
        "**Public synthetic research dataset. The publisher explicitly "
        "confirmed the license, ownership rights and public release.**",
    )
    card = card.split(LICENSE_HEAD)[0] + LICENSE_HEAD + (
        "Publisher-selected license: " + license_id + ". "
        "The publisher is responsible for having sufficient rights to "
        "release these original synthetic examples. Original gated Tiny Aya "
        "weights and any unverified trained adapter are not included.\n"
    )
    files["README.md"] = card.encode("utf-8")
    scope = {
        "schema": "policyproof_track2a_publication_scope_v2",
        "selected_license": license_id,
        "evaluated_base_model_cases": 48,
        "new_unscored_model_cases": 72,
        "independent_human_review_completed": False,
        "tuned_adapter_included": False,
        "original_archive_checksums_sha256": original_checksums,
        "approved_source_archive_sha256": expected_digest,
        "saved_original_model_revision_pinned": False,
        "new_model_inference_performed": False,
        "official_lexsi_submission_completed": False,
        "source_format": "policyproof_track2a_unpublished_research_v2",
    }
    files["provenance/PUBLICATION_SCOPE.json"] = (
        json.dumps(scope, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    files[HASH_FILE] = "".join(
        sha256_bytes(files[name]) + "  " + name + "\n"
        for name in sorted(files)
    ).encode("utf-8")

    # No partial staging trees on validation failure; build all bytes first.
    if len(files) != audit["file_count"]:
        raise ValueError("Staged public-file inventory changed unexpectedly")
    if sum(map(len, files.values())) > 25_000_000:
        raise ValueError("Unexpectedly large public dataset staging payload")
    folder.mkdir(parents=True, exist_ok=True)
    for name in sorted(files):
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[name])
    return {
        "status": "OFFLINE_PUBLICATION_PREVIEW_ONLY",
        "archive_sha256": expected_digest,
        "license": license_id,
        "source_files": audit["file_count"],
        "staged_public_files": len(files),
        "file_names": sorted(files),
        "staged_files_sha256": sha256_bytes(files[HASH_FILE]),
        "new_model_inference_performed": False,
        "human_semantic_review_verified": False,
        "huggingface_upload_performed": False,
        "official_submission_completed": False,
    }


def publish_after_approval(
    folder: Path, names: list[str], *,
    source_digest: str, repo_id: str,
    collection_slug: str | None,
    new_collection: bool,
    token: str,
    allow_existing_repo: bool = False,
    api=None, anonymous_api=None,
) -> dict:
    """Perform explicit owner-approved upload and strong anonymous verification."""
    if not SHA.fullmatch(source_digest):
        raise ValueError("Missing exact source SHA256")
    if not HF_REPO.fullmatch(repo_id):
        raise ValueError("Provide an HF_NAMESPACE/REPO_ID repository")
    if bool(new_collection) == bool(collection_slug):
        raise ValueError("Select exactly one collection creation or existing collection mode")
    if not token:
        raise ValueError("Private HF_TOKEN required")
    receipt = publish(
        folder, names, repo_id, token,
        new_collection=new_collection,
        collection_slug=collection_slug,
        allow_existing=allow_existing_repo,
        api=api, anonymous_api=anonymous_api,
    )
    with tempfile.TemporaryDirectory(prefix="policyproof_hf_public_check_") as scratch:
        manifest = Path(scratch) / "manifest.json"
        manifest.write_text(json.dumps({
            "dataset_url": receipt["public_dataset_repo"],
            "dataset_collection_url": receipt["public_dataset_collection"],
            "model_collection_url": None,
        }), encoding="utf-8")
        # Stronger than upload receipt: anonymous file content hashes, licenses,
        # metadata and both 48/72 published splits are checked.
        verified = verify_anonymous_access(manifest, api=anonymous_api, dataset_only=True)
    if (verified["status"] != "ANONYMOUS_DATASET_ACCESS_VERIFIED_ONLY"
        or verified["dataset_id"] != repo_id
        or verified["declared_dataset_license"] not in LICENSES
        or not verified["public_internal_checksums_verified"]):
        raise RuntimeError("Published dataset did not pass independent anonymous inspection")
    return {
        **receipt,
        "publication_schema": "policyproof_v2_owner_approved_dataset_publication",
        "source_zip_sha256": source_digest,
        "anonymous_dataset_file_hashes_verified": True,
        "anonymous_verification_checked_utc": verified["checked_utc"],
        "dataset_commit_verified": next(x["revision"] for x in verified["dataset_repositories"]
                                        if x["id"] == repo_id),
        "model_collection_requirement_verified": False,
        "trained_adapter_published": False,
        "official_lexsi_submission_completed": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--expected-archive-sha256", required=True,
                   help="Exact 64-hex digest obtained from a trusted successful CI run")
    p.add_argument("--license", choices=sorted(LICENSES),
                   help="Only the content owner may choose/approve a dataset license")
    p.add_argument("--repo-id")
    p.add_argument("--publish", action="store_true")
    p.add_argument("--confirm-rights", action="store_true")
    p.add_argument("--confirm-publication", action="store_true")
    p.add_argument("--create-collection", action="store_true")
    p.add_argument("--collection-slug")
    p.add_argument("--allow-existing-repo", action="store_true")
    p.add_argument("--receipt", type=Path, default=Path("track2a_v2_publication_receipt.json"))
    args = p.parse_args()

    # Permission and config checks precede ALL network requests.
    if args.publish:
        if not (args.confirm_rights and args.confirm_publication):
            p.error("--publish requires --confirm-rights AND --confirm-publication")
        if args.license is None or args.repo_id is None:
            p.error("--publish requires --license and --repo-id")
        if args.repo_id and not HF_REPO.fullmatch(args.repo_id):
            p.error("Invalid Hugging Face repository namespace/name")
        if bool(args.create_collection) == bool(args.collection_slug):
            p.error("Choose exactly one of --create-collection or --collection-slug")
        if not os.environ.get("HF_TOKEN"):
            p.error("No HF_TOKEN configured in this private runtime")
        if args.receipt.exists() or args.receipt.is_symlink():
            p.error("Receipt already exists; refusing to overwrite or duplicate publication")
    elif args.create_collection or args.collection_slug or args.allow_existing_repo:
        p.error("Collection/repository mutation options require --publish")
    if args.confirm_rights != args.confirm_publication:
        p.error("Both confirmation flags must be used together")

    evidence = inspect_source(args.archive, args.expected_archive_sha256)
    if args.license is None:
        if args.publish:
            p.error("License missing")
        print(json.dumps({
            **evidence, "status": "SOURCE_ZIP_VERIFIED_WAITING_OWNER_LICENSE",
            "network_used": False, "staged_for_upload": False,
            "huggingface_upload_performed": False,
        }, indent=2))
        return
    with tempfile.TemporaryDirectory(prefix="policyproof_v2_publish_preview_") as tmp:
        folder = Path(tmp)
        preview = stage_approved_candidate(
            args.archive, folder, license_id=args.license,
            expected_digest=args.expected_archive_sha256,
        )
        print(json.dumps(preview, indent=2))
        if not args.publish:
            return
        receipt = publish_after_approval(
            folder, preview["file_names"],
            source_digest=preview["archive_sha256"],
            repo_id=args.repo_id,
            collection_slug=args.collection_slug,
            new_collection=args.create_collection,
            allow_existing_repo=args.allow_existing_repo,
            token=os.environ["HF_TOKEN"],
        )
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open("x", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
        print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
