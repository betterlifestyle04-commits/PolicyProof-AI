"""Anonymous Hugging Face collection/repository verification for Lexsi Track 2a.

Default command is OFFLINE and READ-ONLY. Only --verify-network uses an
anonymous (token=False) Hugging Face Hub client. The resulting receipt is
evidence that public API reads succeeded at that time; it cannot prove
license ownership, scientific validity, project relevance, or submission.

Never upload model weights, credentials or source data from this module.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from policyproof.core import ROOT, sha256
from scripts.publish_track2a_dataset import LICENSES

MANIFEST_REL = "docs/hf_publication_manifest.json"
RECEIPT_REL = "artifacts/hf_anonymous_access_receipt.json"
MAX_MANIFEST_BYTES = 100_000
MAX_CASE_BYTES = 5_000_000
MAX_COLLECTION_ITEMS = 100
MAX_AGE_HOURS = 72
HF_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
HF_COLLECTION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,125}$")
REQUIRED_DATASET_FILES = (
    "README.md",
    "provenance/SHA256SUMS.txt",
    "provenance/PUBLICATION_SCOPE.json",
    "challenge_72/challenge_v1_UNSCORED.jsonl",
    "evaluated_48/original_saved_tiny_aya_base_model_predictions.json",
)


def _no_duplicate_keys(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Duplicate JSON field: " + key)
        output[key] = value
    return output


def _read_json(path: Path, *, limit: int = MAX_CASE_BYTES):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > limit:
        raise ValueError("Missing, symlinked or oversized JSON evidence")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda val: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON value: " + val)
            ),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Malformed or ambiguous JSON evidence") from exc


def _url_id(url: str, expected_type: str) -> str:
    if not isinstance(url, str) or len(url) > 400:
        raise ValueError("Missing or oversized Hugging Face URL")
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "huggingface.co"
        or parsed.username or parsed.password or parsed.port
        or parsed.query or parsed.fragment):
        raise ValueError("Only canonical HTTPS huggingface.co URLs are accepted")
    parts = [segment for segment in parsed.path.split("/") if segment]
    if expected_type == "collection":
        if len(parts) != 3 or parts[0] != "collections":
            raise ValueError("Expected a public HF collection URL")
        identifier = "/".join(parts[1:])
        if not HF_COLLECTION.fullmatch(identifier):
            raise ValueError("Malformed Hugging Face collection slug")
        return identifier
    if expected_type == "dataset":
        if len(parts) != 3 or parts[0] != "datasets":
            raise ValueError("Expected a public HF dataset repository URL")
        identifier = "/".join(parts[1:])
        if not HF_REPO.fullmatch(identifier):
            raise ValueError("Malformed Hugging Face dataset ID")
        return identifier
    raise ValueError("Unknown Hugging Face URL type")


def validate_manifest(path: Path) -> dict:
    """Validate literal owner-entered links, never trust self-verified booleans."""
    manifest = _read_json(path, limit=MAX_MANIFEST_BYTES)
    if not isinstance(manifest, dict):
        raise ValueError("HF publication manifest must be an object")
    ds = _url_id(manifest.get("dataset_url"), "dataset")
    dscol = _url_id(manifest.get("dataset_collection_url"), "collection")
    modelcol = _url_id(manifest.get("model_collection_url"), "collection")
    if dscol == modelcol:
        raise ValueError("Separate model and dataset collections are required")
    return {
        "manifest_sha256": sha256(path),
        "dataset_id": ds,
        "dataset_collection_slug": dscol,
        "model_collection_slug": modelcol,
        "dataset_url": manifest["dataset_url"],
        "dataset_collection_url": manifest["dataset_collection_url"],
        "model_collection_url": manifest["model_collection_url"],
        "owner_supplied_verification_flags_ignored": True,
    }


def _collection_items(api, slug: str, expected_kind: str):
    collection = api.get_collection(slug, token=False)
    if getattr(collection, "private", None) is not False:
        raise ValueError("Collection is private or visibility cannot be confirmed")
    items = list(getattr(collection, "items", ()) or ())
    if not 1 <= len(items) <= MAX_COLLECTION_ITEMS:
        raise ValueError("Collection must contain 1–100 items")
    ids = []
    for item in items:
        kind, repo = getattr(item, "item_type", None), getattr(item, "item_id", None)
        if kind != expected_kind or not isinstance(repo, str) or not HF_REPO.fullmatch(repo):
            raise ValueError("Collection must include only valid " + expected_kind + " repo items")
        if repo in ids:
            raise ValueError("Duplicate repository in collection")
        ids.append(repo)
    return ids


def _public_repo(api, identifier: str, *, kind: str):
    info = api.repo_info(repo_id=identifier, repo_type=kind, token=False)
    if getattr(info, "private", None) is not False:
        raise ValueError(kind + " repo is private or privacy status is unknown: " + identifier)
    if getattr(info, "gated", None) is not False:
        raise ValueError(kind + " repo is gated or gating status is unknown: " + identifier)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Repository revision is unpinned or missing: " + identifier)
    return {"id": identifier, "revision": revision, "anonymous_access_verified": True}


def _download_public(api, repo: str, revision: str, filename: str) -> Path:
    result = api.hf_hub_download(
        repo_id=repo, repo_type="dataset", revision=revision,
        filename=filename, token=False,
    )
    path = Path(result)
    if not path.is_file() or path.stat().st_size > MAX_CASE_BYTES:
        raise ValueError("Missing or oversized publicly downloaded artifact: " + filename)
    return path


def _check_published_scope(api, dataset: str, revision: str) -> dict:
    available = set(api.list_repo_files(
        repo_id=dataset, repo_type="dataset", revision=revision, token=False
    ))
    missing = set(REQUIRED_DATASET_FILES) - available
    if missing:
        raise ValueError("Public dataset missing mandatory research files: " + ", ".join(sorted(missing)))
    checksums = _download_public(api, dataset, revision, "provenance/SHA256SUMS.txt")
    declared = {}
    try:
        lines = checksums.read_text(encoding="utf-8").splitlines()
    except UnicodeError as exc:
        raise ValueError("Published checksum manifest is unreadable") from exc
    for line in lines:
        match = re.fullmatch(r"([a-f0-9]{64})  (.+)", line)
        if not match or match.group(2) in declared:
            raise ValueError("Published file checksums malformed or duplicated")
        declared[match.group(2)] = match.group(1)
    for filename in REQUIRED_DATASET_FILES:
        if filename == "provenance/SHA256SUMS.txt":
            continue
        artifact = _download_public(api, dataset, revision, filename)
        if declared.get(filename) != sha256(artifact):
            raise ValueError("Published file content differs from public SHA256 manifest: " + filename)
    scope = _read_json(_download_public(
        api, dataset, revision, "provenance/PUBLICATION_SCOPE.json"
    ))
    if not isinstance(scope, dict) or (
        scope.get("selected_license") not in LICENSES
        or not isinstance(scope.get("original_archive_checksums_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", scope["original_archive_checksums_sha256"])
        or scope.get("evaluated_base_model_cases") != 48
        or scope.get("new_unscored_model_cases") != 72
        or scope.get("independent_human_review_completed") is not False
        or scope.get("tuned_adapter_included") is not False
    ):
        raise ValueError("Published scope conflicts with 48 evaluated / 72 unscored research status")
    baseline = _read_json(_download_public(
        api, dataset, revision,
        "evaluated_48/original_saved_tiny_aya_base_model_predictions.json"
    ))
    if (not isinstance(baseline, dict) or baseline.get("is_finetuned") is not False
        or not isinstance(baseline.get("predictions"), list)
        or len(baseline["predictions"]) != 48):
        raise ValueError("Public saved 48-case baseline is incomplete or mislabeled")
    challenge_path = _download_public(
        api, dataset, revision, "challenge_72/challenge_v1_UNSCORED.jsonl"
    )
    try:
        cases = [
            json.loads(line, object_pairs_hook=_no_duplicate_keys)
            for line in challenge_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Public challenge JSONL invalid") from exc
    if (len(cases) != 72 or len({item["id"] for item in cases}) != 72 or
        any(item.get("model_evaluated") is not False
            or item.get("human_reviewed") is not False
            or item.get("used_for_training") is not False
            for item in cases)):
        raise ValueError("Public 72-case challenge is incomplete, already scored or mismarked")
    readme = _download_public(api, dataset, revision, "README.md")
    card = readme.read_text(encoding="utf-8")
    if not card.startswith("---\\n") or not re.search(
        r"(?m)^license: " + re.escape(scope["selected_license"]) + r"$",
        card.split("---", 2)[1] if len(card.split("---", 2)) > 1 else "",
    ):
        raise ValueError("Dataset card has no matching declared public license")
    return {
        "declared_dataset_license": scope["selected_license"],
        "public_required_files_readable": list(REQUIRED_DATASET_FILES),
        "public_baseline_cases": 48,
        "public_unscored_challenge_cases": 72,
        "public_internal_checksums_verified": True,
        "published_scope_sha256": sha256(
            _download_public(api, dataset, revision, "provenance/PUBLICATION_SCOPE.json")
        ),
        "dataset_license_rights_independently_verified": False,
        "model_training_verified": False,
    }


def verify_anonymous_access(manifest_path: Path, api=None) -> dict:
    """Explicit remote read-only inspection. Authenticated credentials are NEVER used."""
    manifest = validate_manifest(manifest_path)
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=False)
    model_ids = _collection_items(api, manifest["model_collection_slug"], "model")
    data_ids = _collection_items(api, manifest["dataset_collection_slug"], "dataset")
    if manifest["dataset_id"] not in data_ids:
        raise ValueError("The published PolicyProof dataset is missing from its dataset collection")
    models = [_public_repo(api, identifier, kind="model") for identifier in model_ids]
    datasets = [_public_repo(api, identifier, kind="dataset") for identifier in data_ids]
    published = next(d for d in datasets if d["id"] == manifest["dataset_id"])
    verified_scope = _check_published_scope(api, published["id"], published["revision"])
    return {
        "schema": "policyproof_hf_anonymous_access_v1",
        "status": "ANONYMOUS_PUBLIC_ACCESS_VERIFIED",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        **manifest,
        "model_repositories": models,
        "dataset_repositories": datasets,
        "all_referenced_repositories_public_ungated": True,
        "all_collections_public": True,
        **verified_scope,
        "public_access_confirmed_by": "huggingface_hub_anonymous_api",
        "model_collection_project_relevance_verified": False,
        "organizer_acceptance_verified": False,
        "official_submission_completed": False,
        "warning": (
            "This receipt proves anonymous visibility and research-file shape at the "
            "observed revisions, not model ownership, scientific validity, license "
            "rights, Track 2a eligibility or successful submission. The benchmark-only "
            "model-collection requirement needs organizer clarification."
        ),
    }


def receipt_is_current(receipt_path: Path, manifest_path: Path,
                       *, now: datetime | None = None) -> bool:
    """Offline match/recency check of a prior network-verification *claim*.

    A stored JSON receipt is not independently signed. Re-run network
    verification before submission; do not treat owner-edited flags as proof.
    """
    try:
        record = _read_json(receipt_path, limit=MAX_MANIFEST_BYTES)
        manifest = validate_manifest(manifest_path)
        if not isinstance(record, dict):
            return False
        required = (
            record.get("schema") == "policyproof_hf_anonymous_access_v1"
            and record.get("status") == "ANONYMOUS_PUBLIC_ACCESS_VERIFIED"
            and record.get("manifest_sha256") == manifest["manifest_sha256"]
            and record.get("dataset_id") == manifest["dataset_id"]
            and record.get("dataset_collection_slug") == manifest["dataset_collection_slug"]
            and record.get("model_collection_slug") == manifest["model_collection_slug"]
            and record.get("public_access_confirmed_by") == "huggingface_hub_anonymous_api"
            and record.get("all_referenced_repositories_public_ungated") is True
            and record.get("all_collections_public") is True
            and record.get("public_baseline_cases") == 48
            and record.get("public_unscored_challenge_cases") == 72
            and record.get("public_internal_checksums_verified") is True
            and record.get("public_required_files_readable") == list(REQUIRED_DATASET_FILES)
            and record.get("official_submission_completed") is False
        )
        if not required:
            return False
        stamp = datetime.fromisoformat(record["checked_utc"].replace("Z", "+00:00"))
        current = now or datetime.now(timezone.utc)
        age = (current - stamp).total_seconds()
        return bool(stamp.tzinfo is not None and 0 <= age <= MAX_AGE_HOURS * 3600)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / MANIFEST_REL)
    parser.add_argument("--output", type=Path, default=ROOT / RECEIPT_REL)
    parser.add_argument("--verify-network", action="store_true",
                        help="Explicit anonymous Hugging Face Hub reads (no token)")
    args = parser.parse_args()
    if not args.verify_network:
        try:
            declared = validate_manifest(args.manifest)
        except (ValueError, OSError) as exc:
            print(json.dumps({
                "status": "WAITING_FOR_REAL_PUBLIC_HF_LINKS",
                "reason": str(exc), "network_used": False,
                "publication_performed": False,
            }, indent=2))
            return
        print(json.dumps({
            "status": "URL_FORMAT_ONLY_UNVERIFIED",
            "links": declared,
            "prior_receipt_matches_and_fresh": receipt_is_current(
                args.output, args.manifest
            ),
            "network_used": False, "publication_performed": False,
        }, indent=2))
        return
    if args.output.exists() or args.output.is_symlink():
        parser.error("Evidence receipt exists; preserve it and choose a new output")
    result = verify_anonymous_access(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
