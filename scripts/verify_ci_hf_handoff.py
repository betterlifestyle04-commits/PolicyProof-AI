"""Verify a *specific successful main-branch* GitHub Actions research handoff.

This is a read-only CPU preflight. It binds a downloaded release ZIP to the
owner-selected SHA-256 AND the GitHub run metadata downloaded over authenticated
GitHub API. Only workflow_dispatch owner actions may proceed to publication;
this script never authenticates Hugging Face or uploads anything.

It also does not prove the historical saved model GPU inference or human
semantic accuracy. GitHub run metadata and checksums are source provenance,
not independent experimental attestation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from scripts.publish_track2a_candidate import inspect_source

RUN = re.compile(r"^[1-9][0-9]{0,17}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY = "betterlifestyle04-commits/PolicyProof-AI"
EXPECTED_WORKFLOW = ".github/workflows/ci.yml"
ARCHIVE = "track2a_release_candidate_v2_UNPUBLISHED.zip"
MAX_METADATA_BYTES = 200_000


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate GitHub workflow metadata field: " + key)
        result[key] = value
    return result


def verify_owner_handoff(
    metadata: Path, download_dir: Path, *, run_id: str,
    expected_sha256: str, repository: str = REPOSITORY,
) -> dict:
    """Validate run origin, non-symlink ZIP and immutable research payload."""
    if (not isinstance(run_id, str) or not RUN.fullmatch(run_id)
        or not isinstance(repository, str) or repository != REPOSITORY):
        raise ValueError("Require a numeric, owner-repository CI run ID")
    metadata, download_dir = Path(metadata), Path(download_dir)
    if (metadata.is_symlink() or not metadata.is_file()
        or not 0 < metadata.stat().st_size <= MAX_METADATA_BYTES):
        raise ValueError("Run metadata is absent, symlinked or oversized")
    try:
        run = json.loads(
            metadata.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicates,
            parse_constant=lambda x: (_ for _ in ()).throw(
                ValueError("Nonfinite workflow run metadata: " + x)
            ),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Malformed or ambiguous GitHub run metadata") from exc
    if not isinstance(run, dict):
        raise ValueError("Run metadata must be one JSON object")
    source_repo = run.get("repository")
    head_repo = run.get("head_repository")
    commit = run.get("head_sha")
    if (type(run.get("id")) is not int
        or str(run["id"]) != run_id
        or not isinstance(source_repo, dict)
        or source_repo.get("full_name") != repository
        or not isinstance(head_repo, dict)
        or head_repo.get("full_name") != repository
        or run.get("path") != EXPECTED_WORKFLOW
        or run.get("event") != "push"
        or run.get("head_branch") != "main"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or not isinstance(commit, str) or not SHA40.fullmatch(commit)):
        raise ValueError(
            "Original source must be a SUCCESSFUL original-repository main "
            "push on the official CI workflow, not a fork, PR, or other branch"
        )
    if download_dir.is_symlink() or not download_dir.is_dir():
        raise ValueError("Artifact download directory missing or unsafe")
    # Artifact root layout depends on upload-artifact's common prefix.
    # Accept only the two actual layouts and reject ambiguity.
    candidates = [
        download_dir / ARCHIVE,
        download_dir / "artifacts" / ARCHIVE,
    ]
    found = []
    for candidate in candidates:
        if candidate.is_symlink() or candidate.parent.is_symlink():
            raise ValueError("Unsafe downloaded original research ZIP symlink")
        if candidate.exists():
            if not candidate.is_file():
                raise ValueError("Downloaded research ZIP is not a regular file")
            found.append(candidate)
    if len(found) != 1:
        raise ValueError("Expected exactly one original research ZIP from the selected CI run")
    audit = inspect_source(found[0], expected_sha256)
    if audit.get("file_count") != 15:
        raise ValueError("Unexpected original research ZIP file count")
    return {
        "status": "VERIFIED_OWNER_SELECTED_MAIN_CI_ZIP_NO_PUBLICATION",
        "ci_run_id": run_id,
        "ci_head_sha": commit,
        "ci_conclusion": "success",
        "official_workflow": EXPECTED_WORKFLOW,
        "original_archive_path": str(found[0]),
        "approved_archive_sha256": expected_sha256,
        "files_strictly_verified": audit["file_count"],
        "model_generated_in_this_process": False,
        "model_training_performed": False,
        "dataset_uploaded": False,
        "model_collection_created": False,
        "official_submission_completed": False,
        "original_gpu_inference_independently_authenticated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--download-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(
        verify_owner_handoff(
            args.metadata, args.download_dir,
            run_id=args.run_id, expected_sha256=args.expected_sha256,
        ), ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
