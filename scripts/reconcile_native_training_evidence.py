"""Read-only, CPU-only post-crash audit of native AlignTune adapter evidence.

No training, GPU, network, file writes or self-certification. Accepts a local
training directory plus an optional archived adapter ZIP. Reconciles the
locally saved receipt and adapter hashes against the independently validated
ZIP contents. Any conflict or interrupted state fails closed; *even* a clean
match is only internally consistent self-reported evidence, not independent
proof of real T4 optimizer steps or semantic model quality.

Run:
 python -m scripts.reconcile_native_training_evidence \
   --directory artifacts/align_qlora_native40 \
   --archive /content/policyproof_align_native40.zip
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from scripts.verify_training_archive import (
    _parse_json, digest, inspect_training_archive,
)

EXPECTED_MEMBERS = (
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "run_info.json",
)
MAX_LOCAL_RECEIPT = 1_000_000
# Volatile fields added only when a final local receipt write fails. Never
# exclude dataset, base revision, backend, actual steps or model hashes.
POSTPUBLICATION_FAILURE_FIELDS = frozenset(
    {"status", "error_type", "error", "optimizer_steps_observed", "seconds"}
)


def _local_receipt(directory: Path) -> dict | None:
    path = directory / "run_info.json"
    if path.is_symlink():
        raise ValueError("Unsafe local training receipt symlink")
    if not path.exists():
        return None
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_LOCAL_RECEIPT:
        raise ValueError("Missing or oversized local training receipt")
    return _parse_json(path.read_bytes(), "local training receipt")


def _safe_local_adapter_hashes(directory: Path) -> dict[str, str] | None:
    hashes: dict[str, str] = {}
    for name in EXPECTED_MEMBERS[:2]:
        path = directory / name
        if path.is_symlink():
            raise ValueError("Unsafe local adapter symlink")
        if not path.is_file():
            return None
        hashes[name] = digest(path)
    return hashes


def _immutable_receipt_view(receipt: dict) -> dict:
    return {k: v for k, v in receipt.items()
            if k not in POSTPUBLICATION_FAILURE_FIELDS}


def inspect_native_training_evidence(
    directory: Path, archive: Path | None = None,
) -> dict:
    """Return an honest status; never modify the training folder or ZIP."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Training evidence directory missing or symlinked")
    local = _local_receipt(directory)

    archive_report = None
    archived = None
    if archive is not None:
        archive = Path(archive)
        if archive.is_symlink():
            raise ValueError("Unsafe ZIP symlink")
        # Always use full safetensors/header/hash and ZIP traversal checks;
        # a ZIP with a 'completed' JSON alone must not be considered valid.
        archive_report = inspect_training_archive(archive)
        with zipfile.ZipFile(archive, "r") as z:
            if tuple(z.namelist()) != EXPECTED_MEMBERS:
                raise ValueError("Native ZIP does not have exactly three members")
            info = z.getinfo("run_info.json")
            if not 0 < info.file_size <= MAX_LOCAL_RECEIPT:
                raise ValueError("Native ZIP training receipt oversized")
            archived = _parse_json(z.read("run_info.json"), "archive training receipt")
        if archived.get("status") != "completed":
            raise ValueError("Archived training receipt is not completed")
        # Non-mutating comparison of the local adapter to the ZIP's recorded
        # content hashes. An archive cannot upgrade a mismatched local model.
        local_hashes = _safe_local_adapter_hashes(directory)
        if local_hashes is not None:
            zip_hashes = {
                EXPECTED_MEMBERS[0]: archived.get("adapter_sha256", {}).get(
                    "adapter_model.safetensors"),
                EXPECTED_MEMBERS[1]: archived.get("adapter_config_sha256"),
            }
        else:
            zip_hashes = None
    else:
        local_hashes = None
        zip_hashes = None

    status = "UNCLASSIFIED"
    match = False
    if archive_report is None:
        if local is None:
            status = "NO_LOCAL_OR_ARCHIVED_EVIDENCE"
        else:
            status = {
                "in_progress": "INTERRUPTED_OR_RUNNING_NO_ZIP",
                "packaging": "PACKAGING_UNPUBLISHED_NO_ZIP",
                "failed": "FAILED_NO_ZIP",
                "archive_published_local_receipt_failed": "PUBLISHED_ZIP_NOT_SUPPLIED",
                "completed": "CLAIMED_COMPLETE_ZIP_NOT_SUPPLIED",
            }.get(local.get("status"), "UNKNOWN_LOCAL_RECEIPT_STATE")
    elif local is None:
        status = "ARCHIVE_ONLY_NO_LOCAL_RECEIPT"
    elif local_hashes is None or local_hashes != zip_hashes:
        status = "LOCAL_ADAPTER_HASH_CONFLICT_OR_MISSING"
    elif local == archived:
        # Matching metadata is insufficient if the ZIP holds a toy/malformed
        # geometry instead of Tiny Aya's 288 LoRA tensors/3,833,856 elements.
        match = True
        status = (
            "MATCHED_SELF_REPORTED_COMPLETION"
            if archive_report["expected_tiny_aya_lora_geometry"]
            else "MATCHED_RECEIPTS_WRONG_ADAPTER_GEOMETRY"
        )
    elif (local.get("status") in {
        "packaging", "archive_published_local_receipt_failed", "failed",
    } and _immutable_receipt_view(local) == _immutable_receipt_view(archived)):
        status = "ARCHIVE_VALID_LOCAL_RECEIPT_UNRECONCILED"
    else:
        status = "RECEIPT_CONTRADICTION"

    # Even perfect local/archive consistency is not independent GPU proof.
    return {
        "status": status,
        "local_status": local.get("status") if local is not None else None,
        "archive_status": archived.get("status") if archived is not None else None,
        "archive_sha256": (
            archive_report["archive_sha256"]
            if archive_report is not None else None
        ),
        "archive_structural_integrity_checked": archive_report is not None,
        "archive_tiny_aya_geometry_matches": (
            archive_report["expected_tiny_aya_lora_geometry"]
            if archive_report is not None else False
        ),
        "local_and_archived_receipts_identical": match,
        "local_adapter_hashes_match_archive": (
            local_hashes == zip_hashes
            if archive_report is not None and local_hashes is not None
            else False
        ),
        "safe_for_unreviewed_published_model_claim": False,
        "independent_gpu_training_verified": False,
        "independent_semantic_quality_verified": False,
        "owner_review_required": not (
            match and archive_report is not None
            and archive_report["expected_tiny_aya_lora_geometry"]
        ),
        "guidance": (
            "Local and archived evidence match internally with expected "
            "Tiny Aya adapter tensor geometry. Preserve both "
            "original artifacts and raw Colab logs. CPU integrity cannot "
            "prove optimizer execution or response correctness."
            if (match and archive_report is not None
                and archive_report["expected_tiny_aya_lora_geometry"]) else
            "Do not retrain or overwrite existing files to fix this state. "
            "Preserve local receipt, adapter directory, ZIP (if any), and "
            "raw Colab logs. Investigate the discrepancy before sharing "
            "a model-completion claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--archive")
    arguments = parser.parse_args()
    print(json.dumps(inspect_native_training_evidence(
        Path(arguments.directory),
        Path(arguments.archive) if arguments.archive else None,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
