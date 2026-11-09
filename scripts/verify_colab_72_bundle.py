"""Read-only verification of an owner's Stage 12 Colab original-72 evidence ZIP.

The private Stage 12 notebook emits six source entries, including original
saved raw responses and a 3-file native AuditKIT 1.1.3 evaluation. This tool
checks an EXACT ZIP inventory, bounded regular members, the exact frozen
dataset SHA-256, recorded no-training/no-review status, all 72 saved response
IDs, native source-bound scoring/receipt hashes and a fresh CPU AuditKIT
replay. No original source ZIP is extracted into the project and no results
are auto-published, trained or sent to an external service.

CRITICAL: all self-reported GPU metadata and model outputs may have been
fabricated outside our verifier. Passing local source consistency is NOT
independent GPU provenance, semantic human review or Lexsi submission.
"""
from __future__ import annotations

import argparse
from hashlib import sha256 as hashlib_sha256
import json
from pathlib import Path
import re
import stat
import tempfile
from zipfile import BadZipFile, ZipFile, ZIP_STORED

from policyproof.core import ROOT, sha256
from scripts.score_track2a_72 import (
    KNOWN_CHALLENGE_SHA256, MAX_SOURCE_REPORT_BYTES,
    MAX_CHALLENGE_DATASET_BYTES,
)
from scripts.verify_evidence import read_evidence_json
from scripts.verify_track2a_72_receipts import verify_completed_72

# Exact names as exported by notebooks/12_track2a_72_drive_resumable_t4.ipynb.
SOURCE_NAMES = {
    "frozen_72_challenge.jsonl": "artifacts/track2a_challenge_unscored/challenge_v1.jsonl",
    "original_72_raw_predictions.json": "artifacts/track2a_72_unchanged_base_predictions.json",
    "auditkit_72/measured_72_report.json":
        "artifacts/actual_72_case_evaluation/measured_72_report.json",
    "auditkit_72/auditkit_72_receipt.json":
        "artifacts/actual_72_case_evaluation/auditkit_72_receipt.json",
    "auditkit_72/EVIDENCE_COMPLETE.json":
        "artifacts/actual_72_case_evaluation/EVIDENCE_COMPLETE.json",
}
MANIFEST = "OWNER_SCOPE_MANIFEST.json"
EXPECTED = set(SOURCE_NAMES) | {MANIFEST}
SHA40 = re.compile(r"^[a-f0-9]{40}$")
PER_MEMBER_BYTES = {
    "frozen_72_challenge.jsonl": MAX_CHALLENGE_DATASET_BYTES,
    "original_72_raw_predictions.json": MAX_SOURCE_REPORT_BYTES,
    "auditkit_72/measured_72_report.json": 2_000_000,
    "auditkit_72/auditkit_72_receipt.json": 2_000_000,
    "auditkit_72/EVIDENCE_COMPLETE.json": 50_000,
    MANIFEST: 20_000,
}
MAX_ZIP_BYTES = 10_000_000
MAX_TOTAL_UNCOMPRESSED = sum(PER_MEMBER_BYTES.values())


def _metadata(blob: bytes) -> dict:
    """Strictly parse self-reported owner-scope metadata, no duplicate keys."""
    def unique(pairs):
        out = {}
        for k, v in pairs:
            if k in out:
                raise ValueError("Duplicate owner ZIP manifest field: " + k)
            out[k] = v
        return out
    try:
        record = json.loads(
            blob.decode("utf-8"), object_pairs_hook=unique,
            parse_constant=lambda t: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON manifest value: " + t)
            ),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Malformed original Colab owner ZIP manifest") from exc
    if (not isinstance(record, dict)
        or set(record) != {
            "status", "dataset_sha256", "code_commit",
            "original_gpu_provenance_independently_verified",
            "human_semantic_review_complete", "training_performed",
            "official_lexsi_submission_completed",
        }
        or record.get("status") !=
            "OWNER_SAVED_ORIGINAL_72_MODEL_OUTPUTS_REPLAYED_NOT_EXTERNALLY_ATTESTED"
        or record.get("dataset_sha256") != KNOWN_CHALLENGE_SHA256
        or not isinstance(record.get("code_commit"), str)
        or not SHA40.fullmatch(record["code_commit"])
        or any(record[k] is not False for k in (
            "original_gpu_provenance_independently_verified",
            "human_semantic_review_complete", "training_performed",
            "official_lexsi_submission_completed",
        ))):
        raise ValueError("Owner ZIP manifest contradicts the exact original, unreviewed research scope")
    return record


def inspect_bundle(path: Path) -> dict:
    """Verify preserved, original 72-case bundle without writing to source tree."""
    path = Path(path)
    if (path.is_symlink() or not path.is_file()
        or not 0 < path.stat().st_size <= MAX_ZIP_BYTES):
        raise ValueError("Owner Colab evidence ZIP absent, symlinked, empty or oversized")
    original_zip_sha = sha256(path)
    with ZipFile(path, "r") as archive:
        entries = archive.infolist()
        if len(entries) != len(EXPECTED):
            raise ValueError("Original Colab ZIP has missing/extra or duplicate entries")
        contents = {}
        total = 0
        for entry in entries:
            member_type = stat.S_IFMT(entry.external_attr >> 16)
            if (entry.filename not in EXPECTED
                or entry.filename in contents
                or entry.filename.startswith("/")
                or "\\" in entry.filename
                or "\x00" in entry.filename
                or ".." in entry.filename.split("/")
                or entry.is_dir()
                or member_type not in (0, stat.S_IFREG)
                or entry.flag_bits & 1
                or entry.compress_type != ZIP_STORED
                or not 0 < entry.file_size <= PER_MEMBER_BYTES.get(entry.filename, 0)):
                raise ValueError("Unknown, unsafe, duplicated or oversized Colab ZIP entry")
            total += entry.file_size
            if total > MAX_TOTAL_UNCOMPRESSED:
                raise ValueError("Owner ZIP exceeds total size limit")
            contents[entry.filename] = archive.read(entry)
        if set(contents) != EXPECTED or archive.testzip() is not None:
            raise ValueError("Owner ZIP inventory or CRC changed")
    manifest = _metadata(contents[MANIFEST])
    if hashlib_sha256(contents["frozen_72_challenge.jsonl"]).hexdigest() != KNOWN_CHALLENGE_SHA256:
        raise ValueError("Frozen original challenge SHA256 mismatch")

    # Isolate all extracted bytes under a Python-managed temporary directory.
    # No source evidence can be overwritten in the repository.
    with tempfile.TemporaryDirectory(prefix="policyproof_original72_private_replay_") as scratch:
        tmp = Path(scratch)
        for original_name, relative in SOURCE_NAMES.items():
            target = tmp / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents[original_name])

        replay = verify_completed_72(
            "artifacts/actual_72_case_evaluation",
            "artifacts/track2a_challenge_unscored/challenge_v1.jsonl",
            "artifacts/track2a_72_unchanged_base_predictions.json",
            root=tmp,
        )
        if (replay.get("status") != "VERIFIED_LOCAL_72_CASE_SOURCE_REPLAY"
            or replay.get("cases") != 72
            or replay.get("original_gpu_provenance_independently_authenticated") is not False
            or replay.get("human_semantic_quality_verified") is not False):
            raise ValueError("Native 72-case AuditKIT replay not complete or overstated")

        # Verify exactly the preserved ZIP bytes, not a modified original path.
        original_report = read_evidence_json(
            tmp / SOURCE_NAMES["original_72_raw_predictions.json"],
            limit=MAX_SOURCE_REPORT_BYTES,
        )
        if original_report.get("human_semantic_review_completed") is not False:
            raise ValueError("Source report incorrectly claims human semantic review")
        original_revision = original_report.get("model_revision")
        if not isinstance(original_revision, str) or not SHA40.fullmatch(original_revision):
            raise ValueError("Original report missing full model SHA")
        if manifest["dataset_sha256"] != replay["dataset_sha256"]:
            raise ValueError("Owner manifest dataset differs from source-replayed challenge")

    if sha256(path) != original_zip_sha:
        raise ValueError("Owner ZIP changed during independent CPU replay")
    return {
        "status": "VERIFIED_PRIVATE_STAGE12_72_LOCAL_SOURCE_REPLAY_NOT_GPU_ATTESTATION",
        "owner_zip_sha256": original_zip_sha,
        "owner_recorded_code_commit": manifest["code_commit"],
        "owner_recorded_model_revision": original_revision,
        "original_frozen_dataset_sha256": KNOWN_CHALLENGE_SHA256,
        "original_model_report_sha256": replay["original_model_report_sha256"],
        "native_auditkit_version": replay["auditkit_version"],
        "original_raw_saved_responses_source_replayed": 72,
        "zip_members_verified": len(contents),
        "original_zip_changed": False,
        "original_model_inference_performed_by_verifier": False,
        "original_gpu_execution_independently_authenticated": False,
        "human_semantic_quality_verified": False,
        "native_aligntune_training_performed": False,
        "public_hf_publication_performed": False,
        "official_lexsi_submission_completed": False,
        "project_evidence_files_overwritten": False,
        "warning": (
            "All original saved response IDs, frozen source hash, original "
            "AuditKIT receipts and scores replay LOCALLY. Self-declared model "
            "runtime cannot independently prove real GPU inference; a "
            "self-consistent forged archive can still be fabricated. "
            "No independent multilingual semantic review, model improvement, "
            "licensed HF publication or official Lexsi submission is verified."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path,
                        help="Original private Stage 12 ZIP, preserved unchanged")
    args = parser.parse_args()
    try:
        report = inspect_bundle(args.archive)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, BadZipFile) as exc:
        parser.exit(status=2, message="BLOCKED: " + str(exc)[:340] + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
