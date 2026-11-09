"""Build or inspect the EXACT Hugging Face dataset-file preview, offline.

OWNER ACTIONS ARE DIFFERENT:
  - inspect_only: read original CuratorKIT/AuditKIT CI archive; no license.
  - preview_public_dataset: select a TENTATIVE license and download the exact
    files that an approved uploader would stage. NO license/rights approval,
    NO account token, NO Hugging Face access, NO publication or submission.
  - publish_public_dataset: separate workflow run, pinned SHA, explicit legal
    rights and public publication confirmations (handled elsewhere).

Never present an offline preview as an uploaded dataset or an approved license.
No original source or existing ZIP may be overwritten.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from scripts.build_track2a_release_candidate import EXPECTED_FILES, STATUS_FILE
from scripts.publish_track2a_candidate import (
    stage_approved_candidate, validate_public_staging,
)
from scripts.publish_track2a_dataset import LICENSES

PUBLIC_FILES = (EXPECTED_FILES - {STATUS_FILE}) | {"provenance/PUBLICATION_SCOPE.json"}
ZIP_NAME = "track2a_HF_DATASET_PUBLIC_FILES_PREVIEW_NOT_PUBLISHED.zip"
RECEIPT_NAME = "track2a_HF_DATASET_PUBLIC_FILES_PREVIEW_NOT_PUBLISHED.json"
MAX_ARCHIVE_BYTES = 25_000_000
MAX_PUBLIC_FILE_BYTES = 5_000_000
MAX_TOTAL_PUBLIC_BYTES = 25_000_000
SHA = re.compile(r"^[0-9a-f]{64}$")


def _strict_out(output: Path, name: str) -> Path:
    out = Path(output)
    if (out.name != name or out.parent != Path("artifacts")
        or out.is_symlink() or out.exists()
        or (out.parent.exists() and
            (out.parent.is_symlink() or not out.parent.is_dir()))):
        raise ValueError("Use a NEW direct artifacts/" + name + " output; never overwrite")
    return out


def _archive_from_staging(folder: Path, names: list[str]) -> bytes:
    if set(names) != PUBLIC_FILES or len(names) != len(PUBLIC_FILES):
        raise ValueError("Owner preview differs from exact public-file allowlist")
    stream = BytesIO()
    with ZipFile(stream, "w") as zipfile:
        for name in sorted(names):
            raw = (folder / name).read_bytes()
            if not 0 < len(raw) <= MAX_PUBLIC_FILE_BYTES:
                raise ValueError("Owner preview member empty or too large")
            info = ZipInfo(name, (2026, 11, 9, 0, 0, 0))
            info.compress_type = ZIP_STORED
            info.external_attr = 0o100644 << 16
            zipfile.writestr(info, raw)
    blob = stream.getvalue()
    if len(blob) > MAX_ARCHIVE_BYTES:
        raise ValueError("Owner preview ZIP exceeds strict size limit")
    return blob


def inspect_preview_bytes(
    original_archive: Path, digest: str, selected_license: str, blob: bytes,
) -> dict:
    """Independently rebuild approved *staging bytes* and compare every file.

    This is exact byte comparison, NOT independent approval of legal rights.
    """
    if not isinstance(digest, str) or not SHA.fullmatch(digest):
        raise ValueError("Expected pinned lowercase 64-character original source SHA256")
    if selected_license not in LICENSES:
        raise ValueError("Preview requires tentative supported dataset license")
    if not isinstance(blob, bytes) or not 0 < len(blob) <= MAX_ARCHIVE_BYTES:
        raise ValueError("Missing or oversized owner preview bytes")
    with ZipFile(BytesIO(blob)) as z:
        members = z.infolist()
        if len(members) != len(PUBLIC_FILES):
            raise ValueError("Owner preview missing or duplicated public members")
        seen = set()
        values = {}
        total = 0
        for item in members:
            name = item.filename
            file_type = (item.external_attr >> 16) & 0o170000
            if (name not in PUBLIC_FILES or name in seen
                or name.startswith("/") or "\\" in name or ".." in name.split("/")
                or item.is_dir() or item.flag_bits & 1
                or item.compress_type != ZIP_STORED
                or file_type not in (0, 0o100000)
                or not 0 < item.file_size <= MAX_PUBLIC_FILE_BYTES):
                raise ValueError("Unsafe, duplicate or unexpected preview ZIP member")
            total += item.file_size
            if total > MAX_TOTAL_PUBLIC_BYTES:
                raise ValueError("Preview ZIP exceeds decompressed limit")
            seen.add(name)
            values[name] = z.read(item)
        if set(values) != PUBLIC_FILES or z.testzip() is not None:
            raise ValueError("Preview ZIP inventory or CRC corrupt")
    with TemporaryDirectory(prefix="policyproof_owner_preview_verify_") as scratch:
        rebuilt = Path(scratch) / "public"
        result = stage_approved_candidate(
            original_archive, rebuilt,
            license_id=selected_license, expected_digest=digest,
        )
        verify = validate_public_staging(
            rebuilt, result["file_names"],
            source_digest=digest, license_id=selected_license,
        )
        if verify["checksums_verified"] is not True:
            raise ValueError("Original staging SHA manifest failed")
        for name in PUBLIC_FILES:
            if values[name] != (rebuilt / name).read_bytes():
                raise ValueError(
                    "Owner ZIP member changed from the exact original publisher preview: "
                    + name
                )
    return {
        "status": "OWNER_PUBLIC_FILES_PREVIEW_REBUILT_EXACTLY_NO_PUBLICATION",
        "original_source_archive_sha256": digest,
        "owner_preview_archive_sha256": sha256(blob).hexdigest(),
        "tentative_dataset_license_not_approved": selected_license,
        "public_payload_file_count": len(PUBLIC_FILES),
        "all_public_file_bytes_compared_to_source_rebuild": True,
        "all_public_file_checksums_verified": True,
        "rights_approval_obtained_by_script": False,
        "license_rights_independently_verified": False,
        "huggingface_token_used": False,
        "huggingface_upload_performed": False,
        "dataset_collection_created": False,
        "model_collection_created": False,
        "official_lexsi_submission_completed": False,
        "warning": (
            "This ZIP previews EXACT public dataset payload files and one "
            "TENTATIVE license for private rights review. Merely choosing a "
            "preview license does not authorize upload, confirm ownership or "
            "change original research. A separate manually approved workflow "
            "is required for any public publication. Original 48 development "
            "results are source-replayed; 72 challenge rows remain UNSCORED."
        ),
    }


def build_preview(
    source: Path, digest: str, selected_license: str,
) -> tuple[bytes, dict]:
    with TemporaryDirectory(prefix="policyproof_owner_hf_preview_build_") as tmp:
        folder = Path(tmp) / "staging"
        staged = stage_approved_candidate(
            source, folder, license_id=selected_license, expected_digest=digest,
        )
        validate_public_staging(
            folder, staged["file_names"],
            source_digest=digest, license_id=selected_license,
        )
        blob = _archive_from_staging(folder, staged["file_names"])
    # This full readback rebuild is deliberately independent of the initial
    # staging pass and catches changed source, files or invalid ZIP packaging.
    return blob, inspect_preview_bytes(source, digest, selected_license, blob)


def write_preview(
    source: Path, digest: str, selected_license: str,
    *, output: Path = Path("artifacts") / ZIP_NAME,
    receipt: Path = Path("artifacts") / RECEIPT_NAME,
) -> dict:
    output = _strict_out(output, ZIP_NAME)
    receipt = _strict_out(receipt, RECEIPT_NAME)
    blob, report = build_preview(source, digest, selected_license)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as fp:
        fp.write(blob)
    # Inspect copied bytes, *not* only a temporary prewrite buffer.
    independently_checked = inspect_preview_bytes(
        source, digest, selected_license, output.read_bytes()
    )
    if independently_checked != report:
        raise ValueError("Preview changed after exclusive write")
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with receipt.open("x", encoding="utf-8") as fp:
        fp.write(payload)
    return {**report, "owner_preview_zip_path": str(output),
            "owner_preview_receipt_path": str(receipt)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--license", required=True, choices=sorted(LICENSES))
    parser.add_argument("--inspect-zip", type=Path, default=None)
    args = parser.parse_args()
    if args.inspect_zip is not None:
        original = Path(args.inspect_zip)
        if original.is_symlink() or not original.is_file():
            parser.error("Preview ZIP must be a regular original file")
        report = inspect_preview_bytes(
            args.archive, args.expected_sha256,
            args.license, original.read_bytes(),
        )
    else:
        report = write_preview(
            args.archive, args.expected_sha256, args.license,
        )
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
