"""Produce a private blinded human-review packet directly from a saved dev48 ZIP.

CPU-only. No GPU, model inference, training, external calls, human votes or
semantic correctness assertions. Explicitly pinned archive bytes, complete
inventory and per-member manifest checks precede parsing or disk writes.

The source ZIP is NEVER modified or extracted into the public repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import tempfile
import zipfile

from policyproof.core import sha256
from scripts.dev48_gpu_pair import DEV_SHA
from scripts.paired_semantic_review import HEX64, export_packet

# Known original user-provided archive audited on 2026-11-05. Callers can pin
# another genuine archive by explicitly supplying its independently checked hash.
ORIGINAL_DEV48_ZIP_SHA256 = (
    "e98b06482985a9c2f5dfa62100dccda7421830de8292bae013250ccf4ca1099c"
)
REQUIRED = frozenset({
    "BUNDLE_MANIFEST.json", "dev48_base.json",
    "dev48_tuned.json", "dev48_paired_auditkit.json",
})
MAX_ZIP_BYTES = 8_000_000
MAX_MEMBER_BYTES = 4_000_000
MAX_TOTAL_BYTES = 10_000_000


def _strict_json(data: bytes, label: str) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        result = json.loads(
            data.decode("utf-8-sig"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"{label}: malformed JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label}: expected JSON object")
    return result


def verify_bundle(bundle_path: Path, expected_sha256: str) -> tuple[dict, dict]:
    """Verify an exact four-file ZIP and internal manifest; return bytes+metadata."""
    bundle_path = Path(bundle_path)
    if not isinstance(expected_sha256, str) or not HEX64.fullmatch(expected_sha256):
        raise ValueError("Expected original ZIP SHA-256 must be exactly 64 lowercase hex characters")
    if (bundle_path.is_symlink() or not bundle_path.is_file()
        or not 0 < bundle_path.stat().st_size <= MAX_ZIP_BYTES):
        raise ValueError("Source ZIP is missing, symlinked, empty or oversized")
    actual_sha = sha256(bundle_path)
    if actual_sha != expected_sha256:
        raise ValueError("Source ZIP SHA-256 mismatch: abort without writing reviewer files")
    try:
        with zipfile.ZipFile(bundle_path, "r") as archive:
            entries = archive.infolist()
            names = [x.filename for x in entries]
            if len(entries) != len(REQUIRED) or set(names) != REQUIRED:
                raise ValueError("Unexpected, missing or duplicate ZIP member")
            if sum(x.file_size for x in entries) > MAX_TOTAL_BYTES:
                raise ValueError("ZIP uncompressed total is oversized")
            for info in entries:
                filemode = stat.S_IFMT(info.external_attr >> 16)
                if (info.is_dir() or filemode not in (0, stat.S_IFREG)
                    or info.flag_bits & 0x1
                    or info.file_size <= 0 or info.file_size > MAX_MEMBER_BYTES
                    or info.compress_size <= 0
                    or info.file_size > max(info.compress_size * 1000, 500_000)):
                    raise ValueError(f"Unsafe or oversized ZIP member: {info.filename}")
            contents = {name: archive.read(name) for name in REQUIRED}
    except (zipfile.BadZipFile, EOFError, OSError, RuntimeError) as exc:
        raise ValueError("ZIP is malformed or cannot be safely read") from exc

    manifest = _strict_json(contents["BUNDLE_MANIFEST.json"], "bundle manifest")
    if (manifest.get("schema") != "policyproof_dev48_gpu_pair_bundle_v1"
        or manifest.get("split") != "already_inspected_synthetic_development_only"
        or manifest.get("final_blind_evaluation") is not False
        or manifest.get("model_improvement_semantically_verified") is not False):
        raise ValueError("Archive manifest mislabels development results")
    recorded = manifest.get("files_sha256")
    if not isinstance(recorded, dict) or set(recorded) != REQUIRED - {"BUNDLE_MANIFEST.json"}:
        raise ValueError("Manifest file inventory is missing or unexpected")
    for name, stamp in recorded.items():
        if (not isinstance(stamp, str) or not HEX64.fullmatch(stamp)
            or hashlib.sha256(contents[name]).hexdigest() != stamp):
            raise ValueError(f"Manifest SHA-256 mismatch for {name}")

    base = _strict_json(contents["dev48_base.json"], "base report")
    tuned = _strict_json(contents["dev48_tuned.json"], "tuned report")
    paired = _strict_json(contents["dev48_paired_auditkit.json"], "paired report")
    if (base.get("is_real_model_result") is not True
        or base.get("is_finetuned") is not False
        or tuned.get("is_real_model_result") is not True
        or tuned.get("is_finetuned") is not True
        or base.get("dataset_sha256") != DEV_SHA
        or tuned.get("dataset_sha256") != DEV_SHA
        or base.get("model_revision") != tuned.get("model_revision")
        or base.get("generation_config") != tuned.get("generation_config")
        or paired.get("status") != "paired_deterministic_audit"
        or paired.get("sample_count") != 48
        or paired.get("evaluation_checksum") != DEV_SHA
        or paired.get("is_final_blind_test") is not False
        or paired.get("is_model_improvement_semantically_verified") is not False
        or paired.get("base_model_revision_loaded_on_inference") != base.get("model_revision")
        or paired.get("adapter_sha256") != tuned.get("adapter_sha256")
        or paired.get("training_zip_sha256") != tuned.get("adapter_source_zip_sha256")):
        raise ValueError("Saved base/tuned comparison provenance is inconsistent")
    return contents, {
        "status": "PINNED_DEV48_ZIP_AND_MANIFEST_VERIFIED",
        "source_zip_sha256": actual_sha,
        "source_zip_size_bytes": bundle_path.stat().st_size,
        "dataset_sha256": DEV_SHA,
        "model_revision_reported": base["model_revision"],
        "adapter_sha256_reported": tuned["adapter_sha256"],
        "members_verified": len(contents),
        "independent_gpu_rerun_performed": False,
        "human_semantic_review_performed": False,
        "final_benchmark": False,
    }


def export_from_bundle(root: Path, bundle: Path, expected_sha256: str,
                       packet: str = "reviews/dev48_blank.csv",
                       key: str = "reviews/dev48_private_mapping.json") -> dict:
    """Verify complete bundle first, then use existing fail-closed export."""
    contents, checked = verify_bundle(bundle, expected_sha256)
    with tempfile.TemporaryDirectory(prefix="policyproof-dev48-private-") as tmp:
        directory = Path(tmp)
        base = directory / "dev48_base.json"
        tuned = directory / "dev48_tuned.json"
        # All filenames are fixed; never trust member-supplied output paths.
        base.write_bytes(contents["dev48_base.json"])
        tuned.write_bytes(contents["dev48_tuned.json"])
        result = export_packet(Path(root), base, tuned, packet, key)
    return {**checked, **result, "source_zip_sha256": checked["source_zip_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256", default=ORIGINAL_DEV48_ZIP_SHA256,
        help="Trusted ZIP hash; defaults to the original audited Nov 5 archive",
    )
    parser.add_argument("--packet", default="reviews/dev48_blank.csv")
    parser.add_argument("--key", default="reviews/dev48_private_mapping.json")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Validate ZIP and report provenance but create no files",
    )
    args = parser.parse_args()
    if args.verify_only:
        _, result = verify_bundle(args.bundle, args.expected_sha256)
    else:
        result = export_from_bundle(args.root, args.bundle, args.expected_sha256,
                                    args.packet, args.key)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
