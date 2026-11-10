"""Build the ORIGINAL verified 78-row CuratorKIT train ZIP for native Colab.

No model, HF token, GPU, optimizer, holdout, 72-case challenge, or publication.
The exact original training bytes are immutable (EXPECTED_SHA). Only the
official CuratorKIT 0.1.22 train-only export is accepted as input.

Usage after official 'scripts.prepare' and 'scripts.run_curator':
    python -m scripts.build_native_train_owner_zip \
        --source artifacts/curator_train_dual_export \
        --output artifacts/policyproof_curatorkit_train_v1.zip

The ZIP must then be uploaded ONLY into the owner-authorized private Stage 11
Colab session. Never claim this data-only ZIP is a trained model checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

from policyproof.core import ROOT
from scripts.check_align_config import check_train_data
from scripts.restore_colab_session import EXPECTED_SHA, ALLOWLIST
from scripts.verify_curator_archive import check_zip, check_dual_export_consistency

SOURCE_DIR = Path("artifacts/curator_train_dual_export")
OUTPUT = Path("artifacts/policyproof_curatorkit_train_v1.zip")
REQUIRED = frozenset({
    "sft_alpaca.jsonl", "sft_sharegpt.jsonl", "manifest.json",
    "lexsi_provenance.json", "checksums.txt", "rejected.jsonl",
})
MAX_INPUT_FILE_BYTES = 20_000_000
MAX_ZIP_BYTES = 35_000_000


def _strict_under_artifacts(root: Path, rel: str | Path) -> Path:
    """Only direct artifacts/<simple-name>, without symlink traversal."""
    text = str(rel)
    if (not text or len(text) > 230 or "\\" in text
        or any(ord(c) < 32 or ord(c) == 127 for c in text)):
        raise ValueError("Expected direct artifacts/<name> path")
    path = Path(text)
    if path.is_absolute():
        if path.parent != root / "artifacts":
            raise ValueError("Path must be direct artifacts/<name>")
        name = path.name
    else:
        parts = text.split("/")
        if len(parts) != 2 or parts[0] != "artifacts":
            raise ValueError("Path must be direct artifacts/<name>")
        name = parts[1]
    if (not name or name in (".", "..")
        or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in name)
        or name.startswith(".")):
        raise ValueError("Unsafe artifact name")
    target = root / "artifacts" / name
    parent = target.parent
    if (parent.is_symlink() or (parent.exists() and not parent.is_dir())
        or target.is_symlink()):
        raise ValueError("Unsafe symlinked artifact path")
    return target


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_owner_training_zip(*, root: Path = ROOT,
                             source: str | Path = SOURCE_DIR,
                             output: str | Path = OUTPUT) -> dict:
    root = Path(root).resolve()
    src = _strict_under_artifacts(root, source)
    dest = _strict_under_artifacts(root, output)
    if src == dest or not src.is_dir() or dest.exists():
        raise FileExistsError("Missing CuratorKIT source folder or output already exists")
    file_names = sorted(ALLOWLIST)
    if not REQUIRED.issubset(file_names):
        raise ValueError("Internal CuratorKIT allowlist missing essential files")
    present = []
    for name in file_names:
        path = src / name
        if path.is_symlink():
            raise ValueError("Symlinked CuratorKIT source file rejected")
        if not path.exists():
            if name in REQUIRED:
                raise ValueError("Missing required CuratorKIT export: " + name)
            continue
        if not path.is_file() or path.stat().st_size > MAX_INPUT_FILE_BYTES:
            raise ValueError("Oversized or unsafe CuratorKIT export: " + name)
        present.append(name)

    # Independently verify actual 78-row source bytes before exporting
    # anything. No new evaluation, test, or challenge examples can enter.
    dataset = check_train_data(src / "sft_alpaca.jsonl")
    if dataset["dataset_sha256"] != EXPECTED_SHA or dataset["row_count"] != 78:
        raise ValueError("CuratorKIT train data do not match original pinned 78 rows")
    from policyproof.core import read_jsonl
    check_dual_export_consistency(
        read_jsonl(src / "sft_alpaca.jsonl"),
        read_jsonl(src / "sft_sharegpt.jsonl"),
    )
    before = {name: _file_sha(src / name) for name in present}
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=dest.parent, delete=False,
            prefix=".policyproof-original-train-", suffix=".partial"
        ) as handle:
            tmp = Path(handle.name)
        with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            for name in present:
                z.write(src / name, arcname=name)
        if not 0 < tmp.stat().st_size <= MAX_ZIP_BYTES:
            raise ValueError("ZIP size exceeds original CuratorKIT transport bound")
        # This checks CuratorKIT provenance, 78 rows, both output formats,
        # manifest counts, and ALL recorded hashes — not merely ZIP CRC.
        checked = check_zip(tmp)
        if checked["status"] != "pass" or checked["row_count"] != 78:
            raise ValueError("Original CuratorKIT ZIP did not pass strict verifier")
        if {name: _file_sha(src / name) for name in present} != before:
            raise ValueError("Source changed during CuratorKIT archive creation")
        archive_sha = _file_sha(tmp)
        # No overwrite on concurrent jobs, unlike os.replace.
        if dest.exists() or dest.is_symlink():
            raise FileExistsError("Original training ZIP already exists; refusing overwrite")
        os.link(tmp, dest)
        return {
            "status": "ORIGINAL_78_CURATORKIT_TRAIN_ZIP_VERIFIED_NO_MODEL_TRAINING",
            "archive": str(dest),
            "archive_sha256": archive_sha,
            "original_training_rows": checked["row_count"],
            "original_training_sha256": dataset["dataset_sha256"],
            "source_files": present,
            "tool": "CuratorKIT",
            "actual_gpu_training_executed": False,
            "model_weights_included": False,
            "held_out_48_60_72_examples_included": False,
            "license_and_model_download_approved_by_this_script": False,
        }
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE_DIR))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    print(json.dumps(build_owner_training_zip(source=args.source, output=args.output),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
