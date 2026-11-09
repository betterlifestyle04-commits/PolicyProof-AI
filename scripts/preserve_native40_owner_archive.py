"""Preserve an original native AlignTune 40-step ZIP in PRIVATE owner storage.

CPU-only, no Hugging Face access, no optimizer, no model loading, no publication.
This backup is a recovery aid; checksums and a self-reported training receipt do
not independently authenticate CUDA execution or prove model improvements.

Example after the owner separately mounts their PRIVATE Google Drive:
    preserve_native40_archive(
        Path("/content/policyproof_align_native40.zip"),
        Path("/content/drive/MyDrive/PolicyProof_Lexsi_Native40"),
    )
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from scripts.verify_training_archive import ARCHIVE_LIMIT, inspect_training_archive


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _no_symlink_parents(path: Path) -> None:
    """Do not follow symlinked original evidence or backup ancestors."""
    p = Path(path).absolute()
    for candidate in (p, *p.parents):
        if candidate.is_symlink():
            raise ValueError("Native training source/backup path contains a symlink")


def _strict_native40(path: Path) -> dict:
    if (path.is_symlink() or not path.is_file()
        or not 0 < path.stat().st_size <= ARCHIVE_LIMIT):
        raise ValueError("Original native40 archive missing, unsafe or oversized")
    outcome = inspect_training_archive(path)
    if (outcome.get("native_aligntune_40step_receipt_checked") is not True
        or outcome.get("optimizer_steps_self_reported") != 40
        or outcome.get("backend") != "aligntune_wrapper"
        or outcome.get("expected_tiny_aya_lora_geometry") is not True
        or outcome.get("file_hashes_match") is not True
        or outcome.get("training_independently_verified") is not False
        or outcome.get("model_improvement_verified") is not False):
        raise ValueError("Source is not a checked original native AlignTune 40-step archive")
    return outcome


def preserve_native40_archive(original: Path, private_folder: Path) -> dict:
    """Write once into a user-owned private directory, verify bytes and receipt.

    An existing exact matching copy is verified again and reused, NEVER
    replaced. A collision, damaged partial copy, symlink, or changed original
    is a hard error. A crash can leave a partial copy with the hash-bound name;
    it must be inspected by the owner, not overwritten by this workflow.
    """
    original, private_folder = Path(original), Path(private_folder)
    _no_symlink_parents(original)
    _no_symlink_parents(private_folder)
    if original.name != "policyproof_align_native40.zip":
        raise ValueError("Expected original native40 archive filename")
    source_evidence = _strict_native40(original)
    source_digest = _digest(original)
    if source_evidence.get("archive_sha256") != source_digest:
        raise ValueError("Original archive hash mismatches its CPU verifier")
    if private_folder.exists() and not private_folder.is_dir():
        raise ValueError("Private backup destination is not a directory")
    private_folder.mkdir(parents=True, exist_ok=True)
    _no_symlink_parents(private_folder)
    target = private_folder / ("policyproof_align_native40_" + source_digest[:16] + ".zip")
    _no_symlink_parents(target)
    if target.exists():
        if not target.is_file():
            raise ValueError("Existing native40 backup is not a regular file")
        existing = True
        if _digest(target) != source_digest:
            raise FileExistsError("Existing private backup differs; never replace it")
    else:
        existing = False
        # Google Drive's FUSE mount does not guarantee POSIX hardlink support.
        # Exclusive creation enforces no clobber even on the owner mount.
        # On a sudden disconnect, a partial file may remain; it is NEVER
        # recognized as complete and will block later reuse of that filename.
        with original.open("rb") as source, target.open("xb") as out:
            while chunk := source.read(1024 * 1024):
                out.write(chunk)
            out.flush()
            try:
                os.fsync(out.fileno())
            except OSError:
                # Some mounted remote filesystems reject fsync. This is NOT
                # proof of remote durability; full readback is still required.
                pass
    if _digest(original) != source_digest or _digest(target) != source_digest:
        raise ValueError("Original or private backup changed during copying")
    saved_evidence = _strict_native40(target)
    if saved_evidence != source_evidence:
        raise ValueError("Private backup's native evidence does not match original")
    return {
        "status": "PRIVATE_NATIVE40_BACKUP_BYTE_VERIFIED_NOT_GPU_ATTESTED",
        "backup_path": str(target),
        "archive_sha256": source_digest,
        "reused_existing_exact_backup": existing,
        "self_reported_native_aligntune_40steps": True,
        "original_training_gpu_independently_verified": False,
        "model_improvement_verified": False,
        "official_submission_completed": False,
        "published_to_huggingface": False,
        "warning": (
            "Private storage copy was re-read and original adapter integrity "
            "was checked; real GPU execution, trainer steps and model safety "
            "still require original runtime logs and independent assessment. "
            "An attached cloud drive can lose unsynced writes."
        ),
    }
