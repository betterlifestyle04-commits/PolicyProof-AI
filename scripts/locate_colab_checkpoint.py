"""Find the exact checkpoint a Colab upload saved, without moving or reuploading.

google.colab.files.upload() writes into the notebook's current working
directory, which is not always /content. This helper makes the original
archive SHA mandatory even when a file is found in a different directory.

Only disk paths are inspected; no network, GPU, inference or training.
"""
from __future__ import annotations

from pathlib import Path
import re

from scripts.verify_training_archive import digest

SHA = re.compile(r"^[0-9a-f]{64}$")
MAX_ARCHIVE_BYTES = 220_000_000


def locate_verified_checkpoint(
    filename: str,
    expected_sha256: str,
    search_directories: list[Path] | tuple[Path, ...],
) -> Path:
    """Return an existing regular ZIP with the exact approved checksum.

    A missing archive raises FileNotFoundError (the notebook may upload).
    A same-name archive that exists but fails the pinned digest raises
    ValueError (do not silently proceed or treat it as a new checkpoint).
    """
    if (not isinstance(filename, str) or not filename.endswith(".zip")
        or Path(filename).name != filename or "/" in filename
        or chr(92) in filename or chr(0) in filename):
        raise ValueError("Expected a simple .zip checkpoint filename")
    if not isinstance(expected_sha256, str) or not SHA.fullmatch(expected_sha256):
        raise ValueError("Expected the previously verified exact 64-character archive SHA256")
    seen: set[Path] = set()
    wrong_hash: list[str] = []
    for directory in search_directories:
        directory = Path(directory)
        candidate = directory / filename
        if not candidate.exists() and not candidate.is_symlink():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("Uploaded archive path is unsafe: " + str(candidate))
        canonical = candidate.resolve()
        if canonical in seen:
            continue
        seen.add(canonical)
        if not 0 < candidate.stat().st_size <= MAX_ARCHIVE_BYTES:
            raise ValueError("Uploaded archive size is empty or unsafe: " + str(candidate))
        if digest(candidate) == expected_sha256:
            return canonical
        wrong_hash.append(str(candidate))
    if wrong_hash:
        raise ValueError(
            "Found ZIP named " + filename
            + ", but SHA256 differs from the approved checkpoint at "
            + ", ".join(wrong_hash)
        )
    raise FileNotFoundError(
        "No uploaded " + filename + " found in the checked Colab directories"
    )
