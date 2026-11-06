"""Protect deterministic CPU research reports from clobbering source evidence.

The caller computes and serializes an exact result; this module can create it
ONCE under artifacts/, or reuse a byte-identical prior result for idempotent
CI/public-demo rebuilds. A different existing payload is NEVER rewritten.

This does not authenticate the original model/GPU experiment or score quality.
"""
from __future__ import annotations

import os
from pathlib import Path
import re

from policyproof.core import ROOT

_REPORT_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}\.json$")
_MAX_REPORT_BYTES = 8_000_000


def deterministic_report_target(output: str | Path, *, root: Path = ROOT) -> Path:
    """Accept only direct artifacts/<safe-name>.json, including an absolute default."""
    project = Path(root).resolve()
    literal = str(output)
    if (not literal or len(literal) > 1024 or "\\" in literal
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in literal)):
        raise ValueError("Report output must be a direct artifacts/<name>.json")
    candidate = Path(literal)
    if candidate.is_absolute():
        if candidate.parent != project / "artifacts":
            raise ValueError("Report output must be a direct artifacts/<name>.json")
        name = candidate.name
    else:
        parts = literal.split("/")
        if len(parts) != 2 or parts[0] != "artifacts":
            raise ValueError("Report output must be a direct artifacts/<name>.json")
        name = parts[1]
    if not _REPORT_FILE.fullmatch(name):
        raise ValueError("Report output must be a direct artifacts/<name>.json")
    return project / "artifacts" / name


def preserve_deterministic_report(
    payload: str, output: str | Path, *, root: Path = ROOT,
) -> tuple[Path, bool]:
    """Create new only, or reuse identical immutable report.

    Returns (path, reused). Disallows stale/different existing evidence and
    existing symlinks; requires callers to create a NEW filename for a changed
    analysis. New writes use exclusive creation even under concurrent runs.
    """
    if not isinstance(payload, str):
        raise TypeError("CPU report payload must be text")
    content = payload.encode("utf-8")
    if not 0 < len(content) <= _MAX_REPORT_BYTES:
        raise ValueError("CPU report payload is empty or exceeds the report cap")
    target = deterministic_report_target(output, root=root)
    folder = target.parent
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise ValueError("Unsafe artifacts report directory")
    folder.mkdir(exist_ok=True)
    if folder.is_symlink():
        raise ValueError("Unsafe artifacts report directory")
    if target.is_symlink():
        raise ValueError("Unsafe existing report symlink")
    if target.exists():
        if (not target.is_file()
            or not 0 < target.stat().st_size <= _MAX_REPORT_BYTES):
            raise ValueError("Existing report is not a bounded regular file")
        if target.read_bytes() != content:
            raise FileExistsError(
                "Existing deterministic report differs from current source; "
                "preserve it and choose a NEW output filename"
            )
        return target, True
    # 'x' is O_EXCL: an existing file or symlink can never be truncated.
    try:
        with target.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        # A concurrent writer won the create race. Preserve that evidence.
        raise FileExistsError("Concurrent report creation; original preserved")
    return target, False
