"""Optional append-only checkpoint for unfinished 72-case Tiny Aya GPU inference.

A checkpoint is PARTIAL ORIGINAL-OUTPUT EVIDENCE, never proof of a completed
72-case experiment. Each raw response is bound to a frozen dataset SHA,
unchanged-base model commit, exact case position, and observed loaded commit.
Existing checkpoint bytes are never rewritten, even on a resume.

This module performs no model inference or training; a self-reported runtime
record is not independently authenticated GPU provenance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256 as hashlib_sha256
from pathlib import Path
import re

from policyproof.core import ROOT
from scripts.frozen_pair_inference import GENERATION, MODEL_ID
from scripts.paired_run_recovery import write_evidence_once
from scripts.verify_evidence import read_evidence_json

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
SCHEMA = "policyproof_challenge72_partial_gpu_checkpoint_v1"
MAX_CASE_BYTES = 50_000
MAX_MARKER_BYTES = 30_000


def folder_path(root: Path, value: str | Path) -> Path:
    """Keep checkpoints in an unambiguous, direct artifacts/<name>/ folder."""
    root = Path(root).resolve()
    literal = str(value)
    if (not literal or len(literal) > 1024
        or any(ord(c) < 32 or ord(c) == 127 for c in literal)):
        raise ValueError("Checkpoint must be a direct artifacts/<name> directory")
    path = Path(literal)
    if path.is_absolute():
        if path.parent != root / "artifacts":
            raise ValueError("Checkpoint must be a direct artifacts/<name> directory")
        name = path.name
    else:
        parts = literal.split("/")
        if "\\" in literal or len(parts) != 2 or parts[0] != "artifacts":
            raise ValueError("Checkpoint must be a direct artifacts/<name> directory")
        name = parts[1]
    if not NAME.fullmatch(name):
        raise ValueError("Unsafe checkpoint directory name")
    directory = root / "artifacts" / name
    if directory.parent.is_symlink() or (
        directory.parent.exists() and not directory.parent.is_dir()
    ) or directory.is_symlink() or (
        directory.exists() and not directory.is_dir()
    ):
        raise ValueError("Unsafe checkpoint directory or symlink")
    return directory


def _manifest(rows: list[dict], digest: str, revision: str) -> dict:
    if (len(rows) != 72 or not SHA64.fullmatch(digest)
        or not isinstance(revision, str) or not SHA40.fullmatch(revision)):
        raise ValueError("Checkpoint requires exactly 72 cases and pinned hashes")
    return {
        "schema": SCHEMA,
        "status": "PARTIAL_UNVERIFIED_MODEL_RESPONSES_NOT_FINAL_RESULTS",
        "model": MODEL_ID,
        "base_model_revision": revision,
        "frozen_dataset_sha256": digest,
        "generation_config": dict(GENERATION),
        "ordered_ids": [r["id"] for r in rows],
        "original_gpu_provenance_independently_verified": False,
        "training_performed": False,
        "full_evaluation_completed": False,
    }


def read_checkpoint(
    root: Path, value: str | Path, rows: list[dict], digest: str,
    revision: str, *, resume: bool,
) -> tuple[Path, dict[str, str], dict | None]:
    """Read-only strict preflight; only resume allows an existing checkpoint."""
    folder = folder_path(root, value)
    if not resume:
        if folder.exists() or folder.is_symlink():
            raise FileExistsError("Checkpoint already exists; --resume is required and cannot overwrite it")
        return folder, {}, None
    if not folder.is_dir():
        raise ValueError("Resume requires an existing checkpoint directory")
    expected = _manifest(rows, digest, revision)
    info = read_evidence_json(folder / "manifest.json", limit=MAX_MARKER_BYTES)
    if info != expected:
        raise ValueError("Checkpoint manifest does not match frozen dataset or pinned model revision")
    names = {p.name for p in folder.iterdir()}
    names.discard("manifest.json")
    if len(names) > 72:
        raise ValueError("Too many checkpoint files")
    expected_names = {f"case_{i:03}.json" for i in range(len(names))}
    if names != expected_names:
        raise ValueError("Checkpoint files must form a unique, uninterrupted case prefix")
    outputs = {}
    last_runtime = None
    for i in range(len(names)):
        row = rows[i]
        p = folder / f"case_{i:03}.json"
        case = read_evidence_json(p, limit=MAX_CASE_BYTES)
        runtime = case.get("runtime")
        raw = case.get("raw")
        if (not isinstance(raw, str) or len(raw) > 10000
            or not isinstance(runtime, dict)
            or runtime.get("loaded_model_revision_verified") is not True
            or runtime.get("loaded_model_revision_observed") != revision
            or not isinstance(runtime.get("gpu"), str) or not runtime["gpu"]
            or not isinstance(runtime.get("torch_version"), str)
            or not runtime["torch_version"]):
            raise ValueError("Invalid original per-case raw response or observed GPU runtime")
        expected_case = {
            "schema": SCHEMA, "index": i, "id": row["id"], "raw": raw,
            "raw_sha256": hashlib_sha256(raw.encode("utf-8")).hexdigest(),
            "frozen_dataset_sha256": digest,
            "model_revision": revision,
            "generation_config": dict(GENERATION),
            "runtime": runtime,
            "full_evaluation_completed": False,
            "gpu_provenance_independently_verified": False,
        }
        if case != expected_case:
            raise ValueError("Checkpoint case differs from its frozen identity or raw SHA256")
        outputs[row["id"]] = raw
        last_runtime = runtime
    return folder, outputs, last_runtime


def create_checkpoint(
    root: Path, value: str | Path, rows: list[dict],
    digest: str, revision: str,
) -> Path:
    folder, _, _ = read_checkpoint(
        root, value, rows, digest, revision, resume=False
    )
    folder.parent.mkdir(exist_ok=True)
    # Exclusive creation: never reset or replace previous checkpoints.
    folder.mkdir(exist_ok=False)
    write_evidence_once(folder / "manifest.json", _manifest(rows, digest, revision))
    return folder


def save_case(
    folder: Path, *, index: int, row: dict, raw: str,
    digest: str, revision: str, runtime: dict,
) -> None:
    if (not folder.is_dir() or folder.is_symlink()
        or type(index) is not int or not 0 <= index < 72
        or not isinstance(raw, str) or len(raw) > 10000
        or not isinstance(runtime, dict)
        or runtime.get("loaded_model_revision_verified") is not True
        or runtime.get("loaded_model_revision_observed") != revision
        or not isinstance(runtime.get("gpu"), str) or not runtime["gpu"]
        or not isinstance(runtime.get("torch_version"), str)
        or not runtime["torch_version"]):
        raise ValueError("Refusing unverified or malformed checkpoint response")
    # Per-case immutable evidence. The loader verifies these original bytes
    # and the exact expected ordered case before resuming them.
    write_evidence_once(folder / f"case_{index:03}.json", {
        "schema": SCHEMA,
        "index": index, "id": row["id"], "raw": raw,
        "raw_sha256": hashlib_sha256(raw.encode("utf-8")).hexdigest(),
        "frozen_dataset_sha256": digest,
        "model_revision": revision,
        "generation_config": dict(GENERATION),
        "runtime": runtime,
        "full_evaluation_completed": False,
        "gpu_provenance_independently_verified": False,
    })
