"""Validate a CuratorKIT ZIP without extracting or trusting the archive.

Requires only the Python standard library. This verifies transport integrity,
dataset shape, training-only source IDs and source-label consistency. It does
NOT certify semantic quality, safe model behavior or proper licensing.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import stat
from zipfile import ZipFile, BadZipFile

REQUIRED = {
    "manifest.json", "checksums.txt", "lexsi_provenance.json",
    "rejected.jsonl", "sft_alpaca.jsonl", "sft_sharegpt.jsonl",
}
LANGS = {"English", "Norwegian", "Arabic"}
TEMPLATE = re.compile(r"Answer in (English|Norwegian|Arabic)\.")
SOURCES = re.compile(r"\[(POL-[A-Za-z0-9-]+)\]")

# Bound untrusted archive metadata before any checksum, JSON, or training read.
MAX_ARCHIVE_BYTES = 35_000_000
MAX_MEMBERS = 80
MAX_UNCOMPRESSED_BYTES = 60_000_000
MAX_MEMBER_BYTES = 20_000_000
MAX_COMPRESSION_RATIO = 1500
MAX_CHECKSUM_BYTES = 100_000
MAX_MANIFEST_BYTES = 1_000_000
MAX_PROVENANCE_BYTES = 250_000


def _safe_inventory(z: ZipFile) -> dict:
    """Reject excessive expansion, malicious names, symlinks and devices."""
    entries = z.infolist()
    if not 0 < len(entries) <= MAX_MEMBERS:
        raise ValueError("Unexpected ZIP member count")
    names = [entry.filename for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive entry names")
    file_names = set()
    total = 0
    for member in entries:
        name = member.filename
        core = name[:-1] if name.endswith("/") else name
        parts = core.split("/")
        if (not core or len(name) > 240 or name.startswith("/")
            or "\\" in name or ":" in name or chr(0) in name
            or any(ord(ch) < 32 for ch in name)
            or len(parts) > 8
            or any(not part or part in {".", ".."} for part in parts)
            or member.file_size < 0 or member.compress_size < 0):
            raise ValueError("Unsafe archive entry path or ZIP metadata")
        kind = stat.S_IFMT(member.external_attr >> 16)
        if member.create_system == 3 and kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError("Unsafe archive symlink or special file")
        if member.is_dir():
            if member.file_size != 0:
                raise ValueError("Archive directory contains data")
            continue
        if core in file_names:
            raise ValueError("Duplicate normalized file paths")
        file_names.add(core)
        if (member.file_size > MAX_MEMBER_BYTES
            or (member.compress_size == 0 and member.file_size > 0)
            or (member.compress_size > 0 and
                member.file_size / member.compress_size > MAX_COMPRESSION_RATIO)):
            raise ValueError("Unsafe ZIP expansion ratio or oversized member")
        total += member.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Unsafe ZIP total uncompressed size")
    if not REQUIRED.issubset(file_names):
        raise ValueError("Missing required files: " + ", ".join(sorted(REQUIRED - file_names)))
    for name, limit in {
        "checksums.txt": MAX_CHECKSUM_BYTES,
        "manifest.json": MAX_MANIFEST_BYTES,
        "lexsi_provenance.json": MAX_PROVENANCE_BYTES,
    }.items():
        if z.getinfo(name).file_size > limit:
            raise ValueError("Oversized CuratorKIT provenance/checksum metadata")
    return {"members": len(entries), "uncompressed_bytes": total}


def _strict_json(raw: str, label: str):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("Duplicate JSON field in " + label)
            obj[key] = value
        return obj
    return json.loads(
        raw, object_pairs_hook=unique,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError("Nonfinite JSON value in " + label + ": " + value)
        ),
    )


def _sha_member(z: ZipFile, filename: str) -> str:
    observed = sha256()
    count = 0
    with z.open(filename) as reader:
        while chunk := reader.read(1024 * 1024):
            count += len(chunk)
            if count > MAX_MEMBER_BYTES:
                raise ValueError("Member exceeds ZIP size budget during read")
            observed.update(chunk)
    if count != z.getinfo(filename).file_size:
        raise ValueError("Member length changed during ZIP read")
    return observed.hexdigest()



def check_zip(path: str | Path) -> dict:
    path = Path(path)
    if (not path.is_file() or path.is_symlink()
        or not 0 < path.stat().st_size <= MAX_ARCHIVE_BYTES):
        raise ValueError("Original CuratorKIT ZIP missing, symlinked or oversized")
    with ZipFile(path) as z:
        archive = _safe_inventory(z)
        names = z.namelist()
        checksums = {}
        for line in z.read("checksums.txt").decode("utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not re.fullmatch(r"[a-f0-9]{64}", parts[0]):
                raise ValueError("Invalid checksums.txt line")
            expected, filename = parts
            if filename in checksums or filename not in names:
                raise ValueError("Checksum entry missing or duplicated")
            checksums[filename] = expected
            if z.getinfo(filename).is_dir():
                raise ValueError("Checksum entry refers to directory")
            if _sha_member(z, filename) != expected:
                raise ValueError(f"Checksum verification failed: {filename}")
        missing_checksums = (REQUIRED - {"checksums.txt"}) - set(checksums)
        if missing_checksums:
            raise ValueError("Missing checksums: " + str(sorted(missing_checksums)))
        manifest = _strict_json(z.read("manifest.json").decode("utf-8"), "manifest.json")
        provenance = _strict_json(
            z.read("lexsi_provenance.json").decode("utf-8"), "lexsi_provenance.json"
        )
        if not isinstance(manifest, dict) or not isinstance(provenance, dict):
            raise ValueError("CuratorKIT metadata must be JSON objects")
        alpaca = [
            _strict_json(line, "sft_alpaca.jsonl")
            for line in z.read("sft_alpaca.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
        share = [
            _strict_json(line, "sft_sharegpt.jsonl")
            for line in z.read("sft_sharegpt.jsonl").decode("utf-8").splitlines()
            if line.strip()
        ]
        if len(alpaca) != len(share) or not alpaca:
            raise ValueError("Alpaca / ShareGPT export counts missing or mismatched")
        exported = manifest.get("stage_counts", {}).get("AlpacaExporter", {}).get("exported_count")
        if exported != len(alpaca):
            raise ValueError("Manifest exported count differs from actual rows")
        if provenance.get("library") != "curatorkit":
            raise ValueError("Expected official CuratorKIT provenance metadata")
        languages, labels, unique, abstentions = Counter(), Counter(), set(), 0
        for index, row in enumerate(alpaca):
            if set(row) != {"instruction", "input", "output"} or not all(isinstance(v, str) for v in row.values()):
                raise ValueError(f"Malformed Alpaca export row {index}")
            inst = row["instruction"]
            matches = TEMPLATE.findall(inst)
            if len(matches) != 1:
                raise ValueError(f"Missing/ambiguous language metadata in row {index}")
            languages[matches[0]] += 1
            if inst in unique:
                raise ValueError("Duplicate instructions in the training export")
            unique.add(inst)
            sources = set(SOURCES.findall(inst))
            if not sources or any(not re.fullmatch(r"POL-\d{3}", sid) for sid in sources):
                raise ValueError("Unexpected non-training policy ID in training instructions")
            try:
                target = _strict_json(row["output"], "Alpaca target")
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed reference JSON at row {index}") from exc
            if set(target) != {"answer", "source_id", "abstain"} or type(target["abstain"]) is not bool:
                raise ValueError(f"Incorrect target schema at row {index}")
            if target["abstain"]:
                if target["source_id"] is not None or target["answer"] != "INSUFFICIENT_EVIDENCE":
                    raise ValueError(f"Invalid reference abstention at row {index}")
                abstentions += 1
            elif target["source_id"] not in sources:
                raise ValueError(f"Reference cites a policy not present in evidence at row {index}")
            labels[target["source_id"] or "ABSTAIN"] += 1
        if set(languages) != LANGS:
            raise ValueError("Unexpected language coverage")
        if not all(isinstance(obj, dict) and isinstance(obj.get("conversations"), list) for obj in share):
            raise ValueError("Malformed ShareGPT export")
        # CuratorKIT dataset README advertises non-authoritative export hashes.
        # checksums.txt is the archive-integrity source of truth.
        return {
            "status": "pass",
            "bounded_zip_inventory": archive,
            "duplicate_json_keys_rejected": True,
            "zip_special_entries_rejected": True,
            "training_performed": False,
            "tool_version": manifest.get("tool_versions", {}).get("curatorkit"),
            "row_count": len(alpaca),
            "languages": dict(languages),
            "abstentions": abstentions,
            "policy_labels": dict(sorted(labels.items())),
            "verified_files": sorted(checksums),
            "source_files_logged": manifest.get("source_files"),
            "generation_or_semantic_validation": False,
            "human_reviewed": False,
            "limits": [
                "All data are fictional and template-generated",
                "Exact duplicates only: near-duplicates and semantic contradictions unchecked",
                "Not proof of real-world safety or multilingual quality",
                "Only training IDs permitted; held-out policies cannot enter training",
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    report = check_zip(args.archive)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
