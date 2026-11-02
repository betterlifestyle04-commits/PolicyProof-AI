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
from zipfile import ZipFile, BadZipFile

REQUIRED = {
    "manifest.json", "checksums.txt", "lexsi_provenance.json",
    "rejected.jsonl", "sft_alpaca.jsonl", "sft_sharegpt.jsonl",
}
LANGS = {"English", "Norwegian", "Arabic"}
TEMPLATE = re.compile(r"Answer in (English|Norwegian|Arabic)\.")
SOURCES = re.compile(r"\[(POL-[A-Za-z0-9-]+)\]")


def check_zip(path: str | Path) -> dict:
    issues = []
    with ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive entry names")
        if any(n.startswith("/") or ".." in Path(n).parts or "\\" in n for n in names):
            raise ValueError("Unsafe archive entry path")
        if not REQUIRED.issubset(names):
            raise ValueError("Missing required files: " + ", ".join(sorted(REQUIRED - set(names))))
        if any(z.getinfo(n).file_size > 20_000_000 for n in names):
            raise ValueError("Unexpectedly large file in small synthetic curation archive")
        checksums = {}
        for line in z.read("checksums.txt").decode("utf-8").splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not re.fullmatch(r"[a-f0-9]{64}", parts[0]):
                raise ValueError("Invalid checksums.txt line")
            expected, filename = parts
            if filename in checksums or filename not in names:
                raise ValueError("Checksum entry missing or duplicated")
            checksums[filename] = expected
            if sha256(z.read(filename)).hexdigest() != expected:
                raise ValueError(f"Checksum verification failed: {filename}")
        missing_checksums = (REQUIRED - {"checksums.txt"}) - set(checksums)
        if missing_checksums:
            raise ValueError("Missing checksums: " + str(sorted(missing_checksums)))
        manifest = json.loads(z.read("manifest.json"))
        provenance = json.loads(z.read("lexsi_provenance.json"))
        alpaca = [json.loads(line) for line in z.read("sft_alpaca.jsonl").decode("utf-8").splitlines() if line.strip()]
        share = [json.loads(line) for line in z.read("sft_sharegpt.jsonl").decode("utf-8").splitlines() if line.strip()]
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
                target = json.loads(row["output"])
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
