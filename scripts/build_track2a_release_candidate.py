"""Build a checksum-verified, explicitly UNPUBLISHED Track 2a evidence handoff.

Uses only repository-owned synthetic source files + actual, already generated
CuratorKIT/AuditKIT artifacts. This is NOT the historical 20-file owner-reviewed
ZIP and is NOT accepted by its hash-pinned publisher. It is a fresh versioned
candidate for OWNER review and future license decision.

Default: read-only input preflight. --build: create a reproducible ZIP from
existing inputs, with fixed ordering/metadata; never runs a GPU, model, HF
publisher or LLM. --inspect: validate an existing ZIP without extracting.
"""
from __future__ import annotations

import argparse
from hashlib import sha256 as digest
import json
import math
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile, ZipInfo, ZIP_STORED

from policyproof.core import ROOT, aggregate, score, sha256
from scripts.auditkit_replay import check_saved_outputs
from scripts.generate_track2a_challenge import build_challenge, validate_challenge
from scripts.track2a_chain import match_curated, validate_benchmark


VERSION = "policyproof_track2a_unpublished_research_v2"
EXPECTED_BASELINE_SHA = "d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c"
EXPECTED_CHALLENGE_SHA = "cdd5cf4e54caa6c79ce1f0cd5bd0f05344f0dc22039a2ad56a3c1c0ee2d32cd1"
PINNED_TOOLKITS = {"curatorkit_version": "0.1.22", "auditkit_version": "1.1.3"}
SINGLE_LIMIT = 5_000_000
TOTAL_LIMIT = 25_000_000
BUNDLE_LIMIT = 26_000_000

FILES = {
    "README.md": "templates/huggingface/track2a_dataset_README.md",
    "evaluated_48/benchmark_48.jsonl": "data/unseen_policy_test.jsonl",
    "evaluated_48/original_saved_tiny_aya_base_model_predictions.json":
        "artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json",
    "evaluated_48/curator_input.jsonl": "artifacts/track2a_safety/curator_input.jsonl",
    "evaluated_48/curator_export.jsonl":
        "artifacts/track2a_safety/curator_output/sft_alpaca.jsonl",
    "evaluated_48/curator_manifest.json":
        "artifacts/track2a_safety/curator_output/manifest.json",
    "evaluated_48/curator_checksums.txt":
        "artifacts/track2a_safety/curator_output/checksums.txt",
    "evaluated_48/curator_audit_chain_report.json":
        "artifacts/track2a_safety/track2a_chain_report.json",
    "evaluated_48/auditkit_native_receipt.json":
        "artifacts/track2a_safety/auditkit_native_receipt.json",
    "challenge_72/challenge_v1_UNSCORED.jsonl":
        "artifacts/track2a_challenge_unscored/challenge_v1.jsonl",
    "challenge_72/challenge_manifest_UNSCORED.json":
        "artifacts/track2a_challenge_unscored/challenge_manifest.json",
    "challenge_72/curator_export_UNSCORED.jsonl":
        "artifacts/track2a_challenge_unscored/challenge_curator_output/sft_alpaca.jsonl",
    "docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md":
        "docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md",
}
STATUS_FILE = "provenance/PACKAGE_STATUS.json"
HASH_FILE = "provenance/SHA256SUMS.txt"
EXPECTED_FILES = set(FILES) | {STATUS_FILE, HASH_FILE}
SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(rb"(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{35,}"),
)


def _strict_pairs(pairs):
    obj = {}
    for k, v in pairs:
        if k in obj:
            raise ValueError("Duplicated JSON key: " + k)
        obj[k] = v
    return obj


def _json_bytes(data: bytes, label: str) -> dict:
    try:
        value = json.loads(
            data.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Malformed " + label + " JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object in " + label)
    return value


def _jsonl_bytes(data: bytes, label: str) -> list[dict]:
    try:
        result = [
            json.loads(line, object_pairs_hook=_strict_pairs,
                       parse_constant=lambda c: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            for line in data.decode("utf-8").splitlines() if line.strip()
        ]
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Malformed " + label + " JSONL") from exc
    if any(not isinstance(x, dict) for x in result):
        raise ValueError("JSONL records must be objects: " + label)
    return result


def _checksum(data: bytes) -> str:
    return digest(data).hexdigest()


def validate_contents(files: dict[str, bytes]) -> dict:
    """Validate research claims and original data, never infer GPU/semantic quality."""
    if set(files) != EXPECTED_FILES:
        raise ValueError("Unexpected or missing release file inventory")
    if any(len(data) > SINGLE_LIMIT or not isinstance(data, bytes)
           for data in files.values()) or sum(map(len, files.values())) > TOTAL_LIMIT:
        raise ValueError("Oversized research release contents")
    for name, data in files.items():
        if any(p.search(data) for p in SECRET_PATTERNS):
            raise ValueError("Potential secret in release file: " + name)

    checksum_text = files[HASH_FILE].decode("utf-8")
    checksums = {}
    for line in checksum_text.splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not m or m.group(2) in checksums:
            raise ValueError("Invalid or duplicated release checksum entry")
        checksums[m.group(2)] = m.group(1)
    if set(checksums) != set(files) - {HASH_FILE}:
        raise ValueError("Release checksum coverage incomplete")
    for name, expected in checksums.items():
        if _checksum(files[name]) != expected:
            raise ValueError("Release file has invalid SHA256: " + name)

    status = _json_bytes(files[STATUS_FILE], STATUS_FILE)
    if (status.get("version") != VERSION
        or status.get("status") != "LOCAL_CANDIDATE_UNPUBLISHED"
        or status.get("license_selected") is not False
        or status.get("owner_approved_publication") is not False
        or status.get("new_model_inference_executed") is not False
        or status.get("independent_human_review") is not False
        or status.get("trained_adapter_included") is not False
        or status.get("scored_baseline_cases") != 48
        or status.get("new_unscored_cases") != 72):
        raise ValueError("Release status falsely claims training, review or publication")

    if b"license:" in files["README.md"].split(b"---", 2)[1]:
        raise ValueError("Unapproved license in draft HF README")
    if not files["README.md"].startswith(b"---\n"):
        raise ValueError("Draft dataset card missing YAML frontmatter")

    original_rows = _jsonl_bytes(files["evaluated_48/benchmark_48.jsonl"], "original benchmark")
    validate_benchmark(original_rows)
    base_sha = _checksum(files["evaluated_48/benchmark_48.jsonl"])
    if base_sha != EXPECTED_BASELINE_SHA:
        raise ValueError("Original 48-case benchmark SHA differs from audited source")
    original = _json_bytes(
        files["evaluated_48/original_saved_tiny_aya_base_model_predictions.json"],
        "original saved baseline",
    )
    if (original.get("is_finetuned") is not False
        or original.get("model_revision") is not None
        or original.get("model") != "CohereLabs/tiny-aya-global"):
        raise ValueError("Original base-model revision, identity or tuned claim altered")
    mapped = check_saved_outputs(original, original_rows, base_sha)
    scores = [score(row, mapped[row["id"]]) for row in original_rows]
    totals = aggregate(scores)
    stored_metrics = original.get("metrics")
    if not isinstance(stored_metrics, dict):
        raise ValueError("Original saved model metrics missing")
    for key, expected in totals.items():
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            claimed = stored_metrics.get(key)
            if (not isinstance(claimed, (int, float))
                or isinstance(claimed, bool)
                or not math.isclose(claimed, expected, abs_tol=0.011)):
                raise ValueError("Saved original baseline metric mismatch: " + key)
    if "scores" in original:
        reported = original["scores"]
        if (not isinstance(reported, list)
            or len(reported) != 48
            or {item.get("id") for item in reported if isinstance(item, dict)}
               != {row["id"] for row in original_rows}):
            raise ValueError("Incomplete saved per-case model scoring")
        by_id = {item["id"]: item for item in reported}
        for entry in scores:
            if any(by_id[entry["id"]].get(key) != entry[key] for key in (
                "correct", "format_valid", "abstention_correct", "unsupported_claim"
            )):
                raise ValueError("Saved original per-case scores disagree with raw model text")

    report = _json_bytes(files["evaluated_48/curator_audit_chain_report.json"], "chain")
    audit = _json_bytes(files["evaluated_48/auditkit_native_receipt.json"], "AuditKIT")
    exporter_input = files["evaluated_48/curator_input.jsonl"]
    match_curated(original_rows, _jsonl_bytes(exporter_input, "original CuratorKIT input"))
    export = files["evaluated_48/curator_export.jsonl"]
    match_curated(original_rows, _jsonl_bytes(export, "original CuratorKIT export"))
    curator_sha = _checksum(export)
    if not isinstance(report.get("curation"), dict) or not isinstance(report.get("model"), dict):
        raise ValueError("Malformed CuratorKIT chain receipt")
    if (report.get("status") != "curatorkit_auditkit_chain_complete"
        or report.get("case_count") != 48
        or report.get("dataset_sha256") != base_sha
        or any(report.get(k) != v for k, v in PINNED_TOOLKITS.items())
        or report["curation"].get("export_sha256") != curator_sha
        or report["curation"].get("input_sha256") != _checksum(exporter_input)
        or report["curation"].get("manifest_sha256") != _checksum(
            files["evaluated_48/curator_manifest.json"]
        )
        or report["curation"].get("accepted") != 48
        or report["curation"].get("label_and_case_id_integrity_verified") is not True
        or report.get("human_labels_independently_reviewed") is not False
        or report["model"].get("fine_tuned") is not False
        or report["model"].get("new_inference_performed") is not False):
        raise ValueError("Original CuratorKIT/AuditKIT chain claims inconsistent")
    recorded_hashes = files["evaluated_48/curator_checksums.txt"]
    if curator_sha.encode() not in recorded_hashes:
        raise ValueError("Native CuratorKIT checksum does not cover exported data")
    if (audit.get("status") != "complete"
        or audit.get("sample_count") != 48
        or audit.get("dataset_sha256") != base_sha
        or audit.get("auditkit_version") != "1.1.3"
        or audit.get("model_inference_executed") is not False
        or audit.get("is_post_training_evaluation") is not False
        or audit.get("model_accuracy_semantically_verified") is not False
        or audit.get("human_review_completed") is not False
        or audit.get("auditkit_fingerprint") != report.get("auditkit_fingerprint")):
        raise ValueError("Official AuditKIT saved-output receipt inconsistent")

    challenge = files["challenge_72/challenge_v1_UNSCORED.jsonl"]
    challenge_sha = _checksum(challenge)
    if challenge_sha != EXPECTED_CHALLENGE_SHA:
        raise ValueError("Unscored 72-case challenge bytes differ from original frozen data")
    challenge_rows = _jsonl_bytes(challenge, "new challenge")
    validate_challenge(challenge_rows)
    if challenge_rows != build_challenge():
        raise ValueError("Frozen challenge differs from transparent synthetic generation")
    challenge_manifest = _json_bytes(
        files["challenge_72/challenge_manifest_UNSCORED.json"], "challenge manifest",
    )
    challenge_export = files["challenge_72/curator_export_UNSCORED.jsonl"]
    challenge_export_sha = _checksum(challenge_export)
    match_curated(challenge_rows, _jsonl_bytes(challenge_export, "challenge CuratorKIT export"))
    if (challenge_manifest.get("status") != "curated_unscored"
        or challenge_manifest.get("cases") != 72
        or challenge_manifest.get("curated_cases") != 72
        or challenge_manifest.get("dataset_sha256") != challenge_sha
        or challenge_manifest.get("curator_export_sha256") != challenge_export_sha
        or challenge_manifest.get("model_predictions_available") is not False
        or challenge_manifest.get("independent_human_review") is not False
        or challenge_manifest.get("evaluation_run") is not False
        or challenge_manifest.get("training_data") is not False):
        raise ValueError("Challenge curation or status claims inconsistent")
    return {
        "status": "LOCAL_RESEARCH_HANDOFF_VALIDATED_UNPUBLISHED",
        "file_count": len(files),
        "original_evaluated_cases": 48,
        "new_unscored_cases": 72,
        "original_dataset_sha256": base_sha,
        "challenge_dataset_sha256": challenge_sha,
        "original_curator_export_sha256": curator_sha,
        "challenge_curator_export_sha256": challenge_export_sha,
        "human_semantic_review_completed": False,
        "new_gpu_inference_executed": False,
        "trained_adapter_included": False,
        "license_selected": False,
        "huggingface_published": False,
        "official_lexsi_submission_completed": False,
        "source": "repository inputs plus actual local CuratorKIT and AuditKIT CPU outputs",
    }


def gather(root: Path = ROOT) -> dict[str, bytes]:
    root = Path(root).resolve()
    files = {}
    for name, rel in FILES.items():
        path = root / rel
        if not path.is_file() or path.is_symlink() or path.stat().st_size > SINGLE_LIMIT:
            raise ValueError("Missing, symlinked or oversized source: " + rel)
        files[name] = path.read_bytes()
    status = {
        "version": VERSION,
        "status": "LOCAL_CANDIDATE_UNPUBLISHED",
        "license_selected": False,
        "owner_approved_publication": False,
        "new_model_inference_executed": False,
        "independent_human_review": False,
        "trained_adapter_included": False,
        "scored_baseline_cases": 48,
        "new_unscored_cases": 72,
        "historical_20_file_archive_byte_identical": False,
        "research_not_a_blind_test": True,
        "publication_reviewer_note": (
            "Research handoff only. Do not publish until dataset owner approves "
            "a lawful license and content; never claim a tuned model exists."
        ),
    }
    files[STATUS_FILE] = (json.dumps(status, sort_keys=True, indent=2) + "\n").encode()
    files[HASH_FILE] = ("".join(
        _checksum(files[name]) + "  " + name + "\n"
        for name in sorted(files)
    )).encode()
    return files


def _zip_member(name: str, data: bytes) -> ZipInfo:
    info = ZipInfo(name, date_time=(2026, 11, 4, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits |= 0x800
    return info


def write_bundle(root: Path, destination: Path) -> dict:
    files = gather(root)
    details = validate_contents(files)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to overwrite existing release candidate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with ZipFile(destination, "x") as zipped:
            created = True
            for name in sorted(files):
                zipped.writestr(_zip_member(name, files[name]), files[name])
        audited = inspect_zip(destination)
    except BaseException:
        # A concurrent user may create a file between our existence check and
        # ZIP_STORED exclusive open. Never remove someone else's evidence.
        if created:
            destination.unlink(missing_ok=True)
        raise
    return {**details, "archive_sha256": sha256(destination),
            "archive_path": str(destination), "post_write_verified": audited["status"]}


def inspect_zip(path: Path) -> dict:
    path = Path(path)
    if (not path.is_file() or path.is_symlink() or path.stat().st_size > BUNDLE_LIMIT):
        raise ValueError("Missing, symlinked or oversized release ZIP")
    files = {}
    total = 0
    with ZipFile(path) as zipped:
        if len(zipped.infolist()) != len(EXPECTED_FILES):
            raise ValueError("Unexpected number of ZIP members")
        for member in zipped.infolist():
            name = member.filename
            pp = PurePosixPath(name)
            if (not pp.parts or name.startswith("/") or "\\" in name
                or ".." in pp.parts or name not in EXPECTED_FILES or name in files
                or member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000
                or member.file_size > SINGLE_LIMIT):
                raise ValueError("Invalid, duplicated or unsafe archive member")
            total += member.file_size
            if total > TOTAL_LIMIT:
                raise ValueError("ZIP content size limit reached")
            files[name] = zipped.read(member)
    return {**validate_contents(files), "archive_sha256": sha256(path)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--build", action="store_true",
                       help="Explicitly build unpublished release candidate ZIP")
    group.add_argument("--inspect", type=Path,
                       help="Read-only verification of an existing research handoff")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path,
                   default=ROOT / "artifacts" / "track2a_release_candidate_v2_UNPUBLISHED.zip")
    args = p.parse_args()
    if args.inspect is not None:
        result = inspect_zip(args.inspect)
    elif args.build:
        result = write_bundle(args.root, args.output)
    else:
        result = validate_contents(gather(args.root))
        result["mode"] = "READ_ONLY_PREFLIGHT_NO_ZIP_WRITTEN"
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
