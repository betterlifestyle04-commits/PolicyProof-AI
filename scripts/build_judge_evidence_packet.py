"""Build a deterministic, checksum-verifiable SAFE evidence packet for Lexsi judges.

This is a CPU-only packaging and independent SOURCE-REPLAY preflight for the
48 previously saved Tiny Aya base-model responses and 72 entirely UNSCORED
challenge cases. It is NOT a real GPU run, trained adapter, human review, HF
publication, model-safety certification, or an official Lexsi submission.

No uploaded proprietary weights, tokens, reviewer CSVs, Colab logs or synthetic
unit-test model responses are shipped. No files are modified by --inspect.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from policyproof.core import ROOT
from scripts.track2a_source_evidence import verify_original_chain, verify_unscored_challenge
from scripts.track2a_readiness import readiness
from scripts.verify_evidence import read_evidence_json

SOURCES = (
    "docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md",
    "docs/TRACK2A_SUBMISSION_ABSTRACT.md",
    "docs/DEV48_PAIRED_GPU_RESULTS_2026-11-05.md",
    "data/dev48_paired_public_summary.json",
    "data/unseen_policy_test.jsonl",
    "artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json",
    "artifacts/track2a_safety/curator_input.jsonl",
    "artifacts/track2a_safety/curator_output/sft_alpaca.jsonl",
    "artifacts/track2a_safety/curator_output/manifest.json",
    "artifacts/track2a_safety/curator_output/checksums.txt",
    "artifacts/track2a_safety/track2a_chain_report.json",
    "artifacts/track2a_safety/auditkit_native_receipt.json",
    "artifacts/track2a_challenge_unscored/challenge_v1.jsonl",
    "artifacts/track2a_challenge_unscored/challenge_manifest.json",
    "artifacts/track2a_challenge_unscored/challenge_curator_input.jsonl",
    "artifacts/track2a_challenge_unscored/challenge_curator_output/sft_alpaca.jsonl",
    "artifacts/track2a_readiness.json",
)
MANIFEST = "PACKET_MANIFEST.json"
HASHES = "SHA256SUMS.txt"
GUIDE = "START_HERE.txt"
ALLOWED = set(SOURCES) | {MANIFEST, HASHES, GUIDE}
FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,90}\.zip$")
SINGLE_LIMIT = 5_000_000
TOTAL_LIMIT = 18_000_000
ZIP_LIMIT = 20_000_000
SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    re.compile(rb"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    re.compile(rb"(?:ghp|github_pat|sk-proj)_[A-Za-z0-9_-]{35,}"),
)
GUIDANCE = """PolicyProof AI — SOURCE-REPLAYED judge handoff (NOT an official submission)

OPEN docs/TRACK2A_SUBMISSION_ABSTRACT.md for the bounded Track 2a narrative.
OPEN docs/DEV48_PAIRED_GPU_RESULTS_2026-11-05.md for the separate paired
48-case DEVELOPMENT experiment, whose original uploaded GPU ZIP is NOT here.
OPEN PACKET_MANIFEST.json for exact source hashes, scope and missing evidence.
All distributed files are checked against SHA256SUMS.txt on packet inspection.

Research reproduced here: 48 original saved baseline responses were scored
against the original pinned synthetic benchmark, and CuratorKIT/AuditKIT
receipts, groups and 72-curated-but-UNSCORED cases match source data.

This bundle does NOT contain GPU execution attestation, a finished native
AlignTune adapter, independent human labels, original dev48 paired raw ZIP,
completed final 60-case model results, public Hugging Face repositories,
live external verification, or Lexsi organizer submission confirmation.
The 48 original baseline and 48 paired development reports are DISTINCT.

To reproduce as a CPU-only consistency check on an existing source checkout:
  python -m scripts.track2a_readiness
  python -m scripts.build_judge_evidence_packet --inspect /path/to/packet.zip
The receipt files here are synthetic-policy evaluation artifacts, not live
banking advice. Original upstream model weights are license-controlled.
Do NOT mistake a checksum, local score, or exact citation ID for factual
accuracy, real GPU attestation or semantic safety certification.
"""
UNCHECKED = (
    "original_gpu_execution_independently_authenticated",
    "native_aligntune_training_completed",
    "final_60_case_human_review",
    "final_60_case_paired_evaluation",
    "72_case_original_gpu_evaluation_completed",
    "human_semantic_quality_verified",
    "huggingface_publication_verified",
    "official_lexsi_submission_verified",
)


def _file_bytes(root: Path, member: str) -> bytes:
    file = root / member
    parent = file
    while parent != root:
        if parent.is_symlink():
            raise ValueError("Judge source path is symlinked: " + member)
        parent = parent.parent
    if not file.is_file() or not 0 < file.stat().st_size <= SINGLE_LIMIT:
        raise ValueError("Missing, empty or oversized public judge source: " + member)
    raw = file.read_bytes()
    if len(raw) > SINGLE_LIMIT or any(p.search(raw) for p in SECRET_PATTERNS):
        raise ValueError("Unsafe size or apparent credential in judge source: " + member)
    return raw


def _json_bytes(blob: bytes, label: str) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("Duplicate JSON key in " + label)
            out[key] = value
        return out
    try:
        obj = json.loads(blob.decode("utf-8"), object_pairs_hook=unique,
                         parse_constant=lambda t: (_ for _ in ()).throw(
                             ValueError("Nonfinite JSON: " + t)))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("Malformed or ambiguous packet JSON: " + label) from exc
    if not isinstance(obj, dict):
        raise ValueError("Expected JSON object in " + label)
    return obj


def _validate_dev48_public_summary(blob: bytes) -> dict:
    doc = _json_bytes(blob, "dev48 public summary")
    if (doc.get("schema") != "policyproof_saved_dev48_paired_public_summary_v1"
        or doc.get("cases") != 48
        or doc.get("dev_dataset_sha256") !=
            "d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c"
        or doc.get("verification_scope") !=
            "original_uploaded_saved_results_internally_recomputed_without_new_gpu_inference"
        or doc.get("composition") !=
            {"supported": 36, "unsupported": 9, "context_injection": 3}
        or doc.get("language_counts") != {"en": 16, "no": 16, "ar": 16}
        or {m.get("key"): (m.get("base"),m.get("tuned"),m.get("denominator"))
            for m in doc.get("metrics", []) if isinstance(m, dict)} != {
                "schema": (14, 48, 48),
                "citation_id": (1, 39, 39),
                "strict": (4, 6, 48),
                "abstention": (4, 6, 9),
                "supported_exact": (0, 0, 39),
                "unsupported_claim": (5, 3, 9),
            }
        or any(doc.get(f) is not False for f in (
            "semantic_human_review_complete", "final60_human_review_complete",
            "final60_paired_inference_complete",
            "gpu_execution_independently_reproduced",
            "native_auditkit_independently_reexecuted",
            "public_hf_collections_verified", "official_submission_verified"))):
        raise ValueError("Public dev48 summary metrics or honest limits changed")
    return doc


def verify_sources(root: Path = ROOT) -> dict:
    """Recompute source-linked original CPU results; fail on partial/fake sources."""
    root = Path(root).resolve()
    verified48 = verify_original_chain(root)
    challenge72 = verify_unscored_challenge(root)
    state = readiness(root)
    if (verified48.get("cases") != 48
        or verified48.get("source_curation_and_score_recomputed") is not True
        or challenge72.get("curated_cases") != 72
        or state.get("status") != "RESEARCH_READY_HF_PUBLICATION_BLOCKED"
        or state.get("ready_to_submit") is not False
        or state.get("training_claimed") is not False
        or state.get("new_model_evaluation_claimed_for_72_cases") is not False
        or state.get("official_submission_confirmed") is not False
        or state.get("conflicting_optional_evidence_blocks_release") is not False):
        raise ValueError("Source evidence is NOT the approved 48-scored/72-unscored research state")
    expected_state = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    stored_state = _file_bytes(root, "artifacts/track2a_readiness.json")
    if stored_state != expected_state.encode("utf-8"):
        raise ValueError("Saved Track 2a readiness snapshot is stale or forged")
    _validate_dev48_public_summary(_file_bytes(root, "data/dev48_paired_public_summary.json"))
    return {
        "48_saved_source_outputs_recomputed": True,
        "real_curatorkit_native_receipts_source_checked": True,
        "original_auditkit_native_receipt_source_checked": True,
        "72_curated_challenge_cases_verified_UNSCORED": True,
        "48_paired_dev_public_summary_NOT_original_archive": True,
        "track2a_scientific_status": state["status"],
    }


def _manifest(files: dict[str, bytes], verified: dict) -> dict:
    return {
        "schema": "policyproof_lexsi_judge_evidence_packet_v1",
        "status": "CPU_SOURCE_REPLAYED_RESEARCH_UNPUBLISHED_NOT_SUBMITTED",
        "original_evaluated_cases": 48,
        "new_unscored_challenge_cases": 72,
        "final_model_cases": 0,
        "source_sha256": {k: sha256(v).hexdigest() for k, v in sorted(files.items())},
        "source_consistency_verified": verified,
        "unverified_requirements": {name: False for name in UNCHECKED},
        "paired_dev48_original_zip_included": False,
        "private_reviewer_votes_included": False,
        "gated_model_weights_included": False,
        "other_model_inference_performed": False,
        "external_network_used": False,
        "warning": (
            "Hash consistency and native toolkit receipts do not independently "
            "attest original GPU work, semantic quality, legal rights or submission."
        ),
    }


def make_packet(root: Path = ROOT) -> bytes:
    root = Path(root).resolve()
    verified = verify_sources(root)
    files = {name: _file_bytes(root, name) for name in SOURCES}
    if sum(len(v) for v in files.values()) > TOTAL_LIMIT:
        raise ValueError("Judge packet public files exceed bounded total size")
    files[GUIDE] = GUIDANCE.encode("utf-8")
    files[MANIFEST] = (json.dumps(_manifest({k:v for k,v in files.items() if k in SOURCES},
                                          verified), indent=2, ensure_ascii=False,
                                   sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    files[HASHES] = "".join(
        sha256(files[name]).hexdigest() + "  " + name + "\n"
        for name in sorted(files) if name != HASHES
    ).encode("utf-8")
    stream = BytesIO()
    with ZipFile(stream, "w") as archive:
        for name in sorted(files):
            member = ZipInfo(name, (2026, 11, 8, 0, 0, 0))
            member.compress_type = ZIP_STORED
            member.external_attr = (0o100644 << 16)
            archive.writestr(member, files[name])
    blob = stream.getvalue()
    if len(blob) > ZIP_LIMIT:
        raise ValueError("Judge ZIP exceeds size limit")
    inspect_packet_bytes(blob)
    return blob


def inspect_packet_bytes(blob: bytes) -> dict:
    if not blob or len(blob) > ZIP_LIMIT:
        raise ValueError("Missing or oversized judge ZIP")
    with ZipFile(BytesIO(blob)) as archive:
        members = archive.infolist()
        if len(members) != len(ALLOWED) or archive.testzip() is not None:
            raise ValueError("Judge ZIP inventory corrupt or incomplete")
        files = {}
        count = 0
        for member in members:
            p = PurePosixPath(member.filename)
            if (member.filename not in ALLOWED or member.filename in files
                or p.is_absolute() or ".." in p.parts or "\\" in member.filename
                or member.is_dir() or member.file_size > SINGLE_LIMIT
                or member.compress_type != ZIP_STORED
                or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe, duplicated, extra or oversized judge ZIP member")
            count += member.file_size
            if count > TOTAL_LIMIT:
                raise ValueError("Judge ZIP decompressed size exceeded")
            raw = archive.read(member)
            if any(pat.search(raw) for pat in SECRET_PATTERNS):
                raise ValueError("Potential secret found in judge ZIP member")
            files[member.filename] = raw
    if set(files) != ALLOWED:
        raise ValueError("Judge ZIP file inventory differs from strict allowlist")
    checksums = {}
    for line in files[HASHES].decode("utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if not match or match.group(2) in checksums:
            raise ValueError("Invalid/duplicate judge ZIP checksum entry")
        checksums[match.group(2)] = match.group(1)
    if set(checksums) != set(files) - {HASHES} or any(
        sha256(files[name]).hexdigest() != digest
        for name, digest in checksums.items()
    ):
        raise ValueError("Judge ZIP file checksums do not match original bytes")
    manifest = _json_bytes(files[MANIFEST], MANIFEST)
    if manifest != _manifest(
        {name: files[name] for name in SOURCES},
        manifest.get("source_consistency_verified"),
    ) or manifest.get("status") != "CPU_SOURCE_REPLAYED_RESEARCH_UNPUBLISHED_NOT_SUBMITTED":
        raise ValueError("Judge ZIP manifest or honest limitations altered")
    _validate_dev48_public_summary(files["data/dev48_paired_public_summary.json"])
    # Extract only the hardcoded list into a new temporary directory; this
    # allows FULL source recomputation rather than trusting owner JSON claims.
    with tempfile.TemporaryDirectory(prefix="policyproof_judge_verify_") as temp:
        root = Path(temp)
        for name in SOURCES:
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(files[name])
        if verify_sources(root) != manifest["source_consistency_verified"]:
            raise ValueError("Original source scores and receipt checks do not reproduce")
    return {
        "status": "JUDGE_PACKET_SOURCE_REPLAY_VERIFIED_NOT_SUBMITTED",
        "files": len(files),
        "original_scored_development_cases": 48,
        "new_challenge_cases_unscored": 72,
        "sha256_of_all_members_verified": True,
        "native_source_consistency_recomputed": True,
        "original_gpu_inference_independently_attested": False,
        "human_semantic_review_verified": False,
        "public_hf_publication_verified": False,
        "official_lexsi_submission_verified": False,
    }


def _safe_output(output: str, root: Path) -> Path:
    root = Path(root).resolve()
    literal = str(output)
    parts = literal.split("/")
    if (len(parts) != 2 or parts[0] != "artifacts"
        or not FILENAME.fullmatch(parts[1])
        or any(ord(c) < 32 or ord(c) == 127 for c in literal)):
        raise ValueError("New judge ZIP output must be direct artifacts/<name>.zip")
    folder = root / "artifacts"
    if folder.is_symlink():
        raise ValueError("Unsafe symlinked artifacts directory")
    return folder / parts[1]


def write_packet(root: Path = ROOT,
                 output: str = "artifacts/policyproof_lexsi_judge_evidence_UNSUBMITTED.zip") -> dict:
    root = Path(root).resolve()
    path = _safe_output(output, root)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Never overwrite a judge packet or research evidence")
    blob = make_packet(root)
    path.parent.mkdir(exist_ok=True)
    # No overwrites; exclusive creation and exact readback before success.
    with path.open("xb") as out:
        out.write(blob)
    if path.read_bytes() != blob:
        raise RuntimeError("Judge ZIP changed during source write")
    return {
        **inspect_packet_bytes(blob),
        "output": str(path),
        "zip_sha256": sha256(blob).hexdigest(),
        "bytes": len(blob),
        "independent_original_GPU_attestation": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--inspect", type=Path, default=None)
    parser.add_argument("--output",
                        default="artifacts/policyproof_lexsi_judge_evidence_UNSUBMITTED.zip")
    args = parser.parse_args()
    if args.build and args.inspect is not None:
        parser.error("Select --build OR --inspect, not both")
    if args.inspect:
        path = args.inspect
        if path.is_symlink() or not path.is_file() or path.stat().st_size > ZIP_LIMIT:
            parser.error("Missing or unsafe judge ZIP")
        report = inspect_packet_bytes(path.read_bytes())
        report["zip_sha256"] = sha256(path.read_bytes()).hexdigest()
    elif args.build:
        report = write_packet(output=args.output)
    else:
        report = {"status": "READ_ONLY_PREFLIGHT_NO_ZIP",
                  "source_consistency": verify_sources()}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
