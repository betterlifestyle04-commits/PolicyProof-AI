"""Source-linked, CPU-only checks for Track 2a research readiness.

Unlike a status-only receipt check, these routines bind the real 48 saved
outputs, exact frozen benchmarks, CuratorKIT exported labels, source file
digests, original AuditKIT receipt and reported subgroup scores.

The checks still CANNOT attest original GPU execution, autonomous reviewer
judgments or licensing. No network access, model generation or file writes.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from policyproof.core import ROOT, sha256
from scripts.auditkit_replay import check_saved_outputs
from scripts.generate_track2a_challenge import validate_challenge
from scripts.track2a_chain import (
    CATEGORIES, LANGUAGES, aggregate_slice, match_curated, validate_benchmark,
)
from scripts.verify_evidence import read_evidence_json

BASE_DATA_SHA = "d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c"
CHALLENGE_DATA_SHA = "cdd5cf4e54caa6c79ce1f0cd5bd0f05344f0dc22039a2ad56a3c1c0ee2d32cd1"
MAX_BYTES = 5_000_000


def _read_bytes(root: Path, relative: str, *, limit: int = MAX_BYTES) -> bytes:
    """Refuse symlinked files/parents and oversized/missing local evidence."""
    root = Path(root).resolve()
    path = root / relative
    ancestor = path
    while ancestor != root:
        if ancestor.is_symlink():
            raise ValueError("Symlinked research evidence: " + relative)
        ancestor = ancestor.parent
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Missing, empty or oversized research evidence: " + relative)
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError("Oversized research evidence: " + relative)
    return raw


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field in Track 2a evidence: " + key)
        result[key] = value
    return result


def _parse_json(raw: bytes, label: str) -> dict:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_fields,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite value: " + token)
            ),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Ambiguous JSON evidence: " + label) from exc
    if not isinstance(value, dict):
        raise ValueError("JSON research evidence must be an object: " + label)
    return value


def _parse_jsonl(raw: bytes, label: str) -> list[dict]:
    try:
        values = [
            json.loads(
                line, object_pairs_hook=_unique_fields,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError("Non-finite value: " + token)
                ),
            )
            for line in raw.decode("utf-8").splitlines() if line.strip()
        ]
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Ambiguous JSONL evidence: " + label) from exc
    if not values or any(not isinstance(row, dict) for row in values):
        raise ValueError("Invalid JSONL research evidence: " + label)
    return values


def verify_original_chain(root: Path = ROOT) -> dict:
    """Recompute chain totals and trace real CuratorKIT artifacts to sources."""
    root = Path(root).resolve()
    data = _read_bytes(root, "data/unseen_policy_test.jsonl")
    if sha256(root / "data/unseen_policy_test.jsonl") != BASE_DATA_SHA:
        raise ValueError("Original 48-case benchmark SHA-256 differs from frozen source")
    rows = _parse_jsonl(data, "original 48-case benchmark")
    validate_benchmark(rows)
    baseline = _parse_json(
        _read_bytes(root, "artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json"),
        "original Tiny Aya base predictions",
    )
    if baseline.get("model") != "CohereLabs/tiny-aya-global" or baseline.get("is_finetuned") is not False:
        raise ValueError("Unrecognized saved original unchanged model")
    predictions = check_saved_outputs(baseline, rows, BASE_DATA_SHA)
    base_dir = "artifacts/track2a_safety/"
    input_raw = _read_bytes(root, base_dir + "curator_input.jsonl")
    export_raw = _read_bytes(root, base_dir + "curator_output/sft_alpaca.jsonl")
    manifest_raw = _read_bytes(root, base_dir + "curator_output/manifest.json")
    checksum_raw = _read_bytes(root, base_dir + "curator_output/checksums.txt", limit=100_000)
    chain = _parse_json(
        _read_bytes(root, base_dir + "track2a_chain_report.json"),
        "saved CuratorKIT/AuditKIT chain",
    )
    audit = _parse_json(
        _read_bytes(root, base_dir + "auditkit_native_receipt.json"),
        "saved native AuditKIT report",
    )
    _parse_json(manifest_raw, "CuratorKIT manifest")
    match_curated(rows, _parse_jsonl(input_raw, "CuratorKIT input"))
    match_curated(rows, _parse_jsonl(export_raw, "CuratorKIT export"))
    curation = chain.get("curation")
    model = chain.get("model")
    if not isinstance(curation, dict) or not isinstance(model, dict):
        raise ValueError("Missing original toolkit chain provenance")
    export_hash = sha256(root / base_dir / "curator_output/sft_alpaca.jsonl")
    input_hash = sha256(root / base_dir / "curator_input.jsonl")
    manifest_hash = sha256(root / base_dir / "curator_output/manifest.json")
    if (chain.get("status") != "curatorkit_auditkit_chain_complete"
        or chain.get("track") != "2a_adversarial_and_safety_evaluation"
        or chain.get("case_count") != 48
        or chain.get("dataset_sha256") != BASE_DATA_SHA
        or chain.get("curatorkit_version") != "0.1.22"
        or chain.get("auditkit_version") != "1.1.3"
        or chain.get("human_labels_independently_reviewed") is not False
        or curation.get("accepted") != 48
        or curation.get("input_sha256") != input_hash
        or curation.get("export_sha256") != export_hash
        or curation.get("manifest_sha256") != manifest_hash
        or curation.get("label_and_case_id_integrity_verified") is not True
        or curation.get("training_performed") is not False
        or curation.get("semantic_and_hallucination_judges_used") is not False
        or model.get("name") != baseline.get("model")
        or model.get("recorded_model_revision") != baseline.get("model_revision")
        or model.get("new_inference_performed") is not False
        or model.get("fine_tuned") is not False
        or export_hash.encode() not in checksum_raw):
        raise ValueError("Original chain receipt does not match preserved CuratorKIT files")
    if (chain.get("total") != aggregate_slice(rows, predictions)
        or chain.get("by_category") != {
            category: aggregate_slice([r for r in rows if r["category"] == category], predictions)
            for category in CATEGORIES
        }
        or chain.get("by_language") != {
            lang: aggregate_slice([r for r in rows if r["language"] == lang], predictions)
            for lang in LANGUAGES
        }):
        raise ValueError("Original chain reported scores differ from preserved raw model outputs")
    if (audit.get("status") != "complete"
        or audit.get("sample_count") != 48
        or audit.get("dataset_sha256") != BASE_DATA_SHA
        or audit.get("auditkit_version") != "1.1.3"
        or audit.get("model") != baseline.get("model")
        or audit.get("model_inference_executed") is not False
        or audit.get("human_review_completed") is not False
        or audit.get("auditkit_fingerprint") != chain.get("auditkit_fingerprint")
        or audit.get("auditkit_run_id") != chain.get("auditkit_run_id")):
        raise ValueError("Original native AuditKIT receipt conflicts with 48-case chain")
    return {
        "cases": 48,
        "dataset_sha256": BASE_DATA_SHA,
        "original_saved_outputs": 48,
        "source_curation_and_score_recomputed": True,
        "original_model_gpu_provenance_independently_verified": False,
        "semantic_accuracy_human_verified": False,
    }


def verify_unscored_challenge(root: Path = ROOT) -> dict:
    """Validate 72 pristine synthetic cases and original CuratorKIT export."""
    root = Path(root).resolve()
    directory = "artifacts/track2a_challenge_unscored/"
    frozen = _read_bytes(root, directory + "challenge_v1.jsonl")
    if sha256(root / directory / "challenge_v1.jsonl") != CHALLENGE_DATA_SHA:
        raise ValueError("New 72-case frozen challenge SHA-256 changed")
    rows = _parse_jsonl(frozen, "72-case challenge")
    validate_challenge(rows)
    manifest = _parse_json(
        _read_bytes(root, directory + "challenge_manifest.json"),
        "new unscored challenge manifest",
    )
    source = _parse_jsonl(
        _read_bytes(root, directory + "challenge_curator_input.jsonl"),
        "new challenge CuratorKIT input",
    )
    curated_raw = _read_bytes(
        root, directory + "challenge_curator_output/sft_alpaca.jsonl",
    )
    curated = _parse_jsonl(curated_raw, "new challenge CuratorKIT export")
    match_curated(rows, source)
    match_curated(rows, curated)
    export_sha = sha256(root / directory / "challenge_curator_output/sft_alpaca.jsonl")
    if (manifest.get("status") != "curated_unscored"
        or manifest.get("cases") != 72
        or manifest.get("curated_cases") != 72
        or manifest.get("dataset_sha256") != CHALLENGE_DATA_SHA
        or manifest.get("curator_export_sha256") != export_sha
        or manifest.get("curatorkit_executed") is not True
        or manifest.get("model_predictions_available") is not False
        or manifest.get("evaluation_run") is not False
        or manifest.get("training_data") is not False
        or manifest.get("independent_human_review") is not False
        or manifest.get("language_counts") != {"en": 24, "no": 24, "ar": 24}
        or manifest.get("attack_family_counts") != dict(Counter(r["category"] for r in rows))):
        raise ValueError("Unscored challenge manifest not backed by exact source/curation export")
    return {
        "cases": 72,
        "dataset_sha256": CHALLENGE_DATA_SHA,
        "curated_cases": 72,
        "new_model_inference_confirmed": False,
        "independent_human_review_completed": False,
    }
