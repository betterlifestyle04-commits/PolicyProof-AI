"""CPU-only native AuditKIT 1.1.3 replay of the owner's pinned dev48 paired ZIP.

Unlike inspecting the paired report's saved metrics, this independently
executes the official AuditKIT scorers over the 96 PRESERVED model responses.
No model is downloaded, run, trained or improved; the 48 synthetic examples
were previously inspected, so this is NOT a blind final evaluation.

IMPORTANT: Native evaluator outputs reveal the base/tuned identity. Save
receipts under private git-ignored reviews/ and never share with blinded human
reviewers until their individual judgments are locked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from policyproof.core import ROOT, score, sha256
from scripts.auditkit_compare import comparison
from scripts.dev48_gpu_pair import DEV_REL, check_dev_rows, read_validated_report
from scripts.export_review_from_bundle import (
    ORIGINAL_DEV48_ZIP_SHA256, _strict_json, verify_bundle,
)
from scripts.paired_semantic_review import _private_path

SCHEMA = "policyproof_native_dev48_auditkit_cpu_replay_v1"
RESULT_DEFAULT = "reviews/dev48_native_auditkit_cpu_owner_only.json"
METRICS = ("correct", "format_valid", "unsupported_claim",
           "raw_citation_correct", "abstention_correct")


def _matched_counts(rows: list[dict], base: dict, tuned: dict) -> dict:
    """Recompute exact case-wise deltas, not from archived aggregate fields."""
    a = {p["id"]: p["raw"] for p in base["predictions"]}
    b = {p["id"]: p["raw"] for p in tuned["predictions"]}
    first = [score(row, a[row["id"]]) for row in rows]
    second = [score(row, b[row["id"]]) for row in rows]
    return {
        metric: {
            "baseline_count": sum(v[metric] for v in first),
            "candidate_count": sum(v[metric] for v in second),
            "count_delta": (sum(v[metric] for v in second)
                            - sum(v[metric] for v in first)),
        }
        for metric in METRICS
    }


def validated_dev48_source(root: Path, bundle: Path,
                           expected_sha256: str) -> tuple[list[dict], dict, dict, dict, dict]:
    """Fail closed BEFORE running AuditKIT; source ZIP is never modified."""
    root = Path(root).resolve()
    members, meta = verify_bundle(bundle, expected_sha256)
    rows, dataset_sha = check_dev_rows(root / DEV_REL)
    archived = _strict_json(members["dev48_paired_auditkit.json"], "paired report")
    base_json = _strict_json(members["dev48_base.json"], "base report")
    tuned_json = _strict_json(members["dev48_tuned.json"], "tuned report")
    if (archived.get("split") != "unseen-policy"
        or archived.get("not_comparable_to_historical_unpinned_48_base") is not True
        or archived.get("baseline_model_revision") != base_json.get("model_revision")
        or archived.get("generation_config") != base_json.get("generation_config")
        or archived.get("case_matched_deterministic_counts") is None
        or archived.get("verified_human_semantic_improvement") is not False):
        raise ValueError("Archived comparison lacks strict development provenance")
    with tempfile.TemporaryDirectory(prefix="policyproof-native-auditkit-check-") as tmp:
        a, b = Path(tmp) / "base.json", Path(tmp) / "tuned.json"
        a.write_bytes(members["dev48_base.json"])
        b.write_bytes(members["dev48_tuned.json"])
        base = read_validated_report(
            a, rows, revision=base_json["model_revision"],
            data_sha=dataset_sha, adapter_sha=None,
            zip_sha=tuned_json["adapter_source_zip_sha256"],
        )
        tuned = read_validated_report(
            b, rows, revision=base_json["model_revision"],
            data_sha=dataset_sha, adapter_sha=tuned_json["adapter_sha256"],
            zip_sha=tuned_json["adapter_source_zip_sha256"],
        )
    meta = dict(meta)
    meta["original_base_report_member_sha256"] = hashlib.sha256(
        members["dev48_base.json"]
    ).hexdigest()
    meta["original_tuned_report_member_sha256"] = hashlib.sha256(
        members["dev48_tuned.json"]
    ).hexdigest()
    meta["original_paired_report_member_sha256"] = hashlib.sha256(
        members["dev48_paired_auditkit.json"]
    ).hexdigest()
    counts = _matched_counts(rows, base, tuned)
    if archived.get("case_matched_deterministic_counts") != counts:
        raise ValueError("Archived paired counts conflict with recomputed raw outputs")
    if archived.get("adapter_sha256") != tuned["adapter_sha256"]:
        raise ValueError("Archived adapter identity does not match tuned results")
    return rows, base, tuned, archived, meta


def replay(root: Path, bundle: Path, expected_sha256: str,
           result_name: str = RESULT_DEFAULT) -> dict:
    """Run real AuditKIT on saved model text; private exclusive-write receipt."""
    root = Path(root).resolve()
    target = _private_path(root, result_name, ".json", new=True)
    rows, base, tuned, prior, meta = validated_dev48_source(
        root, bundle, expected_sha256
    )
    # This actually runs AuditKIT twice with model='precomputed'. No
    # Transformers/model import, weights, network token or CUDA required.
    fresh = comparison(base, tuned, rows, meta["dataset_sha256"])
    counts = _matched_counts(rows, base, tuned)
    if (fresh.get("status") != "paired_deterministic_audit"
        or fresh.get("sample_count") != 48
        or fresh.get("evaluation_checksum") != meta["dataset_sha256"]
        or fresh.get("case_matched_deterministic_counts") != counts
        or fresh.get("baseline_model_revision") != prior["baseline_model_revision"]
        or fresh.get("adapter_sha256") != prior["adapter_sha256"]
        or fresh.get("verified_human_semantic_improvement") is not False):
        raise RuntimeError("Fresh native AuditKIT replay is not consistent with original evidence")

    import importlib.metadata
    version = importlib.metadata.version("auditkit")
    if version != "1.1.3":
        raise RuntimeError(
            "Expected official AuditKIT 1.1.3 for comparable evidence, got "
            + version
        )
    result = {
        "schema": SCHEMA,
        "status": "NATIVE_AUDITKIT_SAVED_DEV48_PAIRED_REPLAY_COMPLETED",
        "split": "already_inspected_synthetic_development_only",
        "pinned_source_bundle_sha256": meta["source_zip_sha256"],
        "pinned_dataset_sha256": meta["dataset_sha256"],
        "original_base_report_member_sha256": meta["original_base_report_member_sha256"],
        "original_tuned_report_member_sha256": meta["original_tuned_report_member_sha256"],
        "original_paired_report_member_sha256": meta["original_paired_report_member_sha256"],
        "model_revision_reported": base["model_revision"],
        "adapter_sha256_reported": tuned["adapter_sha256"],
        "auditkit_version_replayed": version,
        "auditkit_executed_in_this_process": True,
        "replayed_model_outputs_per_variant": 48,
        "replayed_model_outputs_total": 96,
        "saved_archived_counts_verified": True,
        "fresh_native_counts_match_archived": True,
        "native_auditkit_base_fingerprint": fresh["baseline_auditkit_fingerprint"],
        "native_auditkit_tuned_fingerprint": fresh["candidate_auditkit_fingerprint"],
        "native_auditkit_base_headline": fresh["auditkit_baseline"],
        "native_auditkit_tuned_headline": fresh["auditkit_candidate"],
        "case_matched_deterministic_counts": counts,
        "actual_model_inference_in_this_process": False,
        "model_training_in_this_process": False,
        "independent_gpu_generation_proven": False,
        "original_gpu_metadata_independently_authenticated": False,
        "semantic_correctness_human_review_completed": False,
        "final_blind_benchmark": False,
        "safe_for_blinded_reviewers_before_voting": False,
        "limitations": [
            "Real AuditKIT 1.1.3 was executed on SAVED predictions, not on a downloaded Tiny Aya model",
            "No independently authenticated GPU inference logs or true training optimizer steps",
            "A native exact/string/source-ID metric does not entail evidence-grounded factual accuracy",
            "Small synthetic previously inspected dev48 examples are not a final blind benchmark",
            "Keep owner-only report private until independent blinded human votes are locked",
        ],
    }
    # Freeze source after compute so a concurrently changed ZIP cannot be
    # labeled as though it was the original pinned source.
    if sha256(Path(bundle)) != expected_sha256:
        raise ValueError("Original source ZIP changed during native replay")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never truncate any pre-existing reviewer data.
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {
        "status": result["status"],
        "private_output": str(target),
        "model_inference": False,
        "human_semantic_review": False,
        "official_auditkit_version": version,
        "paired_cases": 48,
        "comparison_matches_saved_counts": True,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--expected-sha256", default=ORIGINAL_DEV48_ZIP_SHA256)
    p.add_argument("--output", default=RESULT_DEFAULT)
    p.add_argument("--execute", action="store_true",
                   help="Explicitly run native AuditKIT and create private owner-only receipt")
    args = p.parse_args()
    if args.execute:
        result = replay(args.root, args.bundle, args.expected_sha256, args.output)
    else:
        rows, base, tuned, prior, meta = validated_dev48_source(
            args.root, args.bundle, args.expected_sha256
        )
        result = {
            "status": "PINNED_PAIR_VERIFIED_AUDITKIT_NOT_EXECUTED",
            "source_bundle_sha256": meta["source_zip_sha256"],
            "cases": len(rows),
            "saved_comparison_counts_recomputed": True,
            "native_auditkit_executed": False,
            "human_semantic_review": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
