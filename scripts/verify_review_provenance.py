"""Fail-closed, source-linked independent final-benchmark review evidence.

This validator proves FILE CONSISTENCY, not that a human actually reviewed,
independently understood, or was qualified to judge anything. No approval is
generated and no reviewer identity is independently authenticated.

The review files belong in a private local reviews/ folder, never public CI.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.review_holdout import (
    REVIEW_FIELDS, check_reviewer_code, consolidate_reviews,
    read_and_validate_review,
)

SCHEMA = "policyproof_review_evidence_v1"
LANGUAGES = {"en", "no", "ar"}
DIGEST = re.compile(r"^[0-9a-f]{64}$")
MAX_FILE_BYTES = 1_500_000


def _unique_fields(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("Duplicate key in review evidence JSON: " + key)
        obj[key] = value
    return obj


def _strict_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Missing, symlinked, empty or oversized review evidence JSON")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_fields,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON review evidence: " + token)
            ),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Malformed or ambiguous review evidence JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Review evidence must be a JSON object")
    return value


def _safe_private_path(root: Path, relative: str, extension: str) -> Path:
    if (not isinstance(relative, str) or len(relative) > 220
        or "\\" in relative or chr(0) in relative
        or not relative.startswith("reviews/")
        or not relative.endswith(extension)):
        raise ValueError("Review source must be a private reviews/ relative " + extension + " file")
    literal_parts = relative.split("/")
    parts = PurePosixPath(relative).parts
    if (len(parts) < 2 or len(literal_parts) != len(parts)
        or any(item in ("", ".", "..") for item in literal_parts)):
        raise ValueError("Unsafe review evidence path")
    root = root.resolve()
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root / "reviews"):
        raise ValueError("Review evidence cannot escape private reviews directory")
    if path.is_symlink() or any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
        raise ValueError("Symlinked private review evidence is not accepted")
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("Review source is missing, empty or oversized")
    return path


def _declared_hash(path: Path, digest: str) -> None:
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ValueError("Review evidence must include a full SHA-256 checksum")
    if sha256(path) != digest:
        raise ValueError("Review source SHA-256 mismatch: " + path.name)


def verify_review_evidence(
    root: Path, cases: list[dict], dataset_digest: str,
    signoff: dict,
) -> dict:
    """Validate all three criterion votes on all 60 original cases per reviewer.

    Only unanimous all-yes, SHA-bound review packs + an *exact* independently
    regenerated reconciliation + explicit manual approval declarations qualify.
    Negative verdicts require dataset correction/review, not automatic override.
    """
    root = Path(root).resolve()
    if len(cases) != 60 or len({c["id"] for c in cases}) != 60:
        raise ValueError("Review provenance requires 60 distinct frozen case IDs")
    if not isinstance(dataset_digest, str) or not DIGEST.fullmatch(dataset_digest):
        raise ValueError("Invalid frozen final dataset digest")
    evidence = signoff.get("review_evidence")
    if not isinstance(evidence, dict) or evidence.get("schema") != SCHEMA:
        raise ValueError("Missing versioned review-source evidence; signoff metadata alone is insufficient")
    files = evidence.get("review_files")
    if not isinstance(files, list) or not 2 <= len(files) <= 4:
        raise ValueError("Require 2–4 independent reviewer CSV file records")
    review_paths, codes = [], []
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Malformed review file declaration")
        code = check_reviewer_code(entry.get("reviewer_code"))
        path = _safe_private_path(root, entry.get("path"), ".csv")
        _declared_hash(path, entry.get("sha256"))
        try:
            measured_code, ratings = read_and_validate_review(
                path, cases, dataset_digest,
            )
        except (csv.Error, UnicodeError) as exc:
            raise ValueError("Malformed or unreadable reviewer CSV evidence") from exc
        if code != measured_code:
            raise ValueError("Review file code does not match declared reviewer")
        # Every reviewer must independently fill all 180 fields with affirmative
        # judgments; any no/blank requires a corrected benchmark and renewed review.
        if any(ratings[c["id"]][field] != "yes"
               for c in cases for field in REVIEW_FIELDS):
            raise ValueError("Incomplete or negative review judgment; revise/resolve before signoff")
        review_paths.append(path)
        codes.append(code)
    if len(codes) != len(set(codes)) or len({p.resolve() for p in review_paths}) != len(review_paths):
        raise ValueError("Duplicate reviewer pseudonym or repeated review file")
    if signoff.get("reviewers") != codes:
        raise ValueError("Reviewer order/codes do not match signed source review files")

    declared_report = evidence.get("reconciliation")
    if not isinstance(declared_report, dict):
        raise ValueError("Missing pinned independent-review reconciliation")
    summary_path = _safe_private_path(root, declared_report.get("path"), ".json")
    _declared_hash(summary_path, declared_report.get("sha256"))
    observed = _strict_json(summary_path)
    try:
        recomputed = consolidate_reviews(review_paths, cases, dataset_digest)
    except (csv.Error, UnicodeError) as exc:
        raise ValueError("Malformed reviewer CSV during reconciliation") from exc
    if observed != recomputed:
        raise ValueError("Reconciliation differs from the source reviewer CSV evidence")
    if (recomputed["status"] != "READY_FOR_MANUAL_ADJUDICATION_AND_SIGNOFF"
        or recomputed["missing_judgments"]
        or recomputed["disagreement_slots"]
        or recomputed["negative_judgment_slots"]
        or recomputed["unanimous_agreement_slots"] != 60 * len(REVIEW_FIELDS)):
        raise ValueError("Unresolved negative, disputed or missing gold-label judgments")

    approvals = evidence.get("approvals")
    if not isinstance(approvals, list) or len(approvals) != len(codes):
        raise ValueError("Each reviewer must individually declare manual approval")
    languages = set()
    seen = set()
    for approval in approvals:
        if not isinstance(approval, dict):
            raise ValueError("Malformed review manual approval declaration")
        code = check_reviewer_code(approval.get("reviewer_code"))
        if code in seen or code not in codes:
            raise ValueError("Approval does not match a unique source reviewer")
        seen.add(code)
        if (approval.get("approved") is not True
            or approval.get("independent_review_declared") is not True
            or approval.get("blinded_to_model_predictions_declared") is not True):
            raise ValueError("Explicit reviewer approval and no-model-output declarations required")
        claimed = approval.get("competent_languages")
        if (not isinstance(claimed, list) or not claimed
            or not all(isinstance(lang, str) and lang in LANGUAGES for lang in claimed)
            or len(claimed) != len(set(claimed))):
            raise ValueError("Reviewer must declare which languages they can competently assess")
        languages.update(claimed)
    if set(codes) != seen or not LANGUAGES.issubset(languages):
        raise ValueError("Reviewers do not collectively declare English, Norwegian and Arabic competence")
    # Make a double-read race unable to validate an intermediate version of
    # the evidence files. These hashes must still match after reconciliation.
    for entry, path in zip(files, review_paths):
        _declared_hash(path, entry["sha256"])
    _declared_hash(summary_path, declared_report["sha256"])
    final_dataset = root / "data/final_holdout_v1.jsonl"
    if not final_dataset.is_file() or sha256(final_dataset) != dataset_digest:
        raise ValueError("Frozen final benchmark changed during review verification")

    return {
        "source_review_files_verified": len(review_paths),
        "case_count": len(cases),
        "criteria_per_case": len(REVIEW_FIELDS),
        "unanimous_yes_slots": recomputed["unanimous_agreement_slots"],
        "reconciliation_sha256": sha256(summary_path),
        "reviewer_csv_checksums_verified": True,
        "dataset_sha256": dataset_digest,
        "reviewer_identity_independently_verified": False,
        "independent_human_action_verified": False,
        "competence_independently_verified": False,
        "gold_semantics_independently_verified": False,
        "model_inference_performed": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Read-only validation of locally held human review provenance"
    )
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    from scripts.frozen_pair_inference import FINAL_REL, SIGNOFF_REL, verify_review_signoff
    root = args.root.resolve()
    try:
        dataset = root / FINAL_REL
        result = verify_review_signoff(
            root / SIGNOFF_REL, sha256(dataset), root=root,
        )
        print(json.dumps({"status": "SOURCE_LINKED_REVIEW_DECLARATIONS_VALIDATED",
                          **result}, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({
            "status": "BLOCKED",
            "reason": str(exc),
            "human_review_performed_by_this_command": False,
            "model_inference_performed": False,
        }, indent=2))


if __name__ == "__main__":
    main()
