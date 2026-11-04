"""Prepare a *DRAFT ONLY* signoff from genuine independent review CSV files.

No approval is generated. Every approval flag and review_complete remain FALSE,
regardless of apparent verdict agreement. Actual reviewer attestations are
manual and this tool cannot prove human identity, independence or proficiency.

Explicit invocation creates a new, Git-ignored private draft; never overwrites
the existing signoff, frozen dataset, reconciliation or reviewer packets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.review_holdout import consolidate_reviews, read_and_validate_review
from scripts.verify_review_provenance import (
    SCHEMA, _safe_private_path, _strict_json,
)

DATASET_REL = "data/final_holdout_v1.jsonl"


def prepare_draft(
    root: Path, reviewer_files: list[str], reconciliation: str,
    output: str,
) -> dict:
    root = Path(root).resolve()
    if not isinstance(reviewer_files, list) or not 2 <= len(reviewer_files) <= 4:
        raise ValueError("Require 2–4 complete independent reviewer CSV files")
    if len(set(reviewer_files)) != len(reviewer_files):
        raise ValueError("Duplicate source reviewer file")
    data = root / DATASET_REL
    if not data.is_file() or data.is_symlink():
        raise ValueError("Frozen final benchmark is missing or symlinked")
    cases = read_jsonl(data)
    if len(cases) != 60:
        raise ValueError("Expected exactly 60 frozen final candidate cases")
    digest = sha256(data)
    paths = [_safe_private_path(root, relative, ".csv") for relative in reviewer_files]
    rec_path = _safe_private_path(root, reconciliation, ".json")
    out_path = root / output
    # Never write anywhere except private reviews/; never overwrite inputs.
    if not isinstance(output, str) or not output.startswith("reviews/"):
        raise ValueError("Draft output must be inside private reviews/ folder")
    parts = output.split("/")
    if (len(parts) < 2 or any(x in ("", ".", "..") for x in parts)
        or not output.endswith(".json") or chr(92) in output
        or chr(0) in output):
        raise ValueError("Unsafe draft output path")
    if out_path.resolve().parent != (root / "reviews").resolve():
        raise ValueError("Draft output must be directly under reviews/")
    if out_path.is_symlink() or out_path.exists() or out_path in [*paths, rec_path]:
        raise FileExistsError("Draft output already exists or overlaps input evidence")
    if (root / "reviews").is_symlink():
        raise ValueError("Private reviews/ must not be a symlink")

    # Revalidate all CSVs against the frozen data and the original reconcile
    # BEFORE writing anything. No missing/negative/disagreement accepted.
    summary = consolidate_reviews(paths, cases, digest)
    if summary["status"] != "READY_FOR_MANUAL_ADJUDICATION_AND_SIGNOFF":
        raise ValueError("Source reviewer judgments not ready for a manual draft")
    stored = _strict_json(rec_path)
    if stored != summary:
        raise ValueError("Original reconciliation differs from current reviewer files")
    records = []
    codes = []
    for path, relative in zip(paths, reviewer_files):
        reviewer, _ = read_and_validate_review(path, cases, digest)
        codes.append(reviewer)
        records.append({
            "reviewer_code": reviewer,
            "path": relative,
            "sha256": sha256(path),
        })
    draft = {
        "review_complete": False,
        "dataset_file": DATASET_REL,
        "dataset_sha256": digest,
        "reviewers": codes,
        "review_evidence": {
            "schema": SCHEMA,
            "review_files": records,
            "reconciliation": {
                "path": reconciliation,
                "sha256": sha256(rec_path),
            },
            "approvals": [
                {
                    "reviewer_code": reviewer,
                    "approved": False,
                    "independent_review_declared": False,
                    "blinded_to_model_predictions_declared": False,
                    "competent_languages": [],
                }
                for reviewer in codes
            ],
        },
        "notes": (
            "DRAFT ONLY: DO NOT mark reviewed or approved until actual qualified "
            "humans independently adjudicate and explicitly approve. A CPU "
            "file validator cannot authenticate reviewer identity or semantics."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as handle:
        json.dump(draft, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {
        "status": "MANUAL_SIGNOFF_DRAFT_ONLY_NO_APPROVAL",
        "path": str(out_path),
        "source_files": len(records),
        "review_complete": False,
        "reviewer_approvals_created": False,
        "human_review_performed_by_this_command": False,
        "model_inference_performed": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--review-csvs", nargs="+", required=True,
                   help="2–4 private reviews/*.csv paths relative to project root")
    p.add_argument("--reconciliation", required=True,
                   help="Private reviews/*.json reconciliation path")
    p.add_argument("--output", default="reviews/signoff_draft.json",
                   help="New private reviews/*.json output; never overwrites")
    args = p.parse_args()
    result = prepare_draft(
        args.root, args.review_csvs, args.reconciliation, args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
