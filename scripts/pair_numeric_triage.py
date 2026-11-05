"""Private, reproducible *paired* numeric triage from the original saved GPU ZIP.

Run after preserving the original ZIP and verifying its exact SHA. No GPU,
training, inference or human semantic grading. The report is owner-only:
sharing it with blinded reviewers before voting would unblind model identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

from policyproof.core import ROOT, sha256
from scripts.dev48_gpu_pair import check_dev_rows, read_validated_report
from scripts.export_review_from_bundle import (
    ORIGINAL_DEV48_ZIP_SHA256, verify_bundle,
)
from scripts.numeric_evidence_audit import audit_saved_predictions
from scripts.source_topic_audit import audit_saved_source_topics
from scripts.paired_semantic_review import _private_path


def make_pair_numeric_triage(root: Path, bundle: Path, trusted_zip_sha: str,
                             destination: str = "reviews/dev48_numeric_triage_owner_only.json") -> dict:
    root = Path(root).resolve()
    target = _private_path(root, destination, ".json", new=True)
    payloads, meta = verify_bundle(bundle, trusted_zip_sha)
    cases, dataset_sha = check_dev_rows(root / "data/unseen_policy_test.jsonl")
    with tempfile.TemporaryDirectory(prefix="policyproof-numeric-pair-") as private:
        private_dir = Path(private)
        base_path = private_dir / "dev48_base.json"
        tuned_path = private_dir / "dev48_tuned.json"
        base_path.write_bytes(payloads["dev48_base.json"])
        tuned_path.write_bytes(payloads["dev48_tuned.json"])
        base_info = json.loads(payloads["dev48_base.json"])
        tuned_info = json.loads(payloads["dev48_tuned.json"])
        revision = base_info["model_revision"]
        adapter_sha = tuned_info["adapter_sha256"]
        original_training_zip_sha = tuned_info["adapter_source_zip_sha256"]
        base = read_validated_report(
            base_path, cases, revision=revision, data_sha=dataset_sha,
            adapter_sha=None, zip_sha=original_training_zip_sha,
        )
        tuned = read_validated_report(
            tuned_path, cases, revision=revision, data_sha=dataset_sha,
            adapter_sha=adapter_sha, zip_sha=original_training_zip_sha,
        )
        base_audit = audit_saved_predictions(base, cases, dataset_sha)
        tuned_audit = audit_saved_predictions(tuned, cases, dataset_sha)
        base_topics = audit_saved_source_topics(base, cases, dataset_sha)
        tuned_topics = audit_saved_source_topics(tuned, cases, dataset_sha)

    base_flags = {entry["id"] for entry in base_audit["flagged_cases"]}
    tuned_flags = {entry["id"] for entry in tuned_audit["flagged_cases"]}
    base_topic_flags = {entry["id"] for entry in base_topics["flagged_cases"]}
    tuned_topic_flags = {entry["id"] for entry in tuned_topics["flagged_cases"]}
    result = {
        "status": "PRIVATE_DEV48_PAIRED_NUMERIC_REVIEW_TRIAGE_ONLY",
        "research_scope": "already_inspected_synthetic_development_not_final",
        "source_bundle_sha256": meta["source_zip_sha256"],
        "dataset_sha256": dataset_sha,
        "requested_base_revision": revision,
        "adapter_sha256": adapter_sha,
        "samples_per_model": len(cases),
        "base": base_audit,
        "tuned": tuned_audit,
        "base_question_source_topic_audit": base_topics,
        "tuned_question_source_topic_audit": tuned_topics,
        "topic_newly_flagged_case_ids": sorted(tuned_topic_flags - base_topic_flags),
        "topic_no_longer_flagged_case_ids": sorted(base_topic_flags - tuned_topic_flags),
        "newly_flagged_case_ids": sorted(tuned_flags - base_flags),
        "no_longer_flagged_case_ids": sorted(base_flags - tuned_flags),
        "model_training_performed": False,
        "model_inference_performed": False,
        "human_semantic_review_completed": False,
        "semantic_model_improvement_verified": False,
        "reviewer_blinding_required": True,
        "do_not_share_with_reviewers": True,
        "limitations": [
            "Numeric and topic flags are syntactic review signals, not independent factual judgments",
            "A topic match does NOT establish answer accuracy or evidence entailment",
            "A flagged topic mismatch requires human investigation, not automatic rejection",
            "A financial amount can be literally present but assigned to the wrong product, time or condition",
            "Differences in alert counts do not prove improvements or regressions",
            "Do not reveal model labels or this report to reviewers until their independent blind votes are locked",
            "The ZIP metadata and raw outputs are internally checked, not independent confirmation of CUDA execution",
            "The 48 development samples were previously inspected and cannot be called a blind final benchmark",
        ],
    }
    target.parent.mkdir(exist_ok=True, parents=True)
    with target.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")
    return {
        "status": result["status"],
        "private_report": str(target),
        "source_bundle_sha256": meta["source_zip_sha256"],
        "samples_per_model": len(cases),
        "base_number_review_flags": base_audit["numeric_review_flags"],
        "tuned_number_review_flags": tuned_audit["numeric_review_flags"],
        "base_topic_mismatch_review_flags":
            base_topics["topic_mismatch_or_ambiguous_source_review_flags"],
        "tuned_topic_mismatch_review_flags":
            tuned_topics["topic_mismatch_or_ambiguous_source_review_flags"],
        "newly_topic_flagged": len(tuned_topic_flags - base_topic_flags),
        "no_longer_topic_flagged": len(base_topic_flags - tuned_topic_flags),
        "newly_flagged": len(tuned_flags - base_flags),
        "no_longer_flagged": len(base_flags - tuned_flags),
        "human_semantic_review_performed": False,
        "model_inference_performed": False,
        "model_improvement_claim": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=ORIGINAL_DEV48_ZIP_SHA256)
    parser.add_argument("--output", default="reviews/dev48_numeric_triage_owner_only.json")
    args = parser.parse_args()
    result = make_pair_numeric_triage(args.root, args.bundle, args.expected_sha256, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
