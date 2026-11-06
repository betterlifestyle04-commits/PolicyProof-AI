"""AuditKIT offline replay of real PolicyProof model outputs.

This is a real AuditKIT run over SAVED predictions, not another model
generation. It reports strict deterministic diagnostics only, never
misrepresents these scores as human semantic accuracy or post-training gains.

Usage:
 python -m pip install "auditkit>=1.0,<2"
 python -m scripts.auditkit_replay \
   --report artifacts/tiny_aya_base_unseen_policy-rescored-v2.json \
   --dataset data/unseen_policy_test.jsonl

No GPU, Hugging Face token, or external judge API is required.
"""
from __future__ import annotations
from scripts.verify_evidence import read_evidence_json
from scripts.research_report_io import preserve_deterministic_report
import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path

from policyproof.core import read_jsonl, sha256 as file_sha, score


def check_saved_outputs(report: dict, rows: list[dict], checksum: str) -> dict:
    """Require genuine previously saved predictions on the exact frozen split."""
    if report.get("engine") != "transformers" or report.get("is_real_model_result") is not True:
        raise ValueError("Only genuine previously recorded Transformers inference is accepted")
    if report.get("dataset_sha256") != checksum:
        raise ValueError("Dataset SHA mismatch; cannot replay against changed questions")
    predictions = report.get("predictions", [])
    if len(predictions) != len(rows):
        raise ValueError("Missing or extra prediction records")
    mapping = {}
    for p in predictions:
        if not isinstance(p.get("id"), str) or not isinstance(p.get("raw"), str):
            raise ValueError("Malformed prediction entry")
        if p["id"] in mapping:
            raise ValueError("Duplicate prediction ID")
        mapping[p["id"]] = p["raw"]
    ids = {r["id"] for r in rows}
    if set(mapping) != ids or len(ids) != len(rows):
        raise ValueError("Incomplete prediction coverage or duplicated dataset IDs")
    return mapping


def make_audit_samples(rows: list[dict], predictions: dict[str, str]):
    import auditkit as ak
    from policyproof.core import make_prompt, make_gold
    return [
        ak.Sample(
            id=r["id"],
            input=make_prompt(r),
            target=make_gold(r["expected_answer"], r["expected_source_id"]),
            actual_output=predictions[r["id"]],
            task=r["category"],
            metadata={"language": r["language"], "case_id": r["id"]},
        )
        for r in rows
    ]


def make_scorers(rows: list[dict]):
    import auditkit as ak
    lookup = {r["id"]: r for r in rows}

    def metric(name: str, key: str):
        @ak.scorer(direction=ak.Direction.MAXIMIZE, name=name)
        def apply(sample, output):
            if sample.id not in lookup:
                raise ValueError("Unknown audit sample ID")
            return float(score(lookup[sample.id], output)[key])
        return apply

    return [
        metric("policyproof_strict_exact", "correct"),
        metric("policyproof_json_schema_valid", "format_valid"),
        metric("policyproof_raw_expected_source", "raw_citation_correct"),
        metric("policyproof_correct_abstention", "abstention_correct"),
    ]


def audit_replay(report: dict, rows: list[dict], checksum: str) -> dict:
    mapping = check_saved_outputs(report, rows, checksum)
    import auditkit as ak

    samples = make_audit_samples(rows, mapping)
    run = ak.evaluate(
        samples,
        model="precomputed",
        scorers=make_scorers(rows),
        experiment_name=None,  # no uploads or external telemetry
    )
    if run.failed_count or run.errors:
        raise RuntimeError(f"AuditKIT scoring failures: {run.failed_count}; {run.errors}")
    if len(run.predictions) != len(rows):
        raise RuntimeError("AuditKIT output count differs from original inputs")
    # Raw scores are *unconditional* 0/1 diagnostics; abstention_correct and
    # raw source are NOT conditional recall denominators, so derive conditional
    # percentages from policyproof.core if reporting such quantities.
    return {
        "status": "complete",
        "auditkit_version": importlib.metadata.version("auditkit"),
        "auditkit_run_id": run.run_id,
        "auditkit_fingerprint": run.fingerprint,
        "sample_count": len(rows),
        "headline_unconditional_averages": run.headline,
        "source": "previously_saved_real_model_predictions",
        "model": report.get("model"),
        "model_revision": report.get("model_revision"),
        "dataset_sha256": checksum,
        "is_post_training_evaluation": bool(report.get("is_finetuned")),
        "model_inference_executed": False,
        "model_accuracy_semantically_verified": False,
        "human_review_completed": False,
        "limitations": [
            "All benchmark questions and policy documents are fictional",
            "Strict exact text matching is not equivalent to semantic correctness",
            "Citation and abstention scorers shown as unconditional per-case averages",
            "No newly generated model outputs and no fine-tuning validated by this replay",
            "The 48-case split was inspected during development and cannot be final blind holdout",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--dataset", default="data/unseen_policy_test.jsonl")
    p.add_argument("--output", default="artifacts/auditkit_baseline_replay.json")
    args = p.parse_args()
    source = Path(args.report)
    rows_path = Path(args.dataset)
    report = read_evidence_json(source)
    result = audit_replay(report, read_jsonl(rows_path), file_sha(rows_path))
    dest, reused = preserve_deterministic_report(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("AuditKIT output " + ("verified unchanged" if reused else "saved") + ":", dest)


if __name__ == "__main__":
    main()
