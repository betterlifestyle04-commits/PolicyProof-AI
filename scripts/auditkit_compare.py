"""Evaluate paired real saved model results with AuditKIT, no new generation.

Use only after both the base and fine-tuned models produced complete outputs
on the SAME frozen dataset. No model backend or API credentials are needed.
This script cannot turn a mock/dry-run into evidence of improvement.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from policyproof.core import read_jsonl, sha256 as file_sha, score
from scripts.auditkit_replay import check_saved_outputs, make_audit_samples, make_scorers


def validate_pair(base: dict, candidate: dict, rows: list[dict], checksum: str):
    if base.get("is_finetuned") is not False:
        raise ValueError("Baseline must be the original, untuned model")
    if candidate.get("is_finetuned") is not True:
        raise ValueError("Candidate must have verified fine-tuned inference metadata")
    if base.get("model") != candidate.get("model"):
        raise ValueError("Model IDs mismatch")
    if not candidate.get("adapter_sha256"):
        raise ValueError("Missing actual adapter SHA; not reproducible")
    b = check_saved_outputs(base, rows, checksum)
    c = check_saved_outputs(candidate, rows, checksum)
    # Prediction order does not matter: the saved reports may be in different
    # iteration orders while both cover precisely the same frozen IDs.
    # check_saved_outputs already validates uniqueness and full ID coverage.
    if set(b) != set(c):
        raise ValueError("Prediction sample IDs differ")
    return b, c


def comparison(base: dict, candidate: dict, rows: list[dict], checksum: str) -> dict:
    import auditkit as ak
    b, c = validate_pair(base, candidate, rows, checksum)
    scorers = make_scorers(rows)
    baseline_run = ak.evaluate(
        make_audit_samples(rows, b), model="precomputed", scorers=scorers
    )
    candidate_run = ak.evaluate(
        make_audit_samples(rows, c), model="precomputed", scorers=scorers
    )
    for run in (baseline_run, candidate_run):
        if run.errors or run.failed_count or len(run.predictions) != len(rows):
            raise RuntimeError("AuditKIT run incomplete; no improvement conclusion")
    # A paired comparison is meaningful only when the same frozen IDs appear.
    cases = {
        r["id"]: {
            "lang": r["language"],
            "category": r["category"],
            "baseline": score(r, b[r["id"]]),
            "candidate": score(r, c[r["id"]])
        } for r in rows
    }
    metrics = ["correct", "format_valid", "unsupported_claim", "raw_citation_correct",
               "abstention_correct"]
    deltas = {
        k: {
            "baseline_count": sum(v["baseline"][k] for v in cases.values()),
            "candidate_count": sum(v["candidate"][k] for v in cases.values()),
            "count_delta": sum(v["candidate"][k] - v["baseline"][k]
                               for v in cases.values()),
        }
        for k in metrics
    }
    return {
        "status": "paired_deterministic_audit",
        "sample_count": len(rows),
        "evaluation_checksum": checksum,
        "baseline_auditkit_fingerprint": baseline_run.fingerprint,
        "candidate_auditkit_fingerprint": candidate_run.fingerprint,
        "baseline_model_revision": base.get("model_revision"),
        "adapter_sha256": candidate["adapter_sha256"],
        "auditkit_baseline": baseline_run.headline,
        "auditkit_candidate": candidate_run.headline,
        "case_matched_deterministic_counts": deltas,
        "verified_human_semantic_improvement": False,
        "limitations": [
            "Strict exact-match is sensitive to wording",
            "Citation ID match is not evidence-grounded semantic correctness",
            "Synthetic benchmark has small unsupported/injection subsets",
            "Separate human review and uncertainty estimates required",
            "This replays existing predictions; no model inference was performed",
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--dataset", default="data/final_holdout_v1.jsonl")
    p.add_argument("--output", default="artifacts/paired_auditkit.json")
    args = p.parse_args()
    b_path, c_path = Path(args.baseline), Path(args.candidate)
    data = Path(args.dataset)
    base = json.loads(b_path.read_text(encoding="utf-8"))
    cand = json.loads(c_path.read_text(encoding="utf-8"))
    result = comparison(base, cand, read_jsonl(data), file_sha(data))
    dest = Path(args.output)
    if dest.resolve() in {b_path.resolve(), c_path.resolve()}:
        raise ValueError("Refusing to overwrite original model predictions")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved:", dest)


if __name__ == "__main__":
    main()
