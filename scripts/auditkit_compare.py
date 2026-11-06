"""Evaluate paired real saved model results with AuditKIT, no new generation.

Use only after both the base and fine-tuned models produced complete outputs
on the SAME frozen dataset. No model backend or API credentials are needed.
This script cannot turn a mock/dry-run into evidence of improvement.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from policyproof.core import ROOT, read_jsonl, sha256 as file_sha, score
from scripts.auditkit_replay import check_saved_outputs, make_audit_samples, make_scorers
from scripts.final_pair_diagnostics import analyze_scored_pairs
from scripts.paired_run_recovery import validate_saved_base, validate_saved_tuned
from scripts.frozen_pair_inference import GENERATION, MODEL_ID
from scripts.verify_evidence import read_evidence_json
from scripts.submission_gate import write_new_report


def validate_pair(base: dict, candidate: dict, rows: list[dict], checksum: str):
    if base.get("is_finetuned") is not False:
        raise ValueError("Baseline must be the original, untuned model")
    if candidate.get("is_finetuned") is not True:
        raise ValueError("Candidate must have verified fine-tuned inference metadata")
    if base.get("model") != candidate.get("model"):
        raise ValueError("Model IDs mismatch")
    if not candidate.get("adapter_sha256"):
        raise ValueError("Missing actual adapter SHA; not reproducible")
    # The new final candidate must never be scored as a paired experiment
    # if different base revisions, decoding rules, or splits were used.
    # Historical development artifacts predating these fields are unaffected.
    if (base.get("split") == "final_candidate_holdout_v1" or
        candidate.get("split") == "final_candidate_holdout_v1"):
        import re
        if (base.get("split") != candidate.get("split") or
            not isinstance(base.get("model_revision"), str) or
            not re.fullmatch(r"[0-9a-f]{40}", base["model_revision"]) or
            base["model_revision"] != candidate.get("model_revision")):
            raise ValueError("Final paired model revisions/splits are not identical and pinned")
        cfg = base.get("generation_config")
        if not isinstance(cfg, dict) or not cfg or cfg != candidate.get("generation_config"):
            raise ValueError("Final paired model generation settings differ or are missing")
        if (not isinstance(candidate.get("adapter_sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", candidate["adapter_sha256"])):
            raise ValueError("Full 64-hex SHA-256 adapter digest required for final pairing")
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
        "case_matched_exploratory_diagnostics": analyze_scored_pairs(cases),
        "verified_human_semantic_improvement": False,
        "limitations": [
            "Strict exact-match is sensitive to wording",
            "Citation ID match is not evidence-grounded semantic correctness",
            "Synthetic benchmark has small unsupported/injection subsets",
            "Exploratory paired uncertainty and subgroup comparisons are descriptive only, without correction for multiple comparisons",
            "Independent human model-answer review and externally verified inference still required",
            "This replays existing predictions; no model inference was performed",
        ],
    }


def compare_saved_files(
    baseline: Path, candidate: Path, dataset: Path, output: str, *,
    root: Path = ROOT,
) -> dict:
    """Read-only source comparison; write ONE new bounded report.

    Input reports may be absolute paths (e.g. preserved from a GPU session),
    but outputs are always direct artifacts/<new-name>.json in the chosen root.
    This function cannot update reviewer votes, a benchmark, model weights,
    original inference reports, or existing paired audit receipts.
    """
    root = Path(root).resolve()
    b_path, c_path, data = (
        Path(x) if Path(x).is_absolute() else root / x
        for x in (baseline, candidate, dataset)
    )
    if b_path.resolve() == c_path.resolve():
        raise ValueError("Paired comparison requires distinct source reports")
    if (data.is_symlink() or not data.is_file()
        or not 0 < data.stat().st_size <= 3_000_000):
        raise ValueError("Frozen comparison dataset missing, symlinked or oversized")
    # Duplicate fields, NaN and Infinity are not unambiguous saved evidence.
    base = read_evidence_json(b_path, limit=2_000_000)
    cand = read_evidence_json(c_path, limit=2_000_000)
    digest = file_sha(data)
    b_hash, c_hash = file_sha(b_path), file_sha(c_path)
    rows = read_jsonl(data)

    # The 60-case final split must meet the stricter complete original-output
    # validator (observed loaded SHA, matching raw scores, 60 distinct IDs).
    # This direct CLI does not independently authenticate a GPU, review or a
    # trainer and never bypasses the separate Track 1 release gate.
    if (base.get("split") == "final_candidate_holdout_v1"
        or cand.get("split") == "final_candidate_holdout_v1"):
        if (len(rows) != 60 or len({r["id"] for r in rows}) != 60
            or base.get("split") != "final_candidate_holdout_v1"
            or cand.get("split") != "final_candidate_holdout_v1"):
            raise ValueError("Final paired CLI requires exact 60-case source coverage")
        revision, adapter_sha = base.get("model_revision"), cand.get("adapter_sha256")
        base = validate_saved_base(
            b_path, rows, dataset_sha256=digest, revision=revision,
            generation_config=GENERATION, model_id=MODEL_ID,
        )
        cand = validate_saved_tuned(
            c_path, rows, dataset_sha256=digest, revision=revision,
            generation_config=GENERATION, model_id=MODEL_ID,
            adapter_sha256=adapter_sha,
        )

    result = comparison(base, cand, rows, digest)
    # Do not let a concurrent change to the source reports or benchmark
    # publish a comparison based on a mixed/unstable version.
    if (file_sha(data) != digest or file_sha(b_path) != b_hash
        or file_sha(c_path) != c_hash):
        raise ValueError("Original source evidence changed during AuditKIT comparison")
    # This creates only a previously absent report; the shared submission
    # writer restricts filenames to direct artifacts/*.json, never overwrites,
    # and uses exclusive O_EXCL semantics against competing processes.
    write_new_report(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        output, root=root,
    )
    return result


def main():
    p = argparse.ArgumentParser(
        description="CPU-only comparison of source reports; creates one NEW artifacts/*.json"
    )
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--dataset", default="data/final_holdout_v1.jsonl")
    p.add_argument("--output", default="artifacts/paired_auditkit.json")
    args = p.parse_args()
    result = compare_saved_files(
        Path(args.baseline), Path(args.candidate),
        Path(args.dataset), args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
