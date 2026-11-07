"""Track 2a-specific readiness, separate from unfinished Track 1 fine-tuning.

A benchmark + report track does NOT require claiming a tuned adapter.
However Lexsi's submission form asks for public and ungated Hugging Face
model and dataset collections. Until that is resolved, report NOT READY.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json

from policyproof.core import ROOT
from scripts.verify_hf_collections import receipt_is_current
from scripts.track2a_source_evidence import verify_original_chain, verify_unscored_challenge
from scripts.research_report_io import preserve_deterministic_report


def readiness(root:Path=ROOT) -> dict:
    root=Path(root)
    checks={}
    # A plausible status field in an owner-editable JSON file is NOT toolkit
    # evidence. Recompute frozen source hashes, all 48 predictions, grouped
    # scores and exact CuratorKIT/72-case exported labels instead.
    try:
        chain_proof = verify_original_chain(root)
        actual_chain = chain_proof["source_curation_and_score_recomputed"] is True
        chain_details = (
            "Exact 48-case source SHA, raw Tiny Aya outputs, CuratorKIT labels "
            "and grouped AuditKIT chain receipts rechecked on CPU"
        )
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        actual_chain = False
        chain_details = "Source-linked CuratorKIT/AuditKIT evidence missing or invalid: " + str(exc)[:180]
    checks["real_curatorkit_and_auditkit_chain"] = {
        "passed": bool(actual_chain), "details": chain_details,
    }
    try:
        challenge_proof = verify_unscored_challenge(root)
        challenge = challenge_proof["curated_cases"] == 72
        challenge_details = (
            "Exact unscored 72-case SHA, all source labels and actual curated "
            "export checked; NOT independently reviewed or scored by a model"
        )
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        challenge = False
        challenge_details = "Frozen/curated 72-case source evidence missing or invalid: " + str(exc)[:180]
    checks["unscored_challenge_curated"] = {
        "passed": bool(challenge), "details": challenge_details,
    }
    docs=(root/"docs/TRACK2A_SUBMISSION_ABSTRACT.md").is_file() and (
        root/"docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md").is_file()
    checks["public_research_writeup"]={
        "passed":docs,
        "details":"Track 2a research report and alternative abstract in repository"
    }
    # NEVER trust booleans manually entered into the publication manifest.
    # A separate, recent, anonymous HF Hub API verification receipt is
    # required and must bind exactly to the current manifest SHA256/links.
    # Such a receipt is still not a signed third-party proof or organizer approval.
    manifest=root/"docs/hf_publication_manifest.json"
    receipt=root/"artifacts/hf_anonymous_access_receipt.json"
    public=receipt_is_current(receipt,manifest)
    checks["public_ungated_hf_collections"]={
        "passed":bool(public),
        "details":(
            "Recent anonymous HF API receipt matches exact manifest SHA, "
            "actual model/dataset collection membership, public ungated repos, "
            "and downloaded 48/72 research-file provenance; publication "
            "and Track 2a model collection relevance still require human review"
        )
    }
    mandatory=("real_curatorkit_and_auditkit_chain",
               "unscored_challenge_curated","public_research_writeup",
               "public_ungated_hf_collections")
    scientific_ready=all(checks[k]["passed"] for k in mandatory[:3])
    eligible_ready=scientific_ready and checks["public_ungated_hf_collections"]["passed"]
    return {
        "track":"2a_adversarial_safety_evaluation",
        "status":(
            "READY_FOR_MANUAL_SUBMISSION_REVIEW" if eligible_ready
            else "RESEARCH_READY_HF_PUBLICATION_BLOCKED" if scientific_ready
            else "RESEARCH_INCOMPLETE"
        ),
        "ready_to_submit":bool(eligible_ready),
        "checks":checks,
        "training_claimed":False,
        "new_model_evaluation_claimed_for_72_cases":False,
        "official_submission_confirmed":False,
        "advisory":"Confirm with organizer how a benchmark-only entry should fill a model collection; do not falsify one.",
        "limitations":"The 48 scored cases are previously inspected and synthetic; the 72 challenge cases are unscored; no independent semantic review."
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--strict",action="store_true")
    p.add_argument("--output",default="artifacts/track2a_readiness.json")
    args=p.parse_args()
    r=readiness()
    # Readiness reports are evidence snapshots, not scratch logs: only new
    # direct artifacts/*.json outputs, or byte-identical idempotent replays.
    preserve_deterministic_report(
        json.dumps(r, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        args.output,
    )
    print(json.dumps(r,ensure_ascii=False,indent=2))
    if args.strict and not r["ready_to_submit"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
