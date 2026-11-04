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


def readiness(root:Path=ROOT) -> dict:
    root=Path(root)
    checks={}
    old=root/"artifacts/track2a_safety/track2a_chain_report.json"
    actual_chain=False
    if old.is_file():
        try:
            r=json.loads(old.read_text(encoding="utf-8"))
            actual_chain=(
                r.get("status")=="curatorkit_auditkit_chain_complete"
                and r.get("case_count")==48
                and r.get("curatorkit_version")=="0.1.22"
                and r.get("auditkit_version")=="1.1.3"
                and r.get("human_labels_independently_reviewed") is False
                and r.get("model",{}).get("new_inference_performed") is False
                and r.get("curation",{}).get("label_and_case_id_integrity_verified") is True
            )
        except (OSError,ValueError,TypeError):
            pass
    checks["real_curatorkit_and_auditkit_chain"]={
        "passed":bool(actual_chain),
        "details":"48 previously saved base-model outputs, genuine toolchain and exact curation integrity"
    }
    newer=root/"artifacts/track2a_challenge_unscored/challenge_manifest.json"
    challenge=False
    if newer.is_file():
        try:
            m=json.loads(newer.read_text(encoding="utf-8"))
            challenge=(m.get("status")=="curated_unscored"
                and m.get("cases")==72 and m.get("curated_cases")==72
                and m.get("model_predictions_available") is False
                and m.get("independent_human_review") is False)
        except (OSError,ValueError,TypeError):
            pass
    checks["unscored_challenge_curated"]={
        "passed":bool(challenge),
        "details":"72 attack/control cases curated, but not scored or independently reviewed"
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
    file=Path(args.output)
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(r,ensure_ascii=False,indent=2))
    if args.strict and not r["ready_to_submit"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
