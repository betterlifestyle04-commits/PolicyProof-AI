"""Produce a truthful, local-only copy/paste handoff for the official Lexsi form.

OFFICIAL SOURCE (checked 2026-11-09): https://hackathon.lexsi.ai/
Submission form: https://hackathon.lexsi.ai/submission/
Deadline: 2026-11-15 11:59 PM Anywhere on Earth (AoE) (live organizer site 2026-11-10).

This script DOES NOT log in, browse the form, infer a registration email, train
a model, upload to Hugging Face, approve licensing, or submit anything.
Unverified URL fields are NEVER borrowed from an owner-editable manifest.
Local source consistency is distinct from public access and organizer receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from policyproof.core import ROOT, sha256
from scripts.research_report_io import preserve_deterministic_report
from scripts.track2a_readiness import readiness
from scripts.verify_hf_collections import validate_manifest

ABSTRACT_FILE = "docs/TRACK2A_SUBMISSION_ABSTRACT.md"
NEW_NATIVE40_REPORT = "docs/TRACK2A_NATIVE40_DEV48_OBSERVED_PRIVATE_RESULTS_2026-11-10.md"
HF_MANIFEST = "docs/hf_publication_manifest.json"
OFFICIAL_FORM = "https://hackathon.lexsi.ai/submission/"
REPO = "https://github.com/betterlifestyle04-commits/PolicyProof-AI"
DEMO = "https://policyproof-ai-observatory.onrender.com"
BASE = "CohereLabs/tiny-aya-global"
MAX_ABSTRACT_BYTES = 15_000
MAX_ABSTRACT_WORDS = 499
# Each phrase represents a hard boundary between true historical evidence and
# unsupported final quality or model-weight improvement claims.
_REQUIRED_HISTORICAL = (
    "48 previously saved tiny aya base-model responses",
    "33/48",
    "8/48",
    "20/39",
    "72-case",
    "no independent semantic review",
    "not safetune steering or model fine-tuning",
)


def _abstract(root: Path) -> tuple[str, int, str]:
    src = Path(root) / ABSTRACT_FILE
    if (src.is_symlink() or not src.is_file()
        or not 0 < src.stat().st_size <= MAX_ABSTRACT_BYTES):
        raise ValueError("Track 2a abstract is missing, symlinked, empty or oversized")
    original = src.read_bytes()
    try:
        doc = original.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError("Abstract must be UTF-8") from exc
    if not doc.startswith("# PolicyProof AI — Track 2a alternative abstract (draft)\n"):
        raise ValueError("Expected original owner-reviewed Track 2a abstract heading")
    # Only the content after the exact Markdown title is pasted in the form.
    body = doc.split("\n", 1)[1].strip()
    if not body or "\x00" in body or re.search(r"<[/!a-zA-Z]", body):
        raise ValueError("Abstract is blank or has unsafe raw markup")
    count = len(body.split())
    if not 50 <= count <= MAX_ABSTRACT_WORDS:
        raise ValueError(
            "Official form requires abstract UNDER 500 words; got " + str(count)
        )
    # Guard the known factual claims against silently changing historical
    # numbers; this is a mechanical check, NOT independent semantic review.
    folded = body.casefold()
    if not all(phrase in folded for phrase in _REQUIRED_HISTORICAL):
        raise ValueError("Track 2a abstract no longer preserves the source-limited original scope")
    if not (
        "no model was run on these new cases" in folded
        or "source-replayed" in folded
    ):
        raise ValueError("72-case model status no longer bounded to saved source evidence")
    return body, count, sha256(src)


def handoff(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    body, words, abstract_sha = _abstract(root)
    track = readiness(root)
    checks = track["checks"]
    required_checks = (
        "real_curatorkit_and_auditkit_chain",
        "unscored_challenge_curated",
        "public_research_writeup",
        "public_ungated_hf_collections",
    )
    scientific = all(checks[name]["passed"] is True for name in required_checks[:3])
    public_hf = checks["public_ungated_hf_collections"]["passed"] is True
    saved72 = track["saved_72_source_replay"]
    # The draft is intentionally historical and says NO 72-case model run
    # occurred; once actual saved 72 evidence is accepted, this claim is
    # obsolete and requires a new human-reviewed abstract before form paste.
    # The historical 72-case no-inference claim is mandatory while the
    # challenge is unscored, and forbidden once a fully saved 72-source replay
    # qualifies. This two-sided check enables a truthful *revised* abstract.
    measured72 = track["new_model_evaluation_claimed_for_72_cases"] is True
    old_no_72_model_claim = "no model was run on these new cases" in body.casefold()
    stale_72 = measured72 and (
        old_no_72_model_claim or "unscored 72-case" in body.casefold()
    )
    unsupported_72_claim = not measured72 and not old_no_72_model_claim
    stale_hf = public_hf and "still pending" in body.casefold()
    native_report = root / NEW_NATIVE40_REPORT
    native40_source_received = (
        native_report.is_file() and not native_report.is_symlink()
        and 0 < native_report.stat().st_size < 200_000
    )
    # A newer, source-hashed private evaluation cannot safely be pasted into
    # an old abstract that says no model tuning/evaluation was performed.
    # This flag deliberately BLOCKS form copying; it never infers that public
    # model weights or external GPU attestations exist.
    stale_native40 = native40_source_received and not (
        "native aligntune" in body.casefold()
        and "4/9" in body and "2/9" in body
        and "0/39" in body
        and "no safety" in body.casefold()
    )
    stale_abstract = stale_72 or stale_hf or unsupported_72_claim or stale_native40
    verified_model_collection = None
    verified_dataset_collection = None
    verified_dataset_repo = None
    if public_hf:
        # A current anonymous receipt is required by readiness(); still
        # validate the exact canonical owner's links. Owner-entered "verified"
        # flags by themselves cannot unlock these outputs.
        manifest = validate_manifest(root / HF_MANIFEST)
        verified_model_collection = manifest["model_collection_url"]
        verified_dataset_collection = manifest["dataset_collection_url"]
        verified_dataset_repo = manifest["dataset_url"]
    local_ready = bool(track["ready_to_submit"] and not stale_abstract)
    missing = []
    if not scientific:
        missing.append("Original 48 saved model-output + native CuratorKIT/AuditKIT source-chain validation")
    if track["conflicting_optional_evidence_blocks_release"]:
        missing.append("Remove or independently repair invalid optional 72-case source evidence")
    if stale_abstract:
        missing.append(
            "Rewrite and source-check abstract for the privately observed native40 "
            "paired evaluation, newly accepted 72-case results and public HF state"
        )
    if not public_hf:
        missing.append("Owner-authorized public, ungated HF model and dataset collections with fresh anonymous readback")
    # The registration email is always a private human-entered value; never
    # place it into a public ZIP or source-controlled generated manifest.
    missing.extend((
        "Owner enters registration email privately in the official form",
        "Owner checks rights and license for every public model/dataset artifact",
        "Owner independently opens public README, GitHub, demo and HF links",
        "Owner submits the official form and retains actual organizer confirmation",
    ))
    return {
        "schema": "policyproof_lexsi_track2a_owner_form_handoff_v1",
        "status": (
            "LOCAL_EVIDENCE_READY_MANUAL_FORM_REQUIRED"
            if local_ready else
            "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED"
            if stale_abstract else
            "DRAFT_ONLY_MISSING_PUBLICATION_OR_RESEARCH"
        ),
        "official_submission_deadline": "2026-11-15 11:59 PM AoE",
        "official_form_url": OFFICIAL_FORM,
        "official_form_opened_by_script": False,
        "registered_email": None,
        "email_lookup_performed": False,
        "form_fields": {
            "project_title": "PolicyProof AI — multilingual policy safety evidence benchmark",
            "abstract_text": None if stale_abstract else body,
            "abstract_word_count": words,
            "abstract_under_500_words": True,
            "abstract_source_sha256": abstract_sha,
            "track": "Track 2a — AI Safety and Red Teaming",
            "problem_statement": "Adversarial and safety evaluation sets",
            "toolkits_used_in_verified_chain": ["CuratorKIT", "AuditKIT"] if scientific else [],
            "model_used_historically": BASE,
            "github_repository": REPO,
            "github_anonymously_accessed_by_script": False,
            "writeup_or_article": REPO + "/blob/main/README.md",
            "public_demo": DEMO,
            "demo_anonymously_accessed_by_script": False,
            "hf_model_collection": verified_model_collection,
            "hf_dataset_collection": verified_dataset_collection,
            "hf_dataset_repository": verified_dataset_repo,
            "hf_anonymous_access_receipt_current": public_hf,
        },
        "research_evidence": {
            "local_track2a_status": track["status"],
            "original_48_saved_predictions_and_toolchain_source_replayed": scientific,
            "original_72_model_evaluation_locally_source_replayed": bool(
                saved72["cases"] == 72 and track["new_model_evaluation_claimed_for_72_cases"]
            ),
            "unscored_72_claim_must_not_be_pasted_if_new_results_exist": stale_72,
            "new_72_model_results_cannot_be_claimed_without_source_replay": unsupported_72_claim,
            "native_aligntune_training_improvement_claimed": False,
            "native40_private_paired_dev48_report_present": native40_source_received,
            "historical_abstract_blocked_by_new_native40_results": stale_native40,
            "independent_human_semantic_accuracy_verified": False,
        },
        "required_actions": missing,
        "eligible_for_manual_form_review_locally": local_ready,
        "registered_email_verified": False,
        "owner_license_and_rights_approved_by_script": False,
        "official_submission_completed": False,
        "organizer_confirmation_verified": False,
        "external_network_used": False,
        "warning": (
            "A LOCAL evidence handoff is never an official submission. Current "
            "Hugging Face URLs appear only after a fresh source-bound anonymous "
            "verification receipt; your registration email and rights consent "
            "must be supplied privately and manually. Original GPU provenance "
            "and human semantic correctness are not independently attested."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Exit 2 if HF collections or historical abstract are unready")
    parser.add_argument("--output", default=None,
                        help="Optional new immutable direct artifacts/<name>.json")
    args = parser.parse_args()
    report = handoff()
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        preserve_deterministic_report(payload, args.output)
    print(payload, end="")
    if args.strict and not report["eligible_for_manual_form_review_locally"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
