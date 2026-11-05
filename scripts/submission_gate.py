"""Read-only, fail-closed Lexsi submission readiness validator.

Run python -m scripts.submission_gate (or add --strict to fail incomplete).
No network calls or inference. Existence and metadata checks cannot replace
external manual verification of model quality, license or URL access.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import re

from policyproof.core import ROOT, sha256
from scripts.verify_evidence import verify_baseline, read_evidence_json
from scripts.verify_training_archive import inspect_training_directory
from scripts.submission_evidence import (
    paired_evaluation_from_sources, public_hf_evidence, source_linked_review,
)


# A roadmap is not a substitute for training, review, licensing, or submission.
# Keep evidence actions separate from the passing criteria; they are derived
# solely from the validator's actual measured booleans.
_EVIDENCE_ACTIONS = (
    {
        "check": "saved_baseline",
        "title": "Verify the unchanged base-model baseline",
        "work_type": "CPU evidence verification",
        "action": "Recover the original 48 saved Tiny Aya predictions and exact development dataset; run the read-only baseline validator.",
        "proof_needed": "Saved raw predictions, matching dataset SHA-256 and reproducible scores",
        "depends_on": (),
    },
    {
        "check": "trained_adapter",
        "title": "Produce a genuine native AlignTune model adapter",
        "work_type": "User-operated GPU training",
        "action": "On an eligible GPU, execute the opt-in native AlignTune 40-step Colab run using the audited 78-row dataset. Preserve original logs and unmodified adapter ZIP.",
        "proof_needed": "Native trainer logs, observed loaded-model commit, full adapter hashes and validated training receipt",
        "depends_on": (),
    },
    {
        "check": "human_review_advisory",
        "title": "Review the 60-case frozen gold labels independently",
        "work_type": "Human bilingual review",
        "action": "Have at least two competent reviewers inspect all English, Norwegian and Arabic source labels without seeing model outputs; preserve distinct original review files.",
        "proof_needed": "Original reviewer CSVs, unanimous or adjudicated source judgments and SHA-bound reconciliation",
        "depends_on": (),
    },
    {
        "check": "paired_finetuned_evaluation",
        "title": "Run the frozen paired base-versus-trained evaluation",
        "work_type": "GPU inference and CPU audit",
        "action": "After native training and gold review, run both models on the same 60 frozen cases with identical decoding, then preserve raw outputs and recomputed paired scores.",
        "proof_needed": "120 original model responses, adapter and benchmark digests, per-case scores and AuditKIT receipts",
        "depends_on": ("trained_adapter", "human_review_advisory"),
    },
    {
        "check": "huggingface_artifacts",
        "title": "Publish and verify authorized Hugging Face artifacts",
        "work_type": "Owner-approved external publication",
        "action": "Verify license rights, then publish only authorized adapter, dataset and logs; obtain a fresh anonymous Hub-access receipt bound to the publication manifest.",
        "proof_needed": "Authorized published repositories, verified revision and manifest-linked anonymous access receipt",
        "depends_on": ("trained_adapter",),
    },
    {
        "check": "public_hf_collections",
        "title": "Verify both public Hugging Face collections",
        "work_type": "External access verification",
        "action": "Confirm anonymous, ungated access and correct model/dataset collection memberships; recheck freshness before submitting.",
        "proof_needed": "Fresh receipt with separate accessible model and dataset collections and their required repositories",
        "depends_on": ("huggingface_artifacts",),
    },
    {
        "check": "public_repo_materials",
        "title": "Keep source code and submission narrative available",
        "work_type": "Documentation",
        "action": "Maintain public source, research limitations and a factual submission write-up; independently open the site as a judge.",
        "proof_needed": "README, submission draft and independently accessible public repository",
        "depends_on": (),
    },
)


def next_evidence_actions(checks: dict, mandatory_checks: tuple | list) -> list[dict]:
    """Deterministic list of unmet local gates and their prerequisites.

    ACTIONABLE means dependencies *within this local gate* are satisfied; it
    does not mean GPU capacity, reviewer availability or external publishing
    permission is available.
    """
    mandatory = set(mandatory_checks)
    actions = []
    for entry in _EVIDENCE_ACTIONS:
        check = entry["check"]
        if check not in mandatory or checks[check]["passed"] is True:
            continue
        waiting = [
            dependency for dependency in entry["depends_on"]
            if checks[dependency]["passed"] is not True
        ]
        actions.append({
            "priority": len(actions) + 1,
            "requirement": check,
            "title": entry["title"],
            "work_type": entry["work_type"],
            "state": "WAITING_FOR_DEPENDENCIES" if waiting else "ACTIONABLE_LOCALLY",
            "waiting_on": waiting,
            "action": entry["action"],
            "proof_needed": entry["proof_needed"],
            "action_performed_by_report": False,
        })
    if not actions:
        actions.append({
            "priority": 1,
            "requirement": "manual_release_review",
            "title": "Perform final human release and official submission review",
            "work_type": "Human license, accuracy and organizer verification",
            "state": "ACTIONABLE_LOCALLY",
            "waiting_on": [],
            "action": "Revalidate public Hugging Face access, license rights and actual GPU and reviewer evidence; submit through the official process and retain confirmation.",
            "proof_needed": "Independent release signoff and official organizer submission confirmation",
            "action_performed_by_report": False,
        })
    return actions


def evaluate_readiness(root=ROOT):
    root = Path(root)
    checks = {}
    baseline=root/"artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json"
    evalset=root/"data/unseen_policy_test.jsonl"
    valid_baseline=False
    if baseline.exists() and evalset.exists():
        try:
            valid_baseline=verify_baseline(baseline,evalset)["sample_count"] == 48
        except (ValueError,KeyError,TypeError,json.JSONDecodeError):
            pass
    checks["saved_baseline"]={"passed":valid_baseline,"details":"48 unchanged-model responses and exact dataset/scoring consistency; not independent re-inference"}

    audit=root/"artifacts/auditkit_actual_baseline_replay.json"
    valid_audit=False
    if audit.exists() or audit.is_symlink():
        try:
            receipt=read_evidence_json(audit, limit=2_000_000)
            baseline_metadata=read_evidence_json(baseline) if valid_baseline else {}
            valid_audit=(
                receipt.get("status")=="complete"
                and type(receipt.get("sample_count")) is int
                and receipt["sample_count"]==48
                and valid_baseline
                and receipt.get("dataset_sha256")==sha256(evalset)
                and receipt.get("model")=="CohereLabs/tiny-aya-global"
                and receipt.get("model_revision")==baseline_metadata.get("model_revision")
                and receipt.get("source")=="previously_saved_real_model_predictions"
                and receipt.get("is_post_training_evaluation") is False
                and receipt.get("model_inference_executed") is False
                and receipt.get("model_accuracy_semantically_verified") is False
                and receipt.get("human_review_completed") is False
                and isinstance(receipt.get("auditkit_version"),str)
                and bool(receipt["auditkit_version"])
                and isinstance(receipt.get("auditkit_fingerprint"),str)
                and bool(receipt["auditkit_fingerprint"])
            )
        except (ValueError,OSError,TypeError):
            pass
    checks["baseline_auditkit"]={"passed":valid_audit,"details":"Recorded baseline replay through AuditKIT; no training"}

    # A self-reported run_info.json plus an arbitrary non-empty file MUST
    # NOT mark a model as trained. Verify actual PEFT safetensors structure,
    # both file hashes, curated-data digest, source commit and backend claim.
    # Even this does not independently prove GPU execution or model quality.
    model_completed=False
    structurally_checked=False
    adapter_notes=[]
    verified_adapters=[]
    for weights in (root/"artifacts").rglob("adapter_model.safetensors"):
        try:
            verified=inspect_training_directory(weights.parent.parent)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            adapter_notes.append(type(exc).__name__ + ": " + str(exc)[:130])
            continue
        structurally_checked=True
        verified_adapters.append((weights.parent.parent, verified))
        if verified["track1_submission_evidence_candidate"]:
            model_completed=True
    checks["adapter_artifact_integrity"]={
        "passed":structurally_checked,
        "details":"CPU-checked safetensors shape, LoRA tensors and SHA256 metadata; training execution remains self-reported",
    }
    checks["trained_adapter"]={
        "passed":model_completed,
        "details":"Structurally validated 10+ optimizer-step receipt, pinned base revision and declared AlignTune trainer (NOT independently verified training)",
    }

    # A boolean in a saved JSON document is never sufficient evidence for
    # human review. Recompute hashes, the source reviewer CSV judgments,
    # reconciliation, and the exact 60-case frozen dataset.
    reviewed, review_details = source_linked_review(root)
    checks["human_review_advisory"] = {
        "passed": reviewed,
        "details": review_details,
    }

    # A plausible AuditKIT status alone is not paired research evidence.
    # Validate all 120 preserved raw outputs, 60x2 scores, common revision,
    # adapter digest, generation options and recomputed case-matched deltas.
    paired, paired_details = paired_evaluation_from_sources(
        root, verified_adapters, review_is_valid=reviewed,
    )
    checks["paired_finetuned_evaluation"] = {
        "passed": paired,
        "details": paired_details,
    }

    # Never accept "external_access_verified": true or plausible-looking
    # Hugging Face links as proof. Require a fresh manifest-bound anonymous
    # remote-access receipt and verify model/adapter/data/log membership.
    published, collections_valid, hf_details = public_hf_evidence(root)
    checks["huggingface_artifacts"] = {
        "passed": published,
        "details": hf_details,
    }
    checks["public_hf_collections"] = {
        "passed": collections_valid,
        "details": hf_details,
    }

    public_materials=(root/"README.md").is_file() and (root/"docs/LEXSI_SUBMISSION_DRAFT.md").is_file()
    checks["public_repo_materials"]={"passed":public_materials,"details":"GitHub source and write-up present; external accessibility must still be checked"}
    required=("saved_baseline","trained_adapter","human_review_advisory",
              "paired_finetuned_evaluation","huggingface_artifacts",
              "public_hf_collections","public_repo_materials")
    ready=all(checks[name]["passed"] is True for name in required)
    blocking=[name for name in required if checks[name]["passed"] is not True]
    return {
        "blocking_requirements": blocking,
        "next_evidence_actions": next_evidence_actions(checks, required),
        "status":"READY_FOR_MANUAL_REVIEW" if ready else "NOT_READY",
        "ready_for_submission":ready,"checks":checks,
        "mandatory_checks":list(required),
        "source_linked_evidence_required": True,
        "owner_manifest_claims_are_not_proof": True,
        "gpu_execution_independently_verified": False,
        "human_reviewer_identity_independently_verified": False,
        "semantic_model_improvement_independently_verified": False,
        "official_submission_confirmed": False,
        "caution":(
            "Structural consistency and stored anonymous API receipts are not "
            "signed third-party attestations. Recheck public HF access live, "
            "review license rights and obtain organizer submission confirmation."
        ),
        "deadline":"2026-11-14 end of day, per official organizer website",
    }



# Submission reports are *outputs*, not permissions to rewrite frozen data,
# review votes, source manifests, or previously captured run evidence.
_REPORT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}\.json$")


def write_new_report(text: str, output: str, root=ROOT) -> Path:
    """Create a new direct artifacts/*.json report; never overwrite a file.

    The exclusive file mode protects even if a target appears after the
    existence check. The directory and path checks keep this CLI from
    clobbering research evidence, reviewer CSVs, or release manifests.
    """
    if (not isinstance(output, str) or len(output) > 150
        or "\\" in output or ":" in output
        or any(ord(char) < 32 for char in output)):
        raise ValueError("Report output must be a fresh artifacts/<name>.json")
    parts = output.split("/")
    if (len(parts) != 2 or parts[0] != "artifacts"
        or not _REPORT_NAME.fullmatch(parts[1])):
        raise ValueError("Report output must be a fresh direct artifacts/<name>.json")
    root = Path(root).resolve()
    folder = root / "artifacts"
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise ValueError("Unsafe report output directory")
    folder.mkdir(exist_ok=True)
    if folder.is_symlink():
        raise ValueError("Unsafe report output directory")
    path = folder / parts[1]
    if path.is_symlink() or path.exists():
        raise FileExistsError("Refusing to overwrite an existing gate report")
    # 'x' is an O_EXCL create, so a concurrent process cannot replace a
    # pre-existing report, including an existing symlink.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--strict",action="store_true")
    parser.add_argument("--output")
    opts=parser.parse_args()
    report=evaluate_readiness()
    text=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if opts.output:
        write_new_report(text, opts.output)
    print(text)
    if opts.strict and not report["ready_for_submission"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
