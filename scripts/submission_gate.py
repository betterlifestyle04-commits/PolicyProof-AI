"""Read-only, fail-closed Lexsi submission readiness validator.

Run python -m scripts.submission_gate (or add --strict to fail incomplete).
No network calls or inference. Existence and metadata checks cannot replace
external manual verification of model quality, license or URL access.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json

from policyproof.core import ROOT, sha256
from scripts.verify_evidence import verify_baseline
from scripts.verify_training_archive import inspect_training_directory
from scripts.submission_evidence import (
    paired_evaluation_from_sources, public_hf_evidence, source_linked_review,
)


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
    if audit.exists():
        try:
            receipt=json.loads(audit.read_text(encoding="utf-8"))
            baseline_metadata=json.loads(baseline.read_text(encoding="utf-8")) if valid_baseline else {}
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
        except (ValueError,OSError):
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
    ready=all(checks[name]["passed"] for name in required)
    return {
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--strict",action="store_true")
    parser.add_argument("--output")
    opts=parser.parse_args()
    report=evaluate_readiness()
    text=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if opts.output:
        path=Path(opts.output)
        path.parent.mkdir(exist_ok=True,parents=True)
        path.write_text(text,encoding="utf-8")
    print(text)
    if opts.strict and not report["ready_for_submission"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
