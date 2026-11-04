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
            valid_audit=(receipt.get("status")=="complete" and receipt.get("sample_count")==48
                         and receipt.get("model_inference_executed") is False)
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
    for weights in (root/"artifacts").rglob("adapter_model.safetensors"):
        try:
            verified=inspect_training_directory(weights.parent.parent)
        except (OSError,ValueError,TypeError,KeyError) as exc:
            adapter_notes.append(type(exc).__name__ + ": " + str(exc)[:130])
            continue
        structurally_checked=True
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

    review=root/"docs/final_benchmark_review_signoff.json"
    human_reviewed=False
    if review.exists():
        try:
            signed=json.loads(review.read_text(encoding="utf-8"))
            human_reviewed=(signed.get("review_complete") is True
                and signed.get("dataset_sha256")==sha256(root/"data/final_holdout_v1.jsonl")
                and len(signed.get("reviewers",[]))>=2)
        except (OSError,ValueError,TypeError):
            pass
    checks["human_review_advisory"]={"passed":human_reviewed,"details":"Recommended bilingual independent gold review, not a formal eligibility rule"}

    paired_path=root/"artifacts/paired_auditkit.json"
    paired=False
    if paired_path.exists():
        try:
            result=json.loads(paired_path.read_text(encoding="utf-8"))
            paired=bool(result.get("status")=="paired_deterministic_audit"
                and result.get("sample_count",0)>=48
                and result.get("adapter_sha256")
                and result.get("case_matched_deterministic_counts"))
        except (OSError,ValueError,TypeError):
            pass
    checks["paired_finetuned_evaluation"]={"passed":paired,"details":"Real base-vs-tuned frozen-set metrics and adapter provenance required"}

    manifest=root/"docs/hf_publication_manifest.json"
    published=False
    if manifest.exists():
        try:
            urls=json.loads(manifest.read_text(encoding="utf-8"))
            names=("dataset_url","adapter_url","logs_url")
            published=all(isinstance(urls.get(n),str)
                and urls[n].startswith("https://huggingface.co/")
                and "PENDING" not in urls[n] for n in names
            ) and urls.get("external_access_verified") is True
        except (OSError,ValueError,TypeError):
            pass
    checks["huggingface_artifacts"]={"passed":bool(published),"details":"Public dataset/adapter/log URLs and verified external access, with article or source write-up"}
    collections_valid=False
    if manifest.exists():
        try:
            info=json.loads(manifest.read_text(encoding="utf-8"))
            collection_fields=("model_collection_url","dataset_collection_url")
            collections_valid=all(
                isinstance(info.get(name),str)
                and info[name].startswith("https://huggingface.co/collections/")
                and "PENDING" not in info[name]
                and "YOUR_" not in info[name]
                for name in collection_fields
            ) and info.get("external_access_verified") is True
            collections_valid=bool(
                collections_valid and info.get("public_items_ungated_verified") is True
            )
        except (OSError, ValueError):
            pass
    checks["public_hf_collections"]={
        "passed":bool(collections_valid),
        "details":"Official Lexsi form requires separate public model/adapter and dataset/log collections; links alone are not proof of non-gated item access"
    }

    public_materials=(root/"README.md").is_file() and (root/"docs/LEXSI_SUBMISSION_DRAFT.md").is_file()
    checks["public_repo_materials"]={"passed":public_materials,"details":"GitHub source and write-up present; external accessibility must still be checked"}
    required=("saved_baseline","trained_adapter","paired_finetuned_evaluation",
              "huggingface_artifacts","public_hf_collections","public_repo_materials")
    ready=all(checks[name]["passed"] for name in required)
    return {
        "status":"READY_FOR_MANUAL_REVIEW" if ready else "NOT_READY",
        "ready_for_submission":ready,"checks":checks,
        "mandatory_checks":list(required),
        "caution":"Local proof cannot verify public HF links, license conditions or submission confirmation.",
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
