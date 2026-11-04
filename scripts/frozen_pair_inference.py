"""Gated, paired real-model evaluation on the 60-case frozen candidate.

DEFAULT = READ-ONLY CPU PREFLIGHT. A separate --execute --confirm-review
--confirm-gpu-run command is required for actual GPU generation. The current
public repository is NOT READY to execute because human review and training
are unfinished. No model is loaded and no prediction is fabricated in preflight.

The runner loads the exact same pinned base-model revision twice:
  1. unchanged 4-bit Tiny Aya Global
  2. the same base revision with a locally validated PEFT adapter
Both use identical prompt and deterministic generation settings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import subprocess

from policyproof.core import ROOT, aggregate, make_prompt, read_jsonl, score, sha256
from scripts.validate_final_holdout import validate_final_holdout
from scripts.verify_training_archive import inspect_training_directory

MODEL_ID = "CohereLabs/tiny-aya-global"
GENERATION = {"do_sample": False, "num_beams": 1, "max_new_tokens": 160}
FINAL_REL = "data/final_holdout_v1.jsonl"
SIGNOFF_REL = "docs/final_benchmark_review_signoff.json"
REVIEWER_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")


def verify_review_signoff(path: Path, dataset_digest: str) -> dict:
    """Check reported signoff metadata, NOT real human identity or competence."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 100_000:
        raise ValueError("Independent review signoff is missing or oversized")
    def unique_fields(pairs):
        record = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("Duplicate signoff field: " + key)
            record[key] = value
        return record

    try:
        record = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=unique_fields,
        )
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid review signoff JSON") from exc
    if not isinstance(record, dict) or record.get("review_complete") is not True:
        raise ValueError("No completed human review declaration")
    if record.get("dataset_file") != FINAL_REL or record.get("dataset_sha256") != dataset_digest:
        raise ValueError("Review signoff is not for the exact final dataset")
    codes = record.get("reviewers")
    if (not isinstance(codes, list) or len(codes) < 2 or
        not all(isinstance(v, str) and REVIEWER_CODE.fullmatch(v) for v in codes) or
        len(codes) != len(set(codes))):
        raise ValueError("Require two or more distinct reviewer pseudonyms")
    return {
        "review_declared": True,
        "reviewers_declared": len(codes),
        "reviewer_identity_independently_verified": False,
        "translation_validity_independently_verified": False,
        "signoff_sha256": sha256(path),
    }


def build_preflight(root: Path = ROOT, adapter_directory: Path | None = None) -> dict:
    """Fail closed, report blockers; never download a model or call inference."""
    root = Path(root).resolve()
    data = root / FINAL_REL
    adapter = (Path(adapter_directory).resolve() if adapter_directory is not None
               else root / "artifacts" / "align_qlora_smoke")
    checks: dict[str, dict] = {}
    digest = None
    revision = None
    adapter_hash = None
    try:
        details = validate_final_holdout(
            source_file=root / "data/final_holdout_sources_v1.json",
            holdout_file=data,
            existing_train=root / "data/train.jsonl",
            prior_dev=root / "data/unseen_policy_test.jsonl",
        )
        if details["cases"] != 60 or details["status"] != "candidate_frozen_for_review":
            raise ValueError("Unexpected final benchmark validation state")
        digest = details["sha256"]
        checks["frozen_dataset"] = {
            "passed": True, "details": "60 structurally validated, SHA256-pinned cases; not blind"
        }
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as exc:
        checks["frozen_dataset"] = {"passed": False, "details": str(exc)[:260]}
    try:
        if digest is None:
            raise ValueError("Dataset must validate before review signoff")
        signoff = verify_review_signoff(root / SIGNOFF_REL, digest)
        checks["independent_review_declared"] = {
            "passed": True,
            "details": f"Signoff metadata lists {signoff['reviewers_declared']} pseudonyms; identity and judgments not independently verified",
        }
    except (ValueError, OSError) as exc:
        checks["independent_review_declared"] = {"passed": False, "details": str(exc)[:260]}
    try:
        info = inspect_training_directory(adapter)
        if not info["track1_submission_evidence_candidate"]:
            raise ValueError("No full-size, pinned, AlignTune-declared PEFT release candidate")
        record = json.loads((adapter / "run_info.json").read_text(encoding="utf-8"))
        revision = record.get("base_revision", record.get("model_revision"))
        if not isinstance(revision, str) or not REVISION.fullmatch(revision):
            raise ValueError("Full base-model revision missing")
        adapter_hash = info["adapter_sha256"]
        checks["adapter_integrity"] = {
            "passed": True,
            "details": "PEFT layout and SHA verified; GPU execution remains externally unverified",
        }
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        checks["adapter_integrity"] = {"passed": False, "details": str(exc)[:260]}
    return {
        "status": "READY_FOR_EXPLICIT_GPU_EXECUTION" if all(
            check["passed"] for check in checks.values()
        ) else "BLOCKED",
        "inference_performed": False,
        "training_performed": False,
        "model_improvement_proven": False,
        "model": MODEL_ID,
        "base_revision": revision,
        "adapter_sha256": adapter_hash,
        "dataset_sha256": digest,
        "case_count": 60 if checks["frozen_dataset"]["passed"] else None,
        "generation_config": dict(GENERATION),
        "checks": checks,
        "limitations": [
            "Review signoff cannot independently establish reviewer identity or competence",
            "Adapter hashes and receipt cannot independently prove GPU optimizer steps",
            "Public synthetic final candidate is not a secret or genuinely blind benchmark",
            "Real GPU execution and human semantic evaluation have not been performed here",
        ],
    }


def _predict_on_gpu(rows: list[dict], revision: str,
                    adapter_directory: Path | None = None) -> tuple[dict[str, str], dict]:
    """Actual GPU-only text generation. No offline simulation or mock backend."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("GPU execution requested, but CUDA is unavailable")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=revision, trust_remote_code=False
    )
    cfg = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=revision, trust_remote_code=False,
        quantization_config=cfg, device_map={"": 0},
        torch_dtype=torch.float16,
    )
    if adapter_directory is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(
            model, str(adapter_directory), is_trainable=False
        )
    model.eval()
    predictions = {}
    try:
        for row in rows:
            prompt = make_prompt(row)
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt",
            )
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
                raise RuntimeError("Unexpected tokenizer chat-template encoding")
            tokens = tokens.to(device)
            with torch.inference_mode():
                output = model.generate(
                    input_ids=tokens, attention_mask=torch.ones_like(tokens),
                    pad_token_id=tokenizer.eos_token_id,
                    **GENERATION,
                )
            # Decode ONLY newly generated tokens, never the prompt.
            raw = tokenizer.decode(
                output[0, tokens.shape[-1]:], skip_special_tokens=True
            )
            if not isinstance(raw, str) or len(raw) > 10000:
                raise RuntimeError("Invalid generated response length")
            predictions[row["id"]] = raw
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return predictions, {
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "transformers_model_backend": "AutoModelForCausalLM",
        "quantization": "bnb_4bit",
    }


def make_report(rows: list[dict], predictions: dict[str, str], *,
                digest: str, revision: str, adapter_sha: str | None,
                gpu_details: dict) -> dict:
    """Describe collected real predictions; only called after GPU runner returns."""
    if set(predictions) != {row["id"] for row in rows} or len(predictions) != len(rows):
        raise ValueError("Incomplete or duplicated paired inference case IDs")
    if any(not isinstance(v, str) for v in predictions.values()):
        raise ValueError("Invalid raw model output")
    scores = [score(row, predictions[row["id"]]) for row in rows]
    tuned = adapter_sha is not None
    return {
        "project": "PolicyProof AI / Lexsi 2026",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "transformers",
        "is_real_model_result": True,
        "is_finetuned": tuned,
        "model": MODEL_ID,
        "model_revision": revision,
        "adapter_sha256": adapter_sha,
        "split": "final_candidate_holdout_v1",
        "dataset_sha256": digest,
        "generation_config": dict(GENERATION),
        "actual_gpu_inference_metadata": gpu_details,
        "human_semantic_review_completed": False,
        "warning": "Predictions generated on a public synthetic candidate; semantic correctness and model improvement not yet established.",
        "metrics": aggregate(scores),
        "scores": scores,
        "predictions": [{"id": row["id"], "raw": predictions[row["id"]]} for row in rows],
    }


def execute_gpu_pair(
    root: Path, adapter: Path, base_output: Path, tuned_output: Path,
    paired_output: Path,
) -> dict:
    preflight = build_preflight(root, adapter)
    if preflight["status"] != "READY_FOR_EXPLICIT_GPU_EXECUTION":
        raise ValueError("Refusing GPU inference: " + json.dumps(preflight["checks"]))
    paths = [base_output.resolve(), tuned_output.resolve(), paired_output.resolve()]
    if len(set(paths)) != 3 or any(path.exists() for path in paths):
        raise FileExistsError("Output paths overlap or already exist; cannot overwrite evidence")
    protected = {
        (Path(root) / FINAL_REL).resolve(), (Path(root) / SIGNOFF_REL).resolve(),
        (adapter / "run_info.json").resolve(),
    }
    if any(path in protected for path in paths):
        raise ValueError("Output path would overwrite input evidence")
    rows = read_jsonl(Path(root) / FINAL_REL)
    digest = preflight["dataset_sha256"]
    revision = preflight["base_revision"]
    base, base_gpu = _predict_on_gpu(rows, revision)
    tuned, tuned_gpu = _predict_on_gpu(rows, revision, adapter_directory=adapter / "adapter")
    base_report = make_report(
        rows, base, digest=digest, revision=revision, adapter_sha=None,
        gpu_details=base_gpu,
    )
    tuned_report = make_report(
        rows, tuned, digest=digest, revision=revision,
        adapter_sha=preflight["adapter_sha256"], gpu_details=tuned_gpu,
    )
    # Do not persist a misleading paired result if either complete inference
    # report or actual AuditKIT precomputed replay fails.
    from scripts.auditkit_compare import comparison
    paired = comparison(base_report, tuned_report, rows, digest)
    paired["generation_config"] = dict(GENERATION)
    paired["gpu_executed_in_this_process"] = True
    paired["semantics_independently_reviewed"] = False
    for path, content in zip(paths, (base_report, tuned_report, paired)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(content, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return {
        "status": "PAIRED_GPU_INFERENCE_AND_AUDITKIT_COMPLETED",
        "cases": len(rows), "dataset_sha256": digest,
        "files": [str(x) for x in paths],
        "training_performed": False,
        "independent_human_semantic_improvement_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed frozen base-vs-adapter evaluation")
    parser.add_argument("--adapter-directory", default="artifacts/align_qlora_smoke")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-review", action="store_true",
                        help="Operator reviewed real human signoff and benchmark freeze")
    parser.add_argument("--confirm-gpu-run", action="store_true",
                        help="Explicitly authorize actual GPU inference")
    parser.add_argument("--base-output", default="artifacts/final_base_inference.json")
    parser.add_argument("--tuned-output", default="artifacts/final_tuned_inference.json")
    parser.add_argument("--paired-output", default="artifacts/paired_auditkit.json")
    args = parser.parse_args()
    adapter = Path(args.adapter_directory).resolve()
    if not args.execute:
        print(json.dumps(build_preflight(ROOT, adapter), indent=2, ensure_ascii=False))
        print("CPU preflight only. No GPU, model loading, or inference.")
        return
    if not args.confirm_review or not args.confirm_gpu_run:
        parser.error("Actual inference requires --execute --confirm-review --confirm-gpu-run")
    result = execute_gpu_pair(
        ROOT, adapter, Path(args.base_output), Path(args.tuned_output),
        Path(args.paired_output),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
