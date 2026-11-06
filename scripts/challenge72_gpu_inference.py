"""Explicit opt-in 72-case Track 2a real GPU baseline inference.

DEFAULT = read-only CPU preflight. This script never trains a model, scores
answers against gold labels, publishes to Hugging Face or claims semantic
accuracy. GPU model loading occurs ONLY with explicit command-line opt-in,
after frozen SHA, strict base Git revision and safe destination checks.

No GPU execution has occurred merely because this file exists or CI passes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.frozen_pair_inference import (
    GENERATION, MODEL_ID, _predict_on_gpu,
)
from scripts.generate_track2a_challenge import validate_challenge
from scripts.paired_run_recovery import write_evidence_once
from scripts.research_report_io import deterministic_report_target
from scripts.score_track2a_72 import (
    KNOWN_CHALLENGE_SHA256, validate_model_report,
)

DEFAULT_DATASET = "artifacts/track2a_challenge_unscored/challenge_v1.jsonl"
DEFAULT_OUTPUT = "artifacts/track2a_72_unchanged_base_predictions.json"
REVISION = re.compile(r"^[a-f0-9]{40}$")
MAX_DATASET_BYTES = 3_000_000


def _frozen_challenge(dataset: Path) -> tuple[list[dict], str]:
    """Validate the exact known 72-case untrained benchmark, without writes."""
    dataset = Path(dataset)
    if (dataset.is_symlink() or not dataset.is_file()
        or not 0 < dataset.stat().st_size <= MAX_DATASET_BYTES):
        raise ValueError("Frozen 72-case benchmark missing, symlinked or oversized")
    digest = sha256(dataset)
    if digest != KNOWN_CHALLENGE_SHA256:
        raise ValueError("Frozen 72-case benchmark SHA-256 differs from approved source")
    rows = read_jsonl(dataset)
    validate_challenge(rows)
    if sha256(dataset) != digest:
        raise ValueError("Benchmark changed during CPU preflight")
    return rows, digest


def preflight(
    root: Path = ROOT, *, dataset: str | Path = DEFAULT_DATASET,
    revision: str | None = None, output: str | Path = DEFAULT_OUTPUT,
) -> dict:
    """Read-only prerequisite report; 'ready' is NOT proof of model access."""
    root = Path(root).resolve()
    input_path = Path(dataset)
    if not input_path.is_absolute():
        input_path = root / input_path
    checks = {}
    digest = None
    cases = None
    try:
        rows, digest = _frozen_challenge(input_path)
        cases = len(rows)
        checks["frozen_challenge"] = {
            "passed": True, "details": "Exact original 72-case SHA256 and schema verified"
        }
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as exc:
        checks["frozen_challenge"] = {"passed": False, "details": str(exc)[:240]}
    if isinstance(revision, str) and REVISION.fullmatch(revision):
        checks["pinned_base_revision"] = {
            "passed": True, "details": "Full requested Git commit SHA provided; loaded revision not yet observed"
        }
    else:
        checks["pinned_base_revision"] = {
            "passed": False, "details": "Supply --revision with an actual full 40-character model commit SHA"
        }
    try:
        target = deterministic_report_target(output, root=root)
        if target.is_symlink() or target.exists():
            raise FileExistsError("Existing prediction report is immutable; choose a NEW filename")
        if target.parent.is_symlink() or (
            target.parent.exists() and not target.parent.is_dir()
        ):
            raise ValueError("Unsafe output artifacts directory")
        checks["new_prediction_output"] = {
            "passed": True, "details": "New direct artifacts JSON path; will use exclusive creation"
        }
    except (ValueError, OSError, FileExistsError) as exc:
        checks["new_prediction_output"] = {"passed": False, "details": str(exc)[:240]}
    return {
        "status": "READY_FOR_EXPLICIT_GPU_EXECUTION" if all(
            item["passed"] is True for item in checks.values()
        ) else "BLOCKED",
        "model": MODEL_ID,
        "requested_revision": revision,
        "dataset_sha256": digest,
        "case_count": cases,
        "generation_config": dict(GENERATION),
        "checks": checks,
        "gpu_execution_performed": False,
        "training_performed": False,
        "human_semantic_review_performed": False,
        "warning": (
            "Only source preflight; a READY status does not verify GPU access, "
            "licensed model access, future model loading, or evidence quality"
        ),
    }


def execute_gpu(
    root: Path, dataset: str | Path, revision: str, output: str | Path,
) -> dict:
    """Run 72 genuine unchanged-base generations only after full opt-in in CLI.

    Isolated synthetic tests monkeypatch the GPU method. Such tests are NOT
    real model outputs. Actual execution requires licensed CUDA + HF access.
    """
    root = Path(root).resolve()
    state = preflight(root, dataset=dataset, revision=revision, output=output)
    if state["status"] != "READY_FOR_EXPLICIT_GPU_EXECUTION":
        raise ValueError("GPU inference blocked by CPU prerequisites: " + json.dumps(state["checks"]))
    source = Path(dataset)
    if not source.is_absolute():
        source = root / source
    target = deterministic_report_target(output, root=root)
    rows, digest = _frozen_challenge(source)
    if digest != state["dataset_sha256"]:
        raise ValueError("Frozen challenge changed since validated preflight")

    # The existing final-60 GPU loader checks the ACTUALLY loaded model config
    # commit SHA before generation, forces untuned base and no sampling, and
    # returns raw output text plus original runtime metadata.
    predictions, gpu_metadata = _predict_on_gpu(rows, revision)
    if not isinstance(gpu_metadata, dict) or (
        gpu_metadata.get("loaded_model_revision_observed") != revision
        or gpu_metadata.get("loaded_model_revision_verified") is not True
    ):
        raise RuntimeError("GPU runner returned no matching observed loaded-model commit")
    if sha256(source) != digest:
        raise ValueError("Frozen benchmark changed during GPU inference")
    if not isinstance(predictions, dict) or len(predictions) != len(rows):
        raise ValueError("Incomplete original GPU prediction coverage")
    report = {
        "project": "PolicyProof AI / Lexsi Track 2a",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "transformers",
        "is_real_model_result": True,
        "is_finetuned": False,
        "model": MODEL_ID,
        "model_revision": revision,
        "dataset_sha256": digest,
        "generation_config": dict(GENERATION),
        "actual_gpu_inference_metadata": gpu_metadata,
        "model_inference_executed_in_this_process": True,
        "human_semantic_review_completed": False,
        "model_improvement_verified": False,
        "predictions": [
            {"id": row["id"], "raw": predictions[row["id"]]}
            for row in rows
        ],
        "limitations": [
            "This is unchanged-base inference, NOT native AlignTune fine-tuning",
            "Recorded runtime metadata is local evidence, NOT independent GPU attestation",
            "Exact-match grading and model-answer semantics have NOT been reviewed",
            "Attack success and real-world model safety are not established",
        ],
    }
    # Verify the entire report before writing expensive original outputs.
    # This validation checks only self-declared metadata and case coverage.
    validate_model_report(report, rows, digest)
    if sha256(source) != digest:
        raise ValueError("Frozen benchmark changed before immutable publication")
    write_evidence_once(target, report)
    return {
        "status": "CHALLENGE72_GPU_INFERENCE_SAVED",
        "saved_report": str(target),
        "samples": len(rows),
        "dataset_sha256": digest,
        "requested_revision": revision,
        "observed_loaded_revision": gpu_metadata["loaded_model_revision_observed"],
        "base_model_inference_performed_in_this_process": True,
        "training_performed": False,
        "independent_gpu_attestation": False,
        "human_semantic_review_completed": False,
        "official_submission_completed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Guarded 72-case unchanged Tiny Aya GPU inference; read-only by default"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=None,
                        help="Actual 40-hex pinned commit SHA for CohereLabs/tiny-aya-global")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-gpu-run", action="store_true")
    parser.add_argument("--acknowledge-gated-model-license", action="store_true",
                        help="Confirm separately that Hugging Face account access and model license permit this run")
    args = parser.parse_args()
    if not args.execute:
        if args.confirm_gpu_run or args.acknowledge_gated_model_license:
            parser.error("Execution confirmation flags require --execute")
        print(json.dumps(preflight(
            ROOT, dataset=args.dataset, revision=args.revision,
            output=args.output,
        ), indent=2, ensure_ascii=False))
        print("Read-only CPU preflight. No model loaded, no GPU inference, no training.")
        return
    if not (args.confirm_gpu_run and args.acknowledge_gated_model_license):
        parser.error(
            "Actual GPU execution requires --execute --confirm-gpu-run "
            "--acknowledge-gated-model-license"
        )
    result = execute_gpu(ROOT, args.dataset, args.revision, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
