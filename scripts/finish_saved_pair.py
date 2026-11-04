"""Finish a previously interrupted paired AuditKIT comparison with ZERO GPU calls.

DEFAULT mode performs the read-only frozen evaluation preflight only. To
actually write a new paired report, the operator must provide BOTH complete
original prediction reports and explicitly confirm their real provenance.
This cannot prove that the self-declared saved outputs came from a GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.frozen_pair_inference import (
    FINAL_REL, GENERATION, MODEL_ID, SIGNOFF_REL, build_preflight,
)
from scripts.paired_run_recovery import (
    validate_saved_base, validate_saved_tuned, write_evidence_once,
)


def finish_saved_pair(
    root: Path, adapter: Path, base_path: Path,
    tuned_path: Path, paired_path: Path,
) -> dict:
    """Re-audit two complete real-declared saved runs, with no model imports."""
    root, adapter = Path(root).resolve(), Path(adapter).resolve()
    base_path, tuned_path, paired_path = (
        Path(base_path).resolve(), Path(tuned_path).resolve(),
        Path(paired_path).resolve(),
    )
    paths = (base_path, tuned_path, paired_path)
    if len(set(paths)) != 3:
        raise ValueError("Base, tuned and paired paths must all be distinct")
    protected = {
        (root / FINAL_REL).resolve(),
        (root / SIGNOFF_REL).resolve(),
        (adapter / "run_info.json").resolve(),
        (adapter / "adapter" / "adapter_model.safetensors").resolve(),
        (adapter / "adapter" / "adapter_config.json").resolve(),
    }
    if paired_path in protected:
        raise ValueError("Cannot overwrite original review, model or benchmark evidence")
    if paired_path.exists() or paired_path.is_symlink():
        raise FileExistsError("Paired report already exists; refusing to overwrite it")

    preflight = build_preflight(root, adapter)
    if preflight["status"] != "READY_FOR_EXPLICIT_GPU_EXECUTION":
        raise ValueError("Refusing saved comparison; prerequisites missing: "
                         + json.dumps(preflight["checks"]))

    dataset_path = root / FINAL_REL
    rows = read_jsonl(dataset_path)
    digest = preflight["dataset_sha256"]
    revision = preflight["base_revision"]
    adapter_sha = preflight["adapter_sha256"]
    if sha256(dataset_path) != digest or len(rows) != 60:
        raise ValueError("Frozen 60-case benchmark changed after preflight")

    base = validate_saved_base(
        base_path, rows, dataset_sha256=digest,
        revision=revision, generation_config=GENERATION, model_id=MODEL_ID,
    )
    tuned = validate_saved_tuned(
        tuned_path, rows, dataset_sha256=digest,
        revision=revision, generation_config=GENERATION, model_id=MODEL_ID,
        adapter_sha256=adapter_sha,
    )

    from scripts.auditkit_compare import comparison
    paired = comparison(base, tuned, rows, digest)
    if paired.get("status") != "paired_deterministic_audit":
        raise RuntimeError("Actual AuditKIT comparison did not complete")
    if sha256(dataset_path) != digest:
        raise ValueError("Frozen benchmark was modified during AuditKIT replay")
    # The reports remain untouched. A standalone replay is not a fresh GPU run.
    paired.update({
        "generation_config": dict(GENERATION),
        "offline_replayed_from_saved_predictions": True,
        "gpu_executed_in_this_process": False,
        "base_gpu_executed_in_this_process": False,
        "tuned_gpu_executed_in_this_process": False,
        "base_report_sha256": sha256(base_path),
        "tuned_report_sha256": sha256(tuned_path),
        "resumed_existing_base_pass": True,
        "semantics_independently_reviewed": False,
        "saved_gpu_provenance_independently_verified": False,
    })
    write_evidence_once(paired_path, paired)
    return {
        "status": "PAIRED_CPU_AUDITKIT_REPLAY_COMPLETED",
        "cases": len(rows),
        "dataset_sha256": digest,
        "paired_file": str(paired_path),
        "real_model_inference_in_this_process": False,
        "new_optimizer_steps_in_this_process": False,
        "independent_semantic_improvement_verified": False,
        "caution": "Saved raw outputs self-declare prior inference; a CPU replay "
                   "does not independently authenticate GPU logs or model training.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU-only, fail-closed re-audit of completed saved model passes"
    )
    parser.add_argument("--adapter-directory", default="artifacts/align_qlora_smoke")
    parser.add_argument("--base", default="artifacts/final_base_inference.json")
    parser.add_argument("--tuned", default="artifacts/final_tuned_inference.json")
    parser.add_argument("--output", default="artifacts/paired_auditkit.json")
    parser.add_argument("--finalize", action="store_true",
                        help="Explicitly create a new paired AuditKIT report")
    parser.add_argument("--confirm-saved-inference", action="store_true",
                        help="Operator separately inspected original model/GPU run logs")
    args = parser.parse_args()
    adapter = Path(args.adapter_directory)
    if not args.finalize:
        if args.confirm_saved_inference:
            parser.error("--confirm-saved-inference requires --finalize")
        print(json.dumps(build_preflight(ROOT, adapter), indent=2, ensure_ascii=False))
        print("Read-only CPU preflight. No model, GPU, or saved predictions loaded.")
        return
    if not args.confirm_saved_inference:
        parser.error("CPU re-audit requires --finalize --confirm-saved-inference")
    result = finish_saved_pair(ROOT, adapter, Path(args.base),
                               Path(args.tuned), Path(args.output))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
