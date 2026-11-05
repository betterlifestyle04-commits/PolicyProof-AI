"""Guarded one-call recovery, training, and PEFT adapter packaging on Colab T4.

Called IN the already-running notebook kernel with its live AlignTune wrapper:
    from scripts.complete_t4_training import recover_train_save
    archive = recover_train_save(trainer)

This does not reinstall, download a second model, or use test cases.
It runs optimizer steps only after a checked no-AMP SFTConfig preflight.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

from scripts.recover_t4_training import recover_t4
from scripts.check_align_config import check_train_data
from scripts.restore_colab_session import EXPECTED_SHA
from scripts.t4_no_amp_recovery import audit_training_state



def _write_receipt_atomic(path: Path, receipt: dict) -> None:
    """Write a complete local receipt via atomic replace."""
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Unsafe training receipt path")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=".policyproof-receipt-", suffix=".partial", delete=False,
        ) as handle:
            temp = Path(handle.name)
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _publish_adapter_archive(destination: Path, archive: Path, receipt: dict) -> Path:
    """Validate a temporary adapter ZIP before atomic non-overwriting publish."""
    from scripts.verify_training_archive import SINGLE_FILE_LIMIT, ARCHIVE_LIMIT

    destination, archive = Path(destination), Path(archive)
    if archive.is_symlink() or archive.exists():
        raise FileExistsError("Refusing to overwrite existing adapter archive")
    if archive.parent.is_symlink():
        raise ValueError("Unsafe adapter archive parent")
    entries = (
        (destination / "adapter/adapter_model.safetensors",
         "adapter/adapter_model.safetensors"),
        (destination / "adapter/adapter_config.json", "adapter/adapter_config.json"),
    )
    payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if not 0 < len(payload) <= SINGLE_FILE_LIMIT:
        raise RuntimeError("Unsafe training receipt size")
    total = len(payload)
    for file, name in entries:
        if (file.is_symlink() or not file.is_file()
            or not 0 < file.stat().st_size <= SINGLE_FILE_LIMIT):
            raise RuntimeError("Unsafe or oversized PEFT package file: " + name)
        total += file.stat().st_size
    if total > ARCHIVE_LIMIT:
        raise RuntimeError("PEFT package exceeds bounded verifier limit")

    archive.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=archive.parent, delete=False,
            prefix=".policyproof-adapter-", suffix=".partial",
        ) as handle:
            temp = Path(handle.name)
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for file, name in entries:
                z.write(file, name)
            z.writestr("run_info.json", payload)
        with zipfile.ZipFile(temp, "r") as z:
            if (z.namelist() != [name for _, name in entries] + ["run_info.json"]
                or z.testzip() is not None or z.read("run_info.json") != payload):
                raise RuntimeError("Corrupt or unexpected ZIP before publication")
        if temp.stat().st_size > ARCHIVE_LIMIT:
            raise RuntimeError("Compressed PEFT ZIP exceeds verifier limit")
        # Hard link on the SAME filesystem fails if the final name exists.
        # Unlike os.replace, it never overwrites a conflicting archive.
        os.link(temp, archive)
        return archive
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def recover_train_save(wrapper, *, project_dir="/content/PolicyProof-AI",
                       model_revision=None, optimizer_steps: int = 10,
                       loaded_revision_evidence: dict | None = None,
                       archive_path: str | Path | None = None) -> Path:
    """One genuine AlignTune wrapper.train() call; no direct-torch substitution.

    The 10-step smoke and 40-step native experiment are two SEPARATE runs,
    never a continuation or recast of an already-trained adapter.
    This code requires a real Colab GPU; CPU tests stub all training.
    """
    if type(optimizer_steps) is not int or optimizer_steps not in (10, 40):
        raise ValueError("Only 10-step smoke or 40-step native AlignTune runs are allowed")
    if optimizer_steps == 40:
        if (not isinstance(model_revision, str)
            or len(model_revision) != 40
            or any(ch not in "0123456789abcdef" for ch in model_revision)):
            raise ValueError("40-step native run requires the full 40-hex base revision")
        if loaded_revision_evidence is not None:
            if not isinstance(loaded_revision_evidence, dict):
                raise ValueError("Loaded model revision evidence must be an inspection dict")
            observed = loaded_revision_evidence.get("loaded_model_revision_observed")
            verified = loaded_revision_evidence.get("loaded_model_revision_verified")
            if observed is not None and (
                not isinstance(observed, str) or len(observed) != 40
                or observed != model_revision
            ):
                raise ValueError("Loaded model revision contradicts requested revision")
            if verified is True and observed != model_revision:
                raise ValueError("Loaded revision cannot be verified without matching observed SHA")
            if verified not in (True, False, None):
                raise ValueError("Loaded revision verification must be boolean or absent")
    import torch
    from importlib.metadata import version
    from policyproof.core import sha256

    root = Path(project_dir).resolve()
    data = root / "artifacts/verified_curator_train/sft_alpaca.jsonl"
    if not data.exists():
        raise FileNotFoundError("CuratorKIT training-only export missing")
    # Actual file bytes and all language/source/label counts are validated
    # before even loading the GPU safety state, not just a 78-row count.
    audited_data = check_train_data(data)
    if audited_data["dataset_sha256"] != EXPECTED_SHA:
        raise ValueError("Unapproved CuratorKIT training bytes; refusing native training")
    if not torch.cuda.is_available():
        raise RuntimeError("Connect the existing Tesla T4 runtime before training")

    initial = audit_training_state(wrapper)
    if initial["existing_optimizer_steps"] != 0:
        raise RuntimeError("Existing optimizer steps found. Refusing to restart from scratch.")
    if getattr(wrapper, "train_dataset", None) is None or len(wrapper.train_dataset) != 78:
        raise RuntimeError("Expected exactly 78 validated training examples")
    if wrapper.config.model.name_or_path != "CohereLabs/tiny-aya-global":
        raise RuntimeError("Unexpected model; do not train")
    if not initial["base_quantization_4bit"]:
        raise RuntimeError("4-bit frozen base is not configured")
    if initial["trainable_elements"] != 3_833_856:
        raise RuntimeError("Unexpected LoRA trainable size")

    destination = root / (
        "artifacts/align_qlora_native40" if optimizer_steps == 40
        else "artifacts/align_qlora_smoke"
    )
    adapter = destination / "adapter"
    archive = Path(archive_path) if archive_path is not None else Path(
        "/content/policyproof_align_native40.zip" if optimizer_steps == 40
        else "/content/policyproof_align_smoke.zip"
    )
    if (destination.is_symlink()
        or (destination.exists() and (
            not destination.is_dir() or any(destination.iterdir())
        ))
        or archive.is_symlink() or archive.exists()):
        raise FileExistsError(
            "Training output already exists; inspect it rather than overwriting"
        )
    # The AlignTune SFTConfig reads this path when the NEW TRL trainer is
    # constructed. A 40-step run cannot reuse smoke output/checkpoints.
    wrapper.config.logging.output_dir = str(destination)
    destination.mkdir(parents=True, exist_ok=True)
    recovery = recover_t4(wrapper, target_steps=optimizer_steps)
    if not (recovery["trl_amp_fp16"] is False
            and recovery["trl_amp_bf16"] is False
            and recovery["max_steps"] == optimizer_steps
            and recovery["eval_dataset_present"] is False
            and recovery["trl_eval_strategy"] == "no"
            and recovery["trl_load_best_model_at_end"] is False):
        raise RuntimeError(
            "TRL has AMP, wrong step budget, unexpected evaluation, or "
            "load_best_model_at_end enabled — aborting before training"
        )

    # The REAL AlignTune 0.1.28 TRLSFTTrainer.train() uses a separate
    # `self.dataset` cache. If it is None, train() invokes setup_data()
    # again, even when setup_data() was already called and its 78 training
    # rows were checked. Reuse that exact prevalidated object instead.
    # Refuse a different prepopulated training view or an evaluation split.
    if getattr(wrapper, "eval_dataset", None) is not None:
        raise RuntimeError("Unexpected evaluation dataset in training-only run")
    source_dataset = wrapper.train_dataset
    existing_training_view = getattr(wrapper, "dataset", None)
    if existing_training_view is not None and existing_training_view is not source_dataset:
        raise RuntimeError(
            "AlignTune cached dataset differs from the verified 78-row train dataset"
        )
    wrapper.dataset = source_dataset
    if wrapper.dataset is not wrapper.train_dataset or len(wrapper.dataset) != 78:
        raise RuntimeError("Could not bind the original verified training-only dataset")

    metadata = {
        "status": "in_progress",
        "kind": ("40-step native AlignTune QLoRA trial" if optimizer_steps == 40
                 else "10-step QLoRA integration smoke test"),
        "requested_optimizer_steps": optimizer_steps,
        "dataset_sha256": sha256(data),
        "examples": 78,
        "aligntune_cached_train_dataset_reused": True,
        "second_data_manager_load_requested": False,
        "model": "CohereLabs/tiny-aya-global",
        "base_revision": model_revision,
        "loaded_revision_evidence": loaded_revision_evidence,
        "loaded_revision_independently_verified": bool(
            isinstance(loaded_revision_evidence, dict)
            and loaded_revision_evidence.get("loaded_model_revision_verified") is True
            and loaded_revision_evidence.get("loaded_model_revision_observed") == model_revision
        ),
        "precision_recovery": recovery,
        "claims_improved_safety": False,
    }
    note = destination / "run_info.json"
    started = time.monotonic()

    try:
        # One training attempt. Failing here is recorded and not retried.
        training_result = wrapper.train()
        actual_steps = int(wrapper.trainer.state.global_step)
        # Real AlignTune 0.1.28 returns TRL TrainOutput.steps in a dict;
        # the returned count must match trainer state, not just our wish.
        if (not isinstance(training_result, dict)
            or type(training_result.get("steps")) is not int
            or training_result["steps"] != actual_steps):
            raise RuntimeError(
                "AlignTune training-result steps missing or different from trainer state"
            )
        if actual_steps != optimizer_steps:
            raise RuntimeError(
                f"Expected {optimizer_steps} optimizer steps, found {actual_steps}"
            )
        adapter.mkdir(parents=True, exist_ok=True)
        wrapper.save_model(output_dir=str(adapter))
        weights = [adapter / "adapter_model.safetensors"]
        if not (adapter / "adapter_config.json").is_file() or not weights[0].is_file():
            raise RuntimeError(
                "Trainer returned but safe PEFT adapter files are incomplete"
            )
        if any(p.is_symlink() for p in (*weights, adapter / "adapter_config.json")):
            raise RuntimeError("PEFT adapter outputs must be regular files")
        if wrapper.dataset is not source_dataset:
            raise RuntimeError("AlignTune replaced the original checked training dataset")
        metadata.update({
            "status": "completed",
            "optimizer_steps_completed": actual_steps,
            "aligntune_train_result_steps": training_result["steps"],
            "portable_adapter_members": [
                "adapter/adapter_model.safetensors",
                "adapter/adapter_config.json",
                "run_info.json",
            ],
            "artifact_publish_method": "validated_temporary_zip_then_atomic_link",
            "trainer_backend": "aligntune_wrapper",
            "optimizer_steps": actual_steps,
            "training_engine": "AlignTune TRL wrapper.train()",
            "independent_gpu_execution_attested": False,
            "final_benchmark_or_challenge_used_for_training": False,
            "aligntune_trainer_completed": True,
            "seconds": round(time.monotonic() - started, 1),
            "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "gpu": torch.cuda.get_device_name(0),
            "packages": {p: version(p) for p in
                         ("aligntune", "transformers", "trl",
                          "peft", "torch", "bitsandbytes")},
            "adapter_sha256": {p.name: sha256(p) for p in weights},
            "adapter_config_sha256": sha256(adapter / "adapter_config.json"),
            "code_commit": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
        })
        # A local receipt stays at 'packaging' until the final ZIP name is
        # atomically published. The ZIP's contained receipt is 'completed'.
        _write_receipt_atomic(note, {**metadata, "status": "packaging"})
        _publish_adapter_archive(destination, archive, metadata)
        _write_receipt_atomic(note, metadata)
        print(
            f"RECORDED: {optimizer_steps} AlignTune optimizer steps and "
            "adapter checks completed (GPU execution NOT independently attested)",
            flush=True,
        )
        print("Archive path:", archive)
        return archive
    except Exception as exc:
        metadata.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:1200],
            "optimizer_steps_observed": int(
                getattr(getattr(wrapper.trainer, "state", None), "global_step", 0)
            ) if getattr(wrapper, "trainer", None) else None,
            "seconds": round(time.monotonic() - started, 1),
        })
        _write_receipt_atomic(note, metadata)
        print("Training did not complete. Failure saved:", note, flush=True)
        raise
