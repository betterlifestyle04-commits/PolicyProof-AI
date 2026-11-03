"""Guarded one-call recovery, training, and PEFT adapter packaging on Colab T4.

Called IN the already-running notebook kernel with its live AlignTune wrapper:
    from scripts.complete_t4_training import recover_train_save
    archive = recover_train_save(trainer)

This does not reinstall, download a second model, or use test cases.
It runs optimizer steps only after a checked no-AMP SFTConfig preflight.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import time
import zipfile

from scripts.recover_t4_training import recover_t4
from scripts.t4_no_amp_recovery import audit_training_state


def recover_train_save(wrapper, *, project_dir="/content/PolicyProof-AI",
                       model_revision=None) -> Path:
    import torch
    from importlib.metadata import version
    from policyproof.core import sha256

    root = Path(project_dir).resolve()
    data = root / "artifacts/verified_curator_train/sft_alpaca.jsonl"
    if not data.exists():
        raise FileNotFoundError("CuratorKIT training-only export missing")
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

    destination = root / "artifacts/align_qlora_smoke"
    adapter = destination / "adapter"
    if (destination / "run_info.json").exists() or (adapter / "adapter_config.json").exists():
        raise FileExistsError(
            "Training output already exists; inspect it rather than overwriting"
        )
    destination.mkdir(parents=True, exist_ok=True)
    recovery = recover_t4(wrapper)
    if not (recovery["trl_amp_fp16"] is False
            and recovery["trl_amp_bf16"] is False
            and recovery["max_steps"] == 10):
        raise RuntimeError("TRL still has AMP enabled — aborting")

    metadata = {
        "status": "in_progress",
        "kind": "10-step QLoRA integration smoke test",
        "dataset_sha256": sha256(data),
        "examples": 78,
        "model": "CohereLabs/tiny-aya-global",
        "base_revision": model_revision,
        "precision_recovery": recovery,
        "claims_improved_safety": False,
    }
    note = destination / "run_info.json"
    started = time.monotonic()

    try:
        # One training attempt. Failing here is recorded and not retried.
        wrapper.train()
        actual_steps = int(wrapper.trainer.state.global_step)
        if actual_steps != 10:
            raise RuntimeError(f"Expected 10 optimizer steps, found {actual_steps}")
        adapter.mkdir(parents=True, exist_ok=True)
        wrapper.save_model(output_dir=str(adapter))
        weights = [p for p in adapter.iterdir()
                   if p.name in ("adapter_model.safetensors", "adapter_model.bin")]
        if not (adapter / "adapter_config.json").is_file() or not weights:
            raise RuntimeError("Trainer returned but PEFT adapter files are incomplete")
        metadata.update({
            "status": "completed",
            "optimizer_steps_completed": actual_steps,
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
        note.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        archive = Path("/content/policyproof_align_smoke.zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(destination.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(destination))
        print("COMPLETED: 10 optimizer steps and adapter verified", flush=True)
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
        note.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print("Training did not complete. Failure saved:", note, flush=True)
        raise
