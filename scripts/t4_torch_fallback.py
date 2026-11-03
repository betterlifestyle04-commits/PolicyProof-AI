"""T4 rescue: train ALREADY-LOADED Tiny Aya 4-bit PEFT model using plain PyTorch.

This deliberately bypasses the AlignTune/TR L trainer creation path that
resets fp16 configuration and casts QLoRA adapters to BF16 in TRL v1.7.1.
It uses the AlignTune-loaded base/LoRA model and CuratorKIT train data.
An adapter produced here is a PyTorch fallback result, NOT proof of a
successful AlignTune training run or improved safety/performance.
"""
from __future__ import annotations
import gc
import hashlib
import json
import random
import subprocess
import time
import zipfile
from pathlib import Path

TRAIN_SHA = "4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c"


def encode_pair(tokenizer, row, max_length=1024):
    """Mask user/policy tokens; loss only on the JSON assistant completion."""
    question = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True
    )
    if isinstance(prefix, dict):
        prefix = prefix["input_ids"]
    output_ids = tokenizer.encode(row["output"], add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("No tokenizer EOS token")
    target = output_ids + [tokenizer.eos_token_id]
    if len(target) >= max_length:
        raise ValueError("Target exceeds sequence limit")
    prefix = prefix[-(max_length - len(target)):]
    ids = prefix + target
    labels = [-100] * len(prefix) + target
    assert len(ids) == len(labels) and len(target) > 1
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def train_existing_t4(trainer, *, root="/content/PolicyProof-AI", model_sha=None):
    import torch
    import bitsandbytes as bnb
    from peft import PeftModel
    from importlib.metadata import version
    from scripts.check_align_config import check_train_data

    root = Path(root)
    source = root / "artifacts/verified_curator_train/sft_alpaca.jsonl"
    checked = check_train_data(source)
    if checked["dataset_sha256"] != TRAIN_SHA:
        raise RuntimeError("CuratorKIT training hash differs from verified archive")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Current Colab runtime must have a Tesla T4 GPU")
    model = getattr(trainer, "model", None)
    tokenizer = getattr(trainer, "tokenizer", None)
    if not isinstance(model, PeftModel) or tokenizer is None:
        raise RuntimeError("Existing PEFT model and tokenizer are required")
    if trainer.config.model.name_or_path != "CohereLabs/tiny-aya-global":
        raise RuntimeError("Unexpected loaded model")
    if getattr(trainer, "train_dataset", None) is None or len(trainer.train_dataset) != 78:
        raise RuntimeError("AlignTune's 78 training examples must already be prepared")
    old_trainer = getattr(trainer, "trainer", None)
    global_step = int(getattr(getattr(old_trainer, "state", None), "global_step", 0)) if old_trainer else 0
    if global_step:
        raise RuntimeError("Training already progressed; cannot restart in place")
    if sum(isinstance(m, bnb.nn.Linear4bit) for m in model.modules()) < 200:
        raise RuntimeError("Expected 4-bit quantized Tiny Aya model")
    weights = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if (len(weights) != 288 or sum(p.numel() for _, p in weights) != 3_833_856
            or any("lora_" not in n for n, _ in weights)):
        raise RuntimeError("Unexpected trainable adapter configuration")
    out = root / "artifacts/torch_fallback_qlora_smoke"
    receipt = out / "run_info.json"
    if receipt.exists() or (out / "adapter/adapter_config.json").exists():
        raise FileExistsError("Previous run exists; inspect before another attempt")
    out.mkdir(parents=True, exist_ok=True)
    frozen_before = [(id(p), p.dtype) for p in model.parameters() if not p.requires_grad]
    if old_trainer is not None:
        trainer.trainer = None
        del old_trainer
        gc.collect()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for _, p in weights:
            if p.dtype != torch.float32:
                p.data = p.data.float()
            p.grad = None
    if any(p.dtype != torch.float32 for _, p in weights):
        raise RuntimeError("Trainable adapters could not be converted to FP32")
    if frozen_before != [(id(p), p.dtype) for p in model.parameters() if not p.requires_grad]:
        raise RuntimeError("Frozen model parameters changed unexpectedly")
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    shuffled = list(range(len(rows)))
    random.Random(42).shuffle(shuffled)
    optimizer = torch.optim.AdamW([p for _, p in weights], lr=2e-4)
    model.train()
    finished = 0
    losses = []
    started = time.monotonic()
    try:
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            micro_losses = []
            for micro in range(4):
                row = rows[shuffled[(step * 4 + micro) % len(shuffled)]]
                sample = encode_pair(tokenizer, row)
                batch = {k: torch.tensor([v], device="cuda", dtype=torch.long)
                         for k, v in sample.items()}
                # Autocast to fp16 does NOT use GradScaler. T4 supports fp16;
                # PEFT FP32 adapter master parameters remain FP32.
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(**batch).loss.float() / 4
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                micro_losses.append(float(loss.detach().item()) * 4)
            if any(p.dtype != torch.float32 for _, p in weights):
                raise RuntimeError("LoRA parameter dtype changed during the step")
            norm = torch.nn.utils.clip_grad_norm_([p for _, p in weights], 1.0)
            if not bool(torch.isfinite(norm).item()):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            finished += 1
            current_loss = sum(micro_losses) / len(micro_losses)
            losses.append({"step": finished, "loss": round(current_loss, 6)})
            print(f"Step {finished}/10, training loss {current_loss:.5f}", flush=True)

        if finished != 10:
            raise RuntimeError("Not all optimizer steps completed")
        adapter = out / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter), safe_serialization=True)
        tokenizer.save_pretrained(str(adapter))
        saved = sorted(p for p in adapter.glob("adapter_model.*") if p.is_file())
        if not saved or not (adapter / "adapter_config.json").is_file():
            raise RuntimeError("PEFT adapter weights or config missing")
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        data = {
            "status": "completed",
            "trainer_backend": "manual_torch_smoke_fallback",
            "aligntune_trainer_completed": False,
            "model": "CohereLabs/tiny-aya-global",
            "model_revision": model_sha,
            "examples": 78, "dataset_sha256": checked["dataset_sha256"],
            "optimizer_steps": finished, "gradient_accumulation": 4,
            "learning_rate": 2e-4, "max_length": 1024,
            "losses": losses,
            "duration_s": round(time.monotonic() - started, 1),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "adapter_sha256": {p.name: sha(p) for p in saved},
            "code_revision": subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "versions": {p: version(p) for p in ("torch", "peft", "bitsandbytes")},
            "performance_improvement_verified": False,
        }
        receipt.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        archive = Path("/content/policyproof_t4_torch_fallback.zip")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out))
        print("TRAINING FINISHED: adapter saved; no evaluation claim", flush=True)
        return archive
    except Exception as exc:
        receipt.write_text(json.dumps({
            "status": "failed", "trainer_backend": "manual_torch_smoke_fallback",
            "completed_steps": finished, "error": str(exc)[:1200],
            "error_type": type(exc).__name__,
        }, indent=2) + "\n", encoding="utf-8")
        print("Fallback training failed; diagnostic receipt written. Do not repeat blindly.", flush=True)
        raise
