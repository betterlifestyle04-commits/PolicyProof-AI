"""Restore the verified Tiny Aya/AlignTune 4-bit model after Colab runtime reset.

This setup is intentionally GPU-load-only. It never calls a training loop,
never reads development/final holdout examples and never alters the historical
baseline. Returns the live trainer wrapper for an explicitly separate step.
"""
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile
import json

EXPECTED_SHA = "4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c"
ALLOWLIST = {
    "sft_alpaca.jsonl", "sft_sharegpt.jsonl", "manifest.json",
    "lexsi_provenance.json", "checksums.txt", "README.md",
    "dataset_card.md", "rejected.jsonl",
}


def restore_curated_archive(archive: str | Path, destination: str | Path) -> dict:
    """Verify official CuratorKIT ZIP and extract training-only files."""
    from scripts.verify_curator_archive import check_zip
    from scripts.check_align_config import check_train_data

    archive, destination = Path(archive), Path(destination)
    result = check_zip(archive)
    if result["row_count"] != 78 or result["status"] != "pass":
        raise ValueError("Not the verified 78-example CuratorKIT export")
    destination.mkdir(parents=True, exist_ok=True)
    # Archive members were already checked for traversal and integrity.
    with ZipFile(archive) as z:
        for name in sorted(ALLOWLIST.intersection(z.namelist())):
            (destination / name).write_bytes(z.read(name))
    checked = check_train_data(destination / "sft_alpaca.jsonl")
    if checked["dataset_sha256"] != EXPECTED_SHA:
        raise ValueError("Training data differs from original audited 78-row export")
    return checked


def restore_model_and_data(curated_dir: str | Path, *, expected_gpu="T4"):
    """Load one quantized PEFT model and prepare training data; do not train."""
    import torch
    import bitsandbytes as bnb
    from peft import PeftModel
    from scripts.check_align_config import check_train_data, make_training_wrapper
    from scripts.t4_torch_fallback import preflight_tokenization

    curated_dir = Path(curated_dir)
    audit = check_train_data(curated_dir / "sft_alpaca.jsonl")
    if audit["dataset_sha256"] != EXPECTED_SHA:
        raise ValueError("Unexpected dataset SHA; refusing to load train model")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. Choose a Colab T4 runtime.")
    gpu = torch.cuda.get_device_name(0)
    if expected_gpu is not None and expected_gpu not in gpu:
        raise RuntimeError(f"Expected {expected_gpu} GPU, got {gpu}")
    wrapper = make_training_wrapper(curated_dir)
    if wrapper.model is not None:
        raise RuntimeError("Wrapper already contains a model; refusing second load")
    wrapper.setup_model()  # Gated Tiny Aya weights + 4-bit LoRA
    if not isinstance(wrapper.model, PeftModel):
        raise RuntimeError("Model loaded without expected PEFT adapters")
    count_4bit = sum(isinstance(m, bnb.nn.Linear4bit)
                     for m in wrapper.model.modules())
    trainable = sum(p.numel() for p in wrapper.model.parameters()
                    if p.requires_grad)
    if count_4bit != 252 or trainable != 3_833_856:
        raise RuntimeError(
            f"Unexpected model structure: {count_4bit} 4-bit layers, "
            f"{trainable} trainable parameters"
        )
    wrapper.setup_data()
    if len(wrapper.train_dataset) != 78:
        raise RuntimeError("Training dataset length differs from verified source")
    rows = [json.loads(line) for line in
            (curated_dir / "sft_alpaca.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    tokenization = preflight_tokenization(wrapper.tokenizer, rows)
    if tokenization["checked_rows"] != 78:
        raise RuntimeError("Tokenizer preflight did not cover all 78 rows")
    details = {
        "status": "model_and_dataset_ready",
        "gpu": gpu,
        "4bit_layers": count_4bit,
        "trainable_lora_parameters": trainable,
        "data_sha256": audit["dataset_sha256"],
        "rows": len(rows),
        "tokenization": tokenization,
        "gpu_allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "training_performed": False,
        "adapter_saved": False,
    }
    return wrapper, details


def restore_interactive(root: str | Path = "/content/PolicyProof-AI"):
    """Colab UI helper: upload ZIP if missing, authenticate, load and verify.

    Returns (live_trainer, model_revision_SHA, diagnostic_report). This helper
    NEVER starts training or evaluates any held-out dataset.
    """
    import torch
    from huggingface_hub import model_info, notebook_login
    from scripts.check_align_config import check_train_data

    root = Path(root)
    curated = root / "artifacts" / "verified_curator_train"
    train_file = curated / "sft_alpaca.jsonl"
    if train_file.exists():
        audited = check_train_data(train_file)
        if audited["dataset_sha256"] != EXPECTED_SHA:
            raise ValueError("Existing curated dataset has the wrong SHA256")
        print("CuratorKIT training data already present and verified (78 rows).")
    else:
        from google.colab import files
        print("Upload the original policyproof_curatorkit_train_v1.zip")
        uploaded = files.upload()
        if len(uploaded) != 1:
            raise ValueError("Upload exactly one original CuratorKIT ZIP")
        filename = next(iter(uploaded))
        if not filename.lower().endswith(".zip"):
            raise ValueError("Expected a .zip archive")
        audited = restore_curated_archive(filename, curated)
        print("Archive checked and restored:", audited["row_count"], "rows")

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU; select the Colab T4 accelerator first")

    try:
        revision = model_info("CohereLabs/tiny-aya-global").sha
    except Exception:
        print("Approve the gated Tiny Aya model with Hugging Face if prompted.")
        notebook_login()
        revision = model_info("CohereLabs/tiny-aya-global").sha
    if not isinstance(revision, str) or len(revision) < 12:
        raise RuntimeError("Could not confirm model revision")
    wrapper, report = restore_model_and_data(curated)
    print("MODEL RESTORED:", report["rows"], "rows;",
          report["4bit_layers"], "4-bit layers;",
          report["trainable_lora_parameters"], "LoRA parameters.")
    print("RESTORE COMPLETE — no optimizer steps or GPU training performed.")
    return wrapper, revision, report
