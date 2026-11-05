"""Restore the verified Tiny Aya/AlignTune 4-bit model after Colab runtime reset.

This setup is intentionally GPU-load-only. It never calls a training loop,
never reads development/final holdout examples and never alters the historical
baseline. Returns the live trainer wrapper for an explicitly separate step.
"""
from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile
from tempfile import TemporaryDirectory
import json
import re
import shutil

EXPECTED_SHA = "4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c"
ALLOWLIST = {
    "sft_alpaca.jsonl", "sft_sharegpt.jsonl", "manifest.json",
    "lexsi_provenance.json", "checksums.txt", "README.md",
    "dataset_card.md", "rejected.jsonl",
}


def restore_curated_archive(archive: str | Path, destination: str | Path) -> dict:
    """Verify training data *before* touching any existing working-tree files.

    All archive members are checked first, then allowlisted files are staged
    outside destination. Exact 78-row curated training data / SHA must pass
    before any destination file is created. Matching files are left untouched;
    conflicting existing files fail closed. Exclusive writes and best-effort
    cleanup protect original evidence from an interrupted import.
    """
    from scripts.verify_curator_archive import check_zip
    from scripts.check_align_config import check_train_data

    archive, destination = Path(archive), Path(destination)
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("CuratorKIT ZIP missing or symlinked")
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise ValueError("Curated destination is not a safe directory")
    result = check_zip(archive)
    if result["row_count"] != 78 or result["status"] != "pass":
        raise ValueError("Not the verified 78-example CuratorKIT export")

    with TemporaryDirectory(prefix="policyproof-curated-verify-") as temp:
        staged = Path(temp)
        with ZipFile(archive) as z:
            names = sorted(ALLOWLIST.intersection(z.namelist()))
            if "sft_alpaca.jsonl" not in names:
                raise ValueError("CuratorKIT training export missing")
            # Each allowlisted member is a single basename, never a path.
            for name in names:
                if Path(name).name != name:
                    raise ValueError("Unexpected archive member path")
                with z.open(name) as source, (staged / name).open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        checked = check_train_data(staged / "sft_alpaca.jsonl")
        if checked["dataset_sha256"] != EXPECTED_SHA:
            raise ValueError("Training data differs from original audited 78-row export")

        # Preflight *every* target before writing anything. Existing files
        # must be byte-for-byte identical and must not be symlinks.
        for name in names:
            target = destination / name
            if target.is_symlink():
                raise ValueError("Existing curated artifact is a symlink: " + name)
            if target.exists() and (
                not target.is_file() or target.read_bytes() != (staged / name).read_bytes()
            ):
                raise FileExistsError("Existing curated artifact differs; refusing overwrite: " + name)

        destination.mkdir(parents=True, exist_ok=True)
        created = []
        try:
            for name in names:
                target = destination / name
                if target.exists():
                    continue
                with (staged / name).open("rb") as source, target.open("xb") as output:
                    created.append(target)
                    shutil.copyfileobj(source, output, length=1024 * 1024)
            # Race/partial-write safety: verify destination training bytes
            # again. An existing file must never be silently replaced.
            final = check_train_data(destination / "sft_alpaca.jsonl")
            if final["dataset_sha256"] != EXPECTED_SHA:
                raise ValueError("Curated destination changed during restore")
        except BaseException:
            for artifact in reversed(created):
                artifact.unlink(missing_ok=True)
            raise
    return {
        **checked,
        "restore_verified_before_writes": True,
        "existing_files_overwritten": False,
        "native_training_executed": False,
    }

def restore_model_and_data(curated_dir: str | Path, *,
                           expected_gpu="T4", expected_revision: str | None = None):
    """Load one quantized PEFT model and prepare training data; do not train."""
    import torch
    import bitsandbytes as bnb
    from peft import PeftModel
    from scripts.check_align_config import check_train_data, make_training_wrapper
    from scripts.t4_torch_fallback import preflight_tokenization

    # Validate the requested revision before using the Colab GPU or
    # downloading multi-GB weights; the loaded revision is checked separately.
    if expected_revision is not None and (
        not isinstance(expected_revision, str)
        or not re.fullmatch(r"[a-f0-9]{40}", expected_revision)
    ):
        raise ValueError("Require complete pinned 40-hex requested base revision")
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
    # The Hub API's current SHA is not evidence of loaded weight identity.
    # Inspect actual loaded config._commit_hash where available; refuse any
    # conflict and mark a missing hash as UNVERIFIED rather than assumed.
    if expected_revision is not None:
        from scripts.t4_torch_fallback import check_loaded_revision
        observed_revision = check_loaded_revision(wrapper.model, expected_revision)
    else:
        observed_revision = {
            "hub_revision_claim": None,
            "loaded_model_revision_observed": None,
            "loaded_model_revision_verified": False,
            "validation_method": "not_requested",
        }
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
        "loaded_revision_inspection": observed_revision,
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
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise RuntimeError("Could not confirm model revision")
    wrapper, report = restore_model_and_data(curated, expected_revision=revision)
    print("MODEL RESTORED:", report["rows"], "rows;",
          report["4bit_layers"], "4-bit layers;",
          report["trainable_lora_parameters"], "LoRA parameters.")
    observed = report["loaded_revision_inspection"]
    if not observed["loaded_model_revision_verified"]:
        print(
            "WARNING: Requested HF revision is known but loaded weight revision "
            "was not independently visible. Do NOT claim pinned loaded weights."
        )
    print("RESTORE COMPLETE — no optimizer steps or GPU training performed.")
    return wrapper, revision, report
