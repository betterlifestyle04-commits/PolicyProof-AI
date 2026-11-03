"""Recover a failed FP16 AMP/QLoRA smoke test with BF16 trainable gradients.

PyTorch's FP16 GradScaler cannot unscale BF16 parameter gradients on many
Turing-class GPUs (including Colab T4). QLoRA normally keeps the quantized
base frozen while training small LoRA adapters, which may safely use FP32
master parameters with FP16 compute.

Only call this after a failed *first* optimizer step, before any successful
steps. It intentionally modifies TRAINABLE parameters only and does not
convert or dequantize the frozen 4-bit base model.
"""
from __future__ import annotations
from collections import Counter


def trainable_dtype_report(model) -> dict:
    import torch
    names = []
    dtype_counts = Counter()
    grad_counts = Counter()
    num_params = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        names.append(name)
        num_params += p.numel()
        dtype_counts[str(p.dtype)] += p.numel()
        if p.grad is not None:
            grad_counts[str(p.grad.dtype)] += p.numel()
    if not names:
        raise ValueError("No trainable parameters found")
    return {
        "trainable_parameter_tensors": len(names),
        "trainable_parameter_elements": num_params,
        "parameter_elements_by_dtype": dict(dtype_counts),
        "gradient_elements_by_dtype": dict(grad_counts),
        "example_trainable_names": names[:6],
        "all_trainable_are_fp32": set(dtype_counts) == {str(torch.float32)},
    }


def repair_trainable_precision(trainer_wrapper) -> dict:
    """Convert only existing LoRA parameters to FP32; reset failed trainer.

    The wrapper must have a model with LoRA parameters loaded and must not
    have completed any optimizer steps. This does not call train().
    """
    import gc
    import torch

    model = getattr(trainer_wrapper, "model", None)
    if model is None:
        raise ValueError("No loaded model on the AlignTune trainer wrapper")

    old_trainer = getattr(trainer_wrapper, "trainer", None)
    if old_trainer is not None:
        completed = int(getattr(getattr(old_trainer, "state", None), "global_step", 0))
        if completed > 0:
            raise ValueError(
                f"{completed} optimizer steps already completed; do not alter "
                "precision mid-training. Save/restart using a fresh trainer."
            )

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("No LoRA parameters are trainable")
    if any(not ("lora_" in name or "modules_to_save" in name) for name, _ in trainable):
        raise ValueError(
            "Unexpected non-LoRA trainable parameters; refusing a blanket "
            "precision change without inspecting architecture"
        )
    if any(p.dtype not in (torch.float16, torch.bfloat16, torch.float32)
           for _, p in trainable):
        raise ValueError("Unsupported trainable dtype for safe FP32 adapter cast")

    before = trainable_dtype_report(model)
    if before["all_trainable_are_fp32"]:
        raise ValueError(
            "All trainable adapter parameters are already FP32; the BF16 "
            "GradScaler error needs a different diagnosis. No changes made."
        )

    # The first attempt's optimizer/scaler must never be reused. Discard the
    # failed HF Trainer but retain the already-loaded quantized PEFT model.
    if old_trainer is not None:
        trainer_wrapper.trainer = None
        del old_trainer
        gc.collect()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for _name, p in trainable:
            if p.dtype != torch.float32:
                p.data = p.data.to(dtype=torch.float32)
            p.grad = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    after = trainable_dtype_report(model)
    if not after["all_trainable_are_fp32"]:
        raise RuntimeError("FP32 trainable-parameter verification failed")
    # No base-model quantization attributes or frozen weights were changed.
    return {
        "repair_applied": True,
        "training_performed": False,
        "base_weights_dequantized": False,
        "before": before,
        "after": after,
        "next_step": "Run a fresh wrapper.train() only after checking the result",
    }
