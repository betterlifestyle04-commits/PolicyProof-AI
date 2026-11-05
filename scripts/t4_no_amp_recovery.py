"""T4 mixed-precision recovery for AlignTune + TRL after BF16 GradScaler failure.

The failure "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
BFloat16" can persist even after casting LoRA trainable weights to FP32,
because a new HF Trainer is constructed with FP16 automatic mixed precision
and GradScaler. This module prepares *unscaled training* by setting the
AlignTune precision policy to FP32; 4-bit frozen base quantization and NF4
compute dtype are left untouched. It does NOT call train() or load weights.

Works only before ANY optimizer step. A completed partial run must be
checkpointed and restarted, not modified in-place.
"""
from __future__ import annotations
import gc
from pathlib import Path
from collections import Counter


def synchronize_factory_amp_flags(train_config) -> dict:
    """Force AlignTune's late-applied factory kwargs to respect no-AMP policy.

    In v0.1.28 train.extra_params is merged into TRL SFTConfig *after*
    normal model/train precision resolution. Stale fp16=True there restores
    the GradScaler even when model.precision=FP32.
    """
    extras = getattr(train_config, "extra_params", None)
    if not isinstance(extras, dict):
        raise ValueError("Expected AlignTune train.extra_params to be a dict")
    previous = {"fp16": extras.get("fp16"), "bf16": extras.get("bf16")}
    train_config.fp16 = False
    train_config.bf16 = False
    extras["fp16"] = False
    extras["bf16"] = False
    if "precision" in extras:
        extras["precision"] = "fp32"
    if (train_config.fp16 is not False or train_config.bf16 is not False
            or extras["fp16"] is not False or extras["bf16"] is not False):
        raise RuntimeError("Unable to synchronize AlignTune factory AMP flags")
    return previous


def audit_training_state(wrapper) -> dict:
    """Read-only state inspection. No GPU operations."""
    import torch
    model = getattr(wrapper, "model", None)
    if model is None:
        raise ValueError("Model not loaded")
    tensors = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not tensors:
        raise ValueError("No trainable model parameters")
    old = getattr(wrapper, "trainer", None)
    steps = int(getattr(getattr(old, "state", None), "global_step", 0)) if old else 0
    param_dtypes = Counter()
    grad_dtypes = Counter()
    for _name, p in tensors:
        param_dtypes[str(p.dtype)] += p.numel()
        if p.grad is not None:
            grad_dtypes[str(p.grad.dtype)] += p.numel()
    return {
        "existing_optimizer_steps": steps,
        "trainable_tensor_count": len(tensors),
        "trainable_elements": sum(p.numel() for _, p in tensors),
        "trainable_element_dtypes": dict(param_dtypes),
        "gradient_element_dtypes": dict(grad_dtypes),
        "precision": str(wrapper.config.model.precision),
        "training_fp16": bool(wrapper.config.train.fp16),
        "training_bf16": bool(wrapper.config.train.bf16),
        "base_quantization_4bit": bool(wrapper.config.model.quantization.get("load_in_4bit")),
        "base_4bit_compute_dtype": wrapper.config.model.quantization.get("bnb_4bit_compute_dtype"),
    }


def prepare_t4_no_scaler(wrapper, *, target_steps: int = 10) -> dict:
    """Disable FP16 AMP/GradScaler; retain frozen 4-bit base and FP32 LoRA.

    Fails closed if training was partially completed, LoRA weights are not
    FP32, or unsupported trainable parameters appear. Does not train.
    """
    import torch
    from aligntune.core.sft.config import PrecisionType
    from aligntune.core.sft.training_args_builder import build_sft_config

    # Only the two explicitly documented experiments may reach a TRL trainer.
    # Prevent accidental 0/-1/unbounded training or bool-as-int inputs.
    if type(target_steps) is not int or target_steps not in (10, 40):
        raise ValueError("Native training must request exactly 10 or 40 optimizer steps")
    train_config = wrapper.config.train
    original_steps = getattr(train_config, "max_steps", None)
    if (type(original_steps) is not int
        or original_steps not in (10, target_steps)):
        raise ValueError("Unexpected pre-existing AlignTune optimizer budget")
    extras = getattr(train_config, "extra_params", None)
    if not isinstance(extras, dict):
        raise ValueError("AlignTune train.extra_params must be a dict")
    existing_override = extras.get("max_steps")
    if (existing_override is not None
        and (type(existing_override) is not int
             or existing_override not in (10, target_steps))):
        raise ValueError("Unexpected late TRL optimizer-budget override")

    before = audit_training_state(wrapper)
    if before["existing_optimizer_steps"] != 0:
        raise ValueError(
            f"{before['existing_optimizer_steps']} optimizer steps completed: "
            "cannot change precision mid-run"
        )
    if not before["base_quantization_4bit"]:
        raise ValueError("This recovery is only for the already-quantized QLoRA model")
    if before["trainable_element_dtypes"] != {"torch.float32": before["trainable_elements"]}:
        raise ValueError("Expected repaired FP32 adapters; inspect parameter dtypes first")
    model = wrapper.model
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if any("lora_" not in name and "modules_to_save" not in name for name in trainable_names):
        raise ValueError("Unexpected trainable weights beyond LoRA")
    frozen_before = {id(p): (p.dtype, p.requires_grad)
                     for p in model.parameters() if not p.requires_grad}
    if getattr(wrapper, "train_dataset", None) is None:
        raise ValueError("Training data not set up; run setup_data()")
    if len(wrapper.train_dataset) != 78:
        raise ValueError("Expected the 78 validated training-only samples")

    # AlignTune merges train.extra_params AFTER other training options,
    # including max_steps. Synchronize both places before building SFTConfig.
    # This is configuration only; no optimizer or model-loading action occurs.
    train_config.max_steps = target_steps
    extras["max_steps"] = target_steps

    # The original 78 training rows are not an evaluation set. In real
    # AlignTune 0.1.28, SFT TrainingConfig defaults
    # load_best_model_at_end=True, while the SFT builder resolves
    # eval_strategy='no' when eval_dataset is None. Its late config extractor
    # can then re-enable load_best_model_at_end. Disable both sources before
    # constructing TRL SFTConfig, or abort if it reappears afterward.
    no_eval_dataset = getattr(wrapper, "eval_dataset", None) is None
    previous_best_model = getattr(train_config, "load_best_model_at_end", None)
    if no_eval_dataset:
        train_config.load_best_model_at_end = False
        extras["load_best_model_at_end"] = False

    wrapper.config.model.precision = PrecisionType.FP32
    old_factory_amp = synchronize_factory_amp_flags(train_config)
    old_extras = train_config.extra_params

    # Verify SFTConfig actually omits FP16/BF16 AMP. This constructs arguments
    # only; it does not instantiate SFTTrainer or start gradient training.
    config = build_sft_config(
        config=wrapper.config,
        train_dataset=wrapper.train_dataset,
        eval_dataset=getattr(wrapper, "eval_dataset", None),
        output_dir=str(wrapper.config.logging.output_dir),
    )
    if config.fp16 or config.bf16:
        raise RuntimeError(
            "TRL would still enable mixed precision after synchronizing "
            "train.extra_params: fp16=" + str(config.fp16) +
            ", bf16=" + str(config.bf16) + "; abort recovery"
        )
    if type(config.max_steps) is not int or config.max_steps != target_steps:
        raise RuntimeError(
            f"TRL optimizer budget mismatch: wanted {target_steps}, "
            f"configured {config.max_steps!r}. Abort before training."
        )
    if Path(config.output_dir).resolve() != Path(wrapper.config.logging.output_dir).resolve():
        raise RuntimeError("TRL would train in an unexpected output directory")
    if no_eval_dataset:
        strategy = getattr(config, "eval_strategy", None)
        strategy = getattr(strategy, "value", strategy)
        if strategy != "no" or getattr(config, "load_best_model_at_end", None) is not False:
            raise RuntimeError(
                "TRL no-eval SFTConfig must disable evaluation and "
                "load_best_model_at_end before any optimizer step"
            )

    # Do not reuse the old HF Trainer/scaler/optimizer after a failed step.
    old_trainer = getattr(wrapper, "trainer", None)
    if old_trainer is not None:
        wrapper.trainer = None
        del old_trainer
        gc.collect()
    model.zero_grad(set_to_none=True)
    assert all(
        (p.dtype, p.requires_grad) == frozen_before[id(p)]
        for p in model.parameters() if id(p) in frozen_before
    ), "Unexpected frozen weight modification"

    after = audit_training_state(wrapper)
    if after["gradient_element_dtypes"]:
        raise RuntimeError("Failed to clear previous gradients")
    return {
        "status": "ready_for_no_amp_retry",
        "training_performed": False,
        "model_reloaded": False,
        "frozen_base_converted": False,
        "trl_config_fp16": bool(config.fp16),
        "trl_config_bf16": bool(config.bf16),
        "trl_max_steps": int(config.max_steps),
        "configured_target_optimizer_steps": target_steps,
        "original_max_steps": original_steps,
        "original_factory_max_steps_override": existing_override,
        "trl_output_dir": str(config.output_dir),
        "trl_gradient_accumulation": int(config.gradient_accumulation_steps),
        "eval_dataset_present": not no_eval_dataset,
        "original_load_best_model_at_end": previous_best_model,
        "trl_eval_strategy": (
            getattr(getattr(config, "eval_strategy", None), "value",
                    getattr(config, "eval_strategy", None))
        ),
        "trl_load_best_model_at_end": getattr(config, "load_best_model_at_end", None),
        "factory_extra_params_fp16": old_extras.get("fp16"),
        "original_factory_fp16": old_factory_amp["fp16"],
        "factory_extra_params_bf16": old_extras.get("bf16"),
        "before": before,
        "after": after,
        "caveat": "This prevents FP16 GradScaler use. It does not guarantee the next training step will succeed.",
    }
