"""One-shot guarded recovery after repeated AlignTune T4 BF16 GradScaler failures.

Uses the already-loaded 4-bit PEFT model in the current Python process.
Requires 0 completed optimizer steps. No model download or training.
"""
from __future__ import annotations
from .t4_no_amp_recovery import audit_training_state, prepare_t4_no_scaler
from .t4_precision_repair import repair_trainable_precision


def recover_t4(wrapper, *, target_steps: int = 10) -> dict:
    if type(target_steps) is not int or target_steps not in (10, 40):
        raise ValueError("Supported native AlignTune optimizer budgets: 10 or 40")
    state = audit_training_state(wrapper)
    if state["existing_optimizer_steps"] != 0:
        raise ValueError("Completed optimizer steps exist; do not recover in place")
    if not state["base_quantization_4bit"]:
        raise ValueError("Expected 4-bit QLoRA base model")
    if state["trainable_elements"] != 3_833_856 or state["trainable_tensor_count"] != 288:
        raise ValueError("Unexpected adapter parameter count; stop for inspection")
    before = state
    cast_needed = state["trainable_element_dtypes"] != {
        "torch.float32": state["trainable_elements"]
    }
    if cast_needed:
        repair_trainable_precision(wrapper)
    no_amp = prepare_t4_no_scaler(wrapper, target_steps=target_steps)
    after = audit_training_state(wrapper)
    if (after["trainable_element_dtypes"] != {"torch.float32": 3_833_856}
        or after["gradient_element_dtypes"]
        or after["training_fp16"]
        or after["training_bf16"]
        or after["existing_optimizer_steps"] != 0
        or no_amp["trl_config_fp16"] or no_amp["trl_config_bf16"]
        or no_amp["trl_max_steps"] != target_steps):
        raise RuntimeError("Training recovery verification failed; do not retry")
    return {
        "status": "ready_for_controlled_train_attempt",
        "adapter_was_recast": cast_needed,
        "training_performed": False,
        "optimizer_steps_completed": 0,
        "trainable_elements": after["trainable_elements"],
        "parameter_dtype": "torch.float32",
        "old_gradients_cleared": not bool(after["gradient_element_dtypes"]),
        "4bit_base_unchanged": True,
        "trl_amp_fp16": no_amp["trl_config_fp16"],
        "trl_amp_bf16": no_amp["trl_config_bf16"],
        "max_steps": no_amp["trl_max_steps"],
        "native_aligntune_steps_requested": target_steps,
        "precision_before": before["precision"],
        "precision_after": after["precision"],
        "warning": "This is a configuration fix only; actual optimizer execution is not guaranteed.",
    }
