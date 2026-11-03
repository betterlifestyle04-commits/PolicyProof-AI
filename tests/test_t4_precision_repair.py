"""CPU regression tests for the T4 BF16-gradient recovery guard."""
import pytest

torch = pytest.importorskip("torch")
from scripts.t4_precision_repair import (
    repair_trainable_precision, trainable_dtype_report
)


class ToyLoRA(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(4, dtype=dtype), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.ones(3, dtype=dtype), requires_grad=True)


class ToyWrapper:
    def __init__(self, model=None, completed=0):
        self.model = model or ToyLoRA()
        self.trainer = type("FailedTrainer", (), {
            "state": type("State", (), {"global_step": completed})()
        })()


def test_fp32_lora_only_and_no_training():
    w = ToyWrapper()
    base_dtype = w.model.base.dtype
    r = repair_trainable_precision(w)
    assert r["repair_applied"] is True
    assert r["training_performed"] is False
    assert w.model.lora_A.dtype == torch.float32
    assert w.model.base.dtype == base_dtype
    assert w.model.base.requires_grad is False
    assert w.trainer is None
    assert trainable_dtype_report(w.model)["all_trainable_are_fp32"]


def test_reject_partial_training():
    w = ToyWrapper(completed=1)
    with pytest.raises(ValueError, match="already completed"):
        repair_trainable_precision(w)
    assert w.model.lora_A.dtype == torch.bfloat16


def test_reject_non_lora_trainable():
    model = ToyLoRA()
    model.base.requires_grad_(True)
    with pytest.raises(ValueError, match="non-LoRA"):
        repair_trainable_precision(ToyWrapper(model=model))


def test_reject_missing_model():
    w = ToyWrapper()
    w.model = None
    with pytest.raises(ValueError, match="No loaded model"):
        repair_trainable_precision(w)
