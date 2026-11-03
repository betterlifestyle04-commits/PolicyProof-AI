"""Guards for the single-step T4 post-failure recovery orchestrator."""
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")
from scripts.recover_t4_training import recover_t4


class LoRAToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(1, dtype=torch.float16), requires_grad=False)
        for i in range(288):
            self.register_parameter(
                f"lora_{i}",
                torch.nn.Parameter(torch.ones(13312, dtype=torch.bfloat16))
            )


class Wrapper:
    def __init__(self, steps=0):
        self.model = LoRAToy()
        self.trainer = SimpleNamespace(state=SimpleNamespace(global_step=steps))
        self.train_dataset = list(range(78))
        self.config = SimpleNamespace(
            model=SimpleNamespace(precision="fp16",
                                  quantization={"load_in_4bit": True,
                                                "bnb_4bit_compute_dtype": "float16"}),
            train=SimpleNamespace(fp16=True, bf16=False),
        )


def test_combined_recovery_casts_only_lora_and_disables_amp(monkeypatch):
    wrapper = Wrapper()
    def fake_no_amp(t):
        t.config.model.precision = "fp32"
        t.config.train.fp16 = False
        t.config.train.bf16 = False
        t.model.zero_grad(set_to_none=True)
        return {"trl_config_fp16": False, "trl_config_bf16": False, "trl_max_steps": 10}
    monkeypatch.setattr("scripts.recover_t4_training.prepare_t4_no_scaler", fake_no_amp)
    report = recover_t4(wrapper)
    assert report["status"] == "ready_for_controlled_train_attempt"
    assert report["adapter_was_recast"] is True
    assert report["optimizer_steps_completed"] == 0
    assert report["trainable_elements"] == 3_833_856
    assert wrapper.model.base.dtype == torch.float16
    assert wrapper.model.base.requires_grad is False
    assert all(p.dtype == torch.float32
               for p in wrapper.model.parameters() if p.requires_grad)


def test_combined_recovery_refuses_partial_optimizer_progress(monkeypatch):
    wrapper = Wrapper(steps=1)
    with pytest.raises(ValueError, match="Completed optimizer steps"):
        recover_t4(wrapper)
    assert wrapper.model.lora_0.dtype == torch.bfloat16
