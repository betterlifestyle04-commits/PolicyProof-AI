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
    def fake_no_amp(t, *, target_steps=10):
        t.config.model.precision = "fp32"
        t.config.train.fp16 = False
        t.config.train.bf16 = False
        t.model.zero_grad(set_to_none=True)
        return {"trl_config_fp16": False, "trl_config_bf16": False, "trl_max_steps": target_steps,
                "eval_dataset_present": False,
                "trl_eval_strategy": "no",
                "trl_load_best_model_at_end": False}
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



def test_combined_native_40step_recovery_uses_40_instead_of_smoke(monkeypatch):
    wrapper = Wrapper()
    def fake_no_amp(t, *, target_steps=10):
        assert target_steps == 40
        t.config.model.precision = "fp32"
        t.config.train.fp16 = False
        t.config.train.bf16 = False
        t.model.zero_grad(set_to_none=True)
        return {"trl_config_fp16": False, "trl_config_bf16": False,
                "trl_max_steps": target_steps,
                "eval_dataset_present": False,
                "trl_eval_strategy": "no",
                "trl_load_best_model_at_end": False}
    monkeypatch.setattr(
        "scripts.recover_t4_training.prepare_t4_no_scaler", fake_no_amp
    )
    receipt = recover_t4(wrapper, target_steps=40)
    assert receipt["max_steps"] == 40
    assert receipt["native_aligntune_steps_requested"] == 40
    assert receipt["trl_eval_strategy"] == "no"
    assert receipt["trl_load_best_model_at_end"] is False
    assert receipt["training_performed"] is False
    assert receipt["optimizer_steps_completed"] == 0
    assert all(p.dtype == torch.float32 for p in wrapper.model.parameters()
               if p.requires_grad)


def test_recovery_refuses_unapproved_optimizer_steps_before_adapter_recast():
    wrapper = Wrapper()
    for steps in (0, 39, 41, True, "40", -1):
        with pytest.raises(ValueError, match="10 or 40"):
            recover_t4(wrapper, target_steps=steps)
    assert wrapper.model.lora_0.dtype == torch.bfloat16


def test_recovery_detects_wrong_budget_from_trl_config(monkeypatch):
    wrapper = Wrapper()
    def stale_trl(_wrapper, *, target_steps=10):
        assert target_steps == 40
        _wrapper.config.model.precision = "fp32"
        _wrapper.config.train.fp16 = False
        _wrapper.config.train.bf16 = False
        _wrapper.model.zero_grad(set_to_none=True)
        return {"trl_config_fp16": False, "trl_config_bf16": False,
                "trl_max_steps": 10,
                "eval_dataset_present": False,
                "trl_eval_strategy": "no",
                "trl_load_best_model_at_end": False}
    monkeypatch.setattr(
        "scripts.recover_t4_training.prepare_t4_no_scaler", stale_trl
    )
    with pytest.raises(RuntimeError, match="recovery verification failed"):
        recover_t4(wrapper, target_steps=40)
