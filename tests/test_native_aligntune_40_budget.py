"""Native AlignTune 40-step SFTConfig contract: mocked SDK, no GPU/optimizer.

This mimics the critical AlignTune v0.1.28 precedence rule: train.extra_params
overwrites the ordinary train config when the final TRL SFTConfig is built.
Passing this test cannot prove actual model/library compatibility on a T4.
"""
from pathlib import Path
import sys
import types

import pytest

from scripts.t4_no_amp_recovery import prepare_t4_no_scaler


class Param:
    def __init__(self, *, trainable, dtype):
        self.requires_grad = trainable
        self.dtype = dtype


class Model:
    def __init__(self):
        self.items = [
            ("base.weight", Param(trainable=False, dtype="torch.float16")),
            ("layers.lora_A.weight", Param(trainable=True, dtype="torch.float32")),
        ]

    def named_parameters(self):
        return list(self.items)

    def parameters(self):
        return [p for _, p in self.items]

    def zero_grad(self, *, set_to_none):
        assert set_to_none is True


def _simulate_wrapper(monkeypatch, tmp_path, *, max_steps=10, late_steps=10):
    from scripts import t4_no_amp_recovery as module

    # The actual heavyweight SDK and CUDA are deliberately absent. Replace
    # only the imports the preflight needs, and record the SFTConfig it would
    # accept from the pre-existing mutable model/training configs.
    names = [
        "aligntune", "aligntune.core", "aligntune.core.sft",
        "aligntune.core.sft.config",
        "aligntune.core.sft.training_args_builder",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.ModuleType("torch"),
    )
    from aligntune.core.sft import config as mock_config
    from aligntune.core.sft import training_args_builder as mock_builder

    mock_config.PrecisionType = types.SimpleNamespace(FP32="fp32")
    seen = []
    def build_sft_config(*, config, train_dataset, eval_dataset, output_dir):
        merged = {
            "fp16": config.train.fp16,
            "bf16": config.train.bf16,
            "max_steps": config.train.max_steps,
            "output_dir": output_dir,
            "eval_strategy": "no" if eval_dataset is None else "steps",
            "load_best_model_at_end": config.train.load_best_model_at_end,
        }
        merged.update(config.train.extra_params)
        returned = types.SimpleNamespace(
            **{**merged, "gradient_accumulation_steps": 4},
        )
        seen.append(returned)
        return returned
    mock_builder.build_sft_config = build_sft_config

    model = Model()
    wrapper = types.SimpleNamespace(
        model=model,
        trainer=None,
        train_dataset=list(range(78)),
        eval_dataset=None,
        config=types.SimpleNamespace(
            model=types.SimpleNamespace(
                precision="fp16", quantization={"load_in_4bit": True}
            ),
            train=types.SimpleNamespace(
                fp16=True,
                bf16=False,
                max_steps=max_steps,
                load_best_model_at_end=True,
                extra_params={
                    "fp16": True,
                    "bf16": False,
                    "precision": "fp16",
                    "max_steps": late_steps,
                    "load_best_model_at_end": True,
                },
            ),
            logging=types.SimpleNamespace(output_dir=str(tmp_path / "native40")),
        ),
    )
    audit = {
        "existing_optimizer_steps": 0,
        "base_quantization_4bit": True,
        "trainable_elements": 1,
        "trainable_element_dtypes": {"torch.float32": 1},
        "gradient_element_dtypes": {},
    }
    monkeypatch.setattr(module, "audit_training_state", lambda _w: dict(audit))
    return wrapper, seen


def test_real_trl_config_precedence_is_synced_to_40_without_any_train(monkeypatch, tmp_path):
    wrapper, config_seen = _simulate_wrapper(monkeypatch, tmp_path)
    receipt = prepare_t4_no_scaler(wrapper, target_steps=40)
    assert config_seen and config_seen[0].max_steps == 40
    assert config_seen[0].fp16 is False
    assert config_seen[0].bf16 is False
    assert receipt["trl_max_steps"] == 40
    assert receipt["trl_output_dir"] == str(tmp_path / "native40")
    assert receipt["training_performed"] is False
    assert wrapper.config.train.max_steps == 40
    assert wrapper.config.train.extra_params["max_steps"] == 40
    assert wrapper.config.model.precision == "fp32"
    assert wrapper.config.train.load_best_model_at_end is False
    assert wrapper.config.train.extra_params["load_best_model_at_end"] is False
    assert config_seen[0].load_best_model_at_end is False
    assert config_seen[0].eval_strategy == "no"
    assert receipt["trl_load_best_model_at_end"] is False
    assert receipt["trl_eval_strategy"] == "no"


def test_original_ten_step_smoke_remains_compatible(monkeypatch, tmp_path):
    wrapper, seen = _simulate_wrapper(monkeypatch, tmp_path)
    receipt = prepare_t4_no_scaler(wrapper)
    assert receipt["trl_max_steps"] == 10
    assert seen[0].max_steps == 10


@pytest.mark.parametrize("target", [0, -1, 39, 41, True, "40"])
def test_unapproved_budget_rejected_before_mutating_amp_config(monkeypatch, tmp_path, target):
    wrapper, seen = _simulate_wrapper(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="exactly 10 or 40"):
        prepare_t4_no_scaler(wrapper, target_steps=target)
    assert not seen
    assert wrapper.config.train.fp16 is True


@pytest.mark.parametrize("old,late", [(999, 10), (10, 999), (40, "forty")])
def test_conflicting_old_schedule_or_late_override_refused(monkeypatch, tmp_path, old, late):
    wrapper, seen = _simulate_wrapper(
        monkeypatch, tmp_path, max_steps=old, late_steps=late
    )
    with pytest.raises(ValueError, match="Unexpected"):
        prepare_t4_no_scaler(wrapper, target_steps=40)
    assert not seen


@pytest.mark.parametrize("tamper", ["wrong_budget", "amp_reenabled", "wrong_output"])
def test_trl_builder_must_really_resolve_the_requested_policy(
    monkeypatch, tmp_path, tamper
):
    wrapper, seen = _simulate_wrapper(monkeypatch, tmp_path)
    from aligntune.core.sft import training_args_builder as builder
    original = builder.build_sft_config
    def dishonest_builder(**kwargs):
        result = original(**kwargs)
        if tamper == "wrong_budget":
            result.max_steps = 10
        elif tamper == "amp_reenabled":
            result.fp16 = True
        else:
            result.output_dir = str(tmp_path / "wrong")
        return result
    builder.build_sft_config = dishonest_builder
    match = {
        "wrong_budget": "optimizer budget mismatch",
        "amp_reenabled": "still enable mixed precision",
        "wrong_output": "unexpected output directory",
    }[tamper]
    with pytest.raises(RuntimeError, match=match):
        prepare_t4_no_scaler(wrapper, target_steps=40)



@pytest.mark.parametrize("tamper", ["best_model", "eval_steps"])
def test_no_eval_training_never_enables_best_checkpoint_without_eval(
    monkeypatch, tmp_path, tamper,
):
    wrapper, seen = _simulate_wrapper(monkeypatch, tmp_path)
    from aligntune.core.sft import training_args_builder as builder
    original = builder.build_sft_config

    def broken_builder(**kwargs):
        result = original(**kwargs)
        if tamper == "best_model":
            result.load_best_model_at_end = True
        else:
            result.eval_strategy = "steps"
        return result

    builder.build_sft_config = broken_builder
    with pytest.raises(RuntimeError, match="no-eval SFTConfig must disable"):
        prepare_t4_no_scaler(wrapper, target_steps=40)
    assert seen
    assert wrapper.config.train.load_best_model_at_end is False
    assert wrapper.config.train.extra_params["load_best_model_at_end"] is False


def test_10step_default_recovery_also_disables_unavailable_eval_best_model(
    monkeypatch, tmp_path,
):
    wrapper, seen = _simulate_wrapper(monkeypatch, tmp_path)
    receipt = prepare_t4_no_scaler(wrapper)
    assert receipt["trl_max_steps"] == 10
    assert receipt["trl_eval_strategy"] == "no"
    assert receipt["trl_load_best_model_at_end"] is False
    assert seen[0].load_best_model_at_end is False
