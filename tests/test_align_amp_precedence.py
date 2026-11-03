"""Regression tests: AlignTune v0.1.28 extra_params re-enable AMP after FP32 config.

These tests use only the standard library so CI can catch the exact bug,
without importing GPU libraries or downloading Tiny Aya.
"""
from types import SimpleNamespace
import pytest
from scripts.t4_no_amp_recovery import synchronize_factory_amp_flags


def test_stale_fp16_kwarg_cannot_override_no_amp():
    config = SimpleNamespace(
        fp16=True,
        bf16=False,
        extra_params={
            "fp16": True,
            "bf16": False,
            "precision": "fp16",
            "max_steps": 10,
            "gradient_accumulation_steps": 4,
        },
    )
    previous = synchronize_factory_amp_flags(config)
    assert previous["fp16"] is True
    assert config.fp16 is False and config.bf16 is False
    assert config.extra_params["fp16"] is False
    assert config.extra_params["bf16"] is False
    assert config.extra_params["precision"] == "fp32"
    assert config.extra_params["max_steps"] == 10
    assert config.extra_params["gradient_accumulation_steps"] == 4

    # Simulate the exact precedence used by AlignTune's
    # extract_extra_and_missing_params AFTER initial precision resolution.
    final_trl = {"fp16": False, "bf16": False}
    final_trl.update({
        k: v for k, v in config.extra_params.items()
        if k in ("fp16", "bf16")
    })
    assert final_trl == {"fp16": False, "bf16": False}


def test_missing_extra_params_fails_closed():
    config = SimpleNamespace(fp16=True, bf16=False, extra_params=None)
    with pytest.raises(ValueError, match="extra_params"):
        synchronize_factory_amp_flags(config)
    assert config.fp16 is True


def test_repeated_synchronization_is_idempotent():
    config = SimpleNamespace(
        fp16=True, bf16=True, extra_params={"fp16": True, "bf16": True}
    )
    synchronize_factory_amp_flags(config)
    assert synchronize_factory_amp_flags(config) == {"fp16": False, "bf16": False}
    assert config.fp16 is False and config.bf16 is False
