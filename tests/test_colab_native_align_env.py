"""CPU-only contract tests for Colab text-only native AlignTune setup.

No pip changes, torch import, CUDA, model access or training in tests.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.colab_native_align_env import (
    FRESH_IMPORT_CODE, PINNED, plan_changes, verify_text_only_native_env,
)


def env(**kw):
    return {**PINNED, "torchvision": None, "torch": "2.11.0", **kw}


def test_only_trio_pins_are_installed_without_touching_torch():
    packages, remove = plan_changes(env(aligntune=None, torchvision="0.26.0"))
    assert packages == [
        "aligntune==0.1.28", "trl==1.7.1", "bitsandbytes==0.50.2", "jedi"
    ]
    assert remove is True
    assert not any("torch==" in x or "torchvision==" in x for x in packages)


def test_healthy_text_only_env_issues_only_fresh_subprocess_probe():
    calls = []
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="PASS", stderr="")
    result = verify_text_only_native_env(
        runner=runner, python="python3", package_versions=env()
    )
    assert result["status"] == "FRESH_TEXT_ONLY_NATIVE_ALIGNTUNE_IMPORT_PASSED_NO_TRAINING"
    assert result["training_started"] is False
    assert result["model_weights_loaded"] is False
    assert result["gpu_inference_performed"] is False
    assert len(calls) == 1
    cmd, opts = calls[0]
    assert cmd[:2] == ["python3", "-c"]
    assert "BloomPreTrainedModel" in cmd[2]
    assert "from trl import SFTTrainer" in cmd[2]
    assert "from aligntune.core.backend_factory" in cmd[2]
    assert opts["env"]["USE_TF"] == "0"
    assert opts["env"]["USE_FLAX"] == "0"
    assert opts["check"] is False
    assert opts["timeout"] == 180


def test_broken_torchvision_is_uninstalled_then_hard_stops_for_restart():
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    with pytest.raises(RuntimeError, match="RESTART THE RUNTIME"):
        verify_text_only_native_env(
            runner=runner, python="python3",
            package_versions=env(torchvision="0.26.0"),
        )
    assert calls == [["python3", "-m", "pip", "uninstall", "-y", "torchvision"]]


def test_missing_pinned_package_requires_restart_even_without_vision():
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    with pytest.raises(RuntimeError, match="RESTART THE RUNTIME"):
        verify_text_only_native_env(
            runner=runner, python="python3",
            package_versions=env(aligntune=None, bitsandbytes=None),
        )
    assert len(calls) == 1 and calls[0][:4] == [
        "python3", "-m", "pip", "install"
    ]


def test_import_error_does_not_cause_reinstallation_or_training():
    calls = []
    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=1, stderr=(
                "AttributeError: partially initialized module 'torchvision' "
                "has no attribute 'extension'\n"
                "ModuleNotFoundError: Could not import BloomPreTrainedModel"
            )
        )
    with pytest.raises(RuntimeError, match="BloomPreTrainedModel.*"):
        verify_text_only_native_env(
            runner=runner, python="python3", package_versions=env()
        )
    assert len(calls) == 1
    assert calls[0][:2] == ["python3", "-c"]


def test_import_helper_source_never_loads_model_or_trains():
    assert "model_info(" not in FRESH_IMPORT_CODE
    assert "from_pretrained(" not in FRESH_IMPORT_CODE
    assert ".train(" not in FRESH_IMPORT_CODE
    assert "notebook_login(" not in FRESH_IMPORT_CODE
    assert "torchvision" not in FRESH_IMPORT_CODE
