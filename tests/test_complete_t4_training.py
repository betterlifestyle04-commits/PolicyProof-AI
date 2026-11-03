"""CPU-only contract tests for the one-call QLoRA recovery and packaging.

GPU-dependent training is replaced with explicit fakes. These tests prove
control flow, refusal behavior and artifact safeguards, NOT real T4 success.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import scripts.complete_t4_training as target


class FakeCuda:
    def is_available(self): return True
    def max_memory_allocated(self): return 4 * 1024**3
    def get_device_name(self, n): return "Fake T4"


class FakeTrainer:
    def __init__(self, steps=10, fail=False):
        self.config = SimpleNamespace(model=SimpleNamespace(name_or_path="CohereLabs/tiny-aya-global"))
        self.train_dataset = list(range(78))
        self.trainer = SimpleNamespace(state=SimpleNamespace(global_step=0))
        self.steps = steps
        self.fail = fail

    def train(self):
        if self.fail:
            raise RuntimeError("simulated optimizer failure")
        self.trainer.state.global_step = self.steps

    def save_model(self, output_dir):
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        (folder / "adapter_model.safetensors").write_bytes(b"toy-adapter")
        return str(folder)


@pytest.fixture
def fixture(monkeypatch, tmp_path):
    training = tmp_path / "artifacts/verified_curator_train"
    training.mkdir(parents=True)
    (training / "sft_alpaca.jsonl").write_text('{"instruction":"toy"}\n')
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))
    monkeypatch.setattr(target, "audit_training_state", lambda _x: {
        "existing_optimizer_steps": 0, "base_quantization_4bit": True,
        "trainable_elements": 3833856,
    })
    monkeypatch.setattr(target, "recover_t4", lambda _x: {
        "trl_amp_fp16": False, "trl_amp_bf16": False, "max_steps": 10,
    })
    monkeypatch.setattr("importlib.metadata.version", lambda x: "test-version")
    monkeypatch.setattr(target.subprocess, "check_output", lambda *a, **k: "test-rev\n")
    # The archive is placed at /content by design; redirect it to tmp_path
    # by using a tiny, tracked temporary path monkeypatch on pathlib.Path?
    # Here we avoid calling the archiving success branch in this unit fixture.
    return tmp_path


def test_refuses_wrong_model_before_training(fixture):
    wrapper = FakeTrainer()
    wrapper.config.model.name_or_path = "other/model"
    with pytest.raises(RuntimeError, match="Unexpected model"):
        target.recover_train_save(wrapper, project_dir=fixture)


def test_records_training_failure_without_checkpoint_claim(fixture):
    wrapper = FakeTrainer(fail=True)
    with pytest.raises(RuntimeError, match="simulated optimizer"):
        target.recover_train_save(wrapper, project_dir=fixture)
    saved = json.loads(
        (fixture / "artifacts/align_qlora_smoke/run_info.json").read_text()
    )
    assert saved["status"] == "failed"
    assert saved["optimizer_steps_observed"] == 0
    assert saved["claims_improved_safety"] is False


def test_refuses_to_overwrite_prior_result(fixture):
    root = fixture / "artifacts/align_qlora_smoke"
    root.mkdir(parents=True)
    (root / "run_info.json").write_text('{"status":"completed"}')
    with pytest.raises(FileExistsError, match="already exists"):
        target.recover_train_save(FakeTrainer(), project_dir=fixture)
