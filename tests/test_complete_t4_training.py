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
        self.config = SimpleNamespace(
            model=SimpleNamespace(name_or_path="CohereLabs/tiny-aya-global"),
            logging=SimpleNamespace(output_dir="unset"),
        )
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
    monkeypatch.setattr(target, "recover_t4", lambda _x, *, target_steps=10: {
        "trl_amp_fp16": False, "trl_amp_bf16": False, "max_steps": target_steps,
    })
    monkeypatch.setattr(target, "check_train_data", lambda _path: {
        "dataset_sha256": target.EXPECTED_SHA,
        "row_count": 78,
    })
    monkeypatch.setattr("importlib.metadata.version", lambda x: "test-version")
    monkeypatch.setattr(target.subprocess, "check_output", lambda *a, **k: "b" * 40 + "\n")
    # GPU, curated-dataset inspection and official trainer methods are all
    # explicit FAKES. The real Colab GPU and AlignTune optimizer are NOT run.
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



def test_separate_native40_trains_once_with_wrapper_and_exports_pinned_receipt(
    fixture,
):
    import zipfile
    from policyproof.core import sha256
    wrapper = FakeTrainer(steps=40)
    artifact_zip = fixture / "native40_mock.zip"
    reported = "a" * 40
    observed = {
        "loaded_model_revision_observed": reported,
        "loaded_model_revision_verified": True,
        "validation_method": "loaded_config__commit_hash_if_available",
    }
    actual = target.recover_train_save(
        wrapper, project_dir=fixture, model_revision=reported,
        optimizer_steps=40, archive_path=artifact_zip,
        loaded_revision_evidence=observed,
    )
    assert actual == artifact_zip
    assert artifact_zip.is_file()
    directory = fixture / "artifacts/align_qlora_native40"
    metadata = json.loads((directory / "run_info.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["optimizer_steps"] == 40
    assert metadata["requested_optimizer_steps"] == 40
    assert metadata["trainer_backend"] == "aligntune_wrapper"
    assert metadata["training_engine"] == "AlignTune TRL wrapper.train()"
    assert metadata["loaded_revision_independently_verified"] is True
    assert metadata["independent_gpu_execution_attested"] is False
    assert metadata["final_benchmark_or_challenge_used_for_training"] is False
    assert metadata["claims_improved_safety"] is False
    assert metadata["adapter_sha256"]["adapter_model.safetensors"] == sha256(
        directory / "adapter/adapter_model.safetensors"
    )
    assert metadata["code_commit"] == "b" * 40
    assert wrapper.config.logging.output_dir == str(directory)
    with zipfile.ZipFile(actual) as archive:
        assert set(archive.namelist()) == {
            "adapter/adapter_config.json",
            "adapter/adapter_model.safetensors",
            "run_info.json",
        }
    # No smoke-run results or second model loading occurred.
    assert not (fixture / "artifacts/align_qlora_smoke").exists()


def test_rejects_native40_when_model_revision_is_unpinned(fixture):
    for revision in (None, "main", "not40", "z" * 40):
        with pytest.raises(ValueError, match="full 40-hex"):
            target.recover_train_save(
                FakeTrainer(steps=40), project_dir=fixture,
                model_revision=revision, optimizer_steps=40,
            )
    assert not (fixture / "artifacts/align_qlora_native40").exists()


def test_training_budget_cannot_be_unbounded_bool_or_unapproved(fixture):
    for steps in (0, -1, True, 11, 39, 41, "40"):
        with pytest.raises(ValueError, match="Only 10-step smoke or 40-step"):
            target.recover_train_save(
                FakeTrainer(), project_dir=fixture, optimizer_steps=steps,
            )
    assert not (fixture / "artifacts/align_qlora_smoke").exists()


def test_wrong_loaded_revision_disqualifies_native40_before_cuda(fixture):
    for evidence in (
        {"loaded_model_revision_observed": "b" * 40,
         "loaded_model_revision_verified": False},
        {"loaded_model_revision_observed": None,
         "loaded_model_revision_verified": True},
        {"loaded_model_revision_observed": "a" * 40,
         "loaded_model_revision_verified": "yes"},
    ):
        with pytest.raises(ValueError, match="revision"):
            target.recover_train_save(
                FakeTrainer(steps=40), project_dir=fixture,
                model_revision="a" * 40, optimizer_steps=40,
                loaded_revision_evidence=evidence,
            )


def test_native40_unobserved_revision_is_recorded_as_unverified(fixture):
    archive = fixture / "unverified_native40.zip"
    target.recover_train_save(
        FakeTrainer(steps=40), project_dir=fixture,
        model_revision="a" * 40, optimizer_steps=40,
        archive_path=archive,
        loaded_revision_evidence={
            "loaded_model_revision_observed": None,
            "loaded_model_revision_verified": False,
        },
    )
    saved = json.loads(
        (fixture / "artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert saved["loaded_revision_independently_verified"] is False


def test_native40_wrong_optimizer_step_count_records_failed_not_completed(
    fixture,
):
    with pytest.raises(RuntimeError, match="Expected 40 optimizer steps, found 10"):
        target.recover_train_save(
            FakeTrainer(steps=10), project_dir=fixture,
            model_revision="a" * 40, optimizer_steps=40,
            archive_path=fixture / "failed_native40.zip",
        )
    note = fixture / "artifacts/align_qlora_native40/run_info.json"
    receipt = json.loads(note.read_text())
    assert receipt["status"] == "failed"
    assert receipt["optimizer_steps_observed"] == 10
    assert receipt["requested_optimizer_steps"] == 40
    assert not (fixture / "failed_native40.zip").exists()


def test_native40_does_not_overwrite_old_zip_or_output(fixture):
    previous_zip = fixture / "keep_existing.zip"
    previous_zip.write_bytes(b"existing-human-evidence")
    with pytest.raises(FileExistsError, match="already exists"):
        target.recover_train_save(
            FakeTrainer(steps=40), project_dir=fixture,
            optimizer_steps=40, model_revision="a" * 40,
            archive_path=previous_zip,
        )
    assert previous_zip.read_bytes() == b"existing-human-evidence"
    assert not (fixture / "artifacts/align_qlora_native40").exists()
    existing = fixture / "artifacts/align_qlora_native40"
    existing.mkdir()
    (existing / "checkpoints").mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        target.recover_train_save(
            FakeTrainer(steps=40), project_dir=fixture,
            optimizer_steps=40, model_revision="a" * 40,
            archive_path=fixture / "unused.zip",
        )
    assert (existing / "checkpoints").is_dir()


def test_original_curated_training_bytes_are_required_before_any_optimizer_call(
    fixture, monkeypatch
):
    monkeypatch.setattr(target, "check_train_data", lambda _path: {
        "dataset_sha256": "0" * 64
    })
    with pytest.raises(ValueError, match="Unapproved CuratorKIT training bytes"):
        target.recover_train_save(
            FakeTrainer(steps=40), project_dir=fixture,
            optimizer_steps=40, model_revision="a" * 40,
        )
    assert not (fixture / "artifacts/align_qlora_native40").exists()


def test_native40_training_exception_keeps_failure_receipt_without_archive(fixture):
    with pytest.raises(RuntimeError, match="simulated optimizer failure"):
        target.recover_train_save(
            FakeTrainer(steps=40, fail=True),
            project_dir=fixture, optimizer_steps=40,
            model_revision="a" * 40,
            archive_path=fixture / "failed.zip",
        )
    record = json.loads(
        (fixture / "artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert record["status"] == "failed"
    assert record["optimizer_steps_observed"] == 0
    assert not (fixture / "failed.zip").exists()
