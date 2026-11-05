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
        self.dataset = None
        self.eval_dataset = None
        self.data_reload_calls = 0
        self.trainer = SimpleNamespace(state=SimpleNamespace(global_step=0))
        self.steps = steps
        self.fail = fail

    def setup_data(self):
        self.data_reload_calls += 1
        raise AssertionError("AlignTune tried a second unverified dataset load")

    def train(self):
        # Match the real pinned AlignTune TRLSFTTrainer.train() behavior:
        # it invokes setup_data() if the separate self.dataset view is None.
        if self.dataset is None:
            self.setup_data()
        if self.dataset is not self.train_dataset:
            raise AssertionError("Different training dataset entered optimizer")
        if self.fail:
            raise RuntimeError("simulated optimizer failure")
        self.trainer.state.global_step = self.steps
        return {"steps": self.steps}

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
        "eval_dataset_present": False, "trl_eval_strategy": "no",
        "trl_load_best_model_at_end": False,
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



def test_native_training_refuses_invalid_no_eval_best_model_policy(fixture, monkeypatch):
    for problem in ("best_model", "wrong_strategy", "fake_eval"):
        original = {
            "trl_amp_fp16": False, "trl_amp_bf16": False, "max_steps": 40,
            "eval_dataset_present": False, "trl_eval_strategy": "no",
            "trl_load_best_model_at_end": False,
        }
        if problem == "best_model":
            original["trl_load_best_model_at_end"] = True
        elif problem == "wrong_strategy":
            original["trl_eval_strategy"] = "steps"
        else:
            original["eval_dataset_present"] = True
        monkeypatch.setattr(target, "recover_t4",
                            lambda _w, *, target_steps=40, data=original: data)
        with pytest.raises(RuntimeError, match="unexpected evaluation|load_best_model"):
            target.recover_train_save(
                FakeTrainer(steps=40),
                project_dir=fixture,
                optimizer_steps=40,
                model_revision="a"*40,
                archive_path=fixture/(problem + ".zip"),
            )
        assert not (fixture/(problem + ".zip")).exists()
    # The no-eval config gate runs before wrapper.train() or adapter export.



def test_native40_binds_verified_dataset_before_real_train_method_can_reload(fixture):
    wrapper = FakeTrainer(steps=40)
    source = wrapper.train_dataset
    output = fixture / "single_dataset.zip"
    target.recover_train_save(
        wrapper, project_dir=fixture, model_revision="a" * 40,
        optimizer_steps=40, archive_path=output,
    )
    assert wrapper.dataset is source
    assert wrapper.train_dataset is source
    assert wrapper.data_reload_calls == 0
    saved = json.loads(
        (fixture / "artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert saved["aligntune_cached_train_dataset_reused"] is True
    assert saved["second_data_manager_load_requested"] is False


def test_native40_rejects_unverified_cached_dataset_without_training(fixture):
    wrapper = FakeTrainer(steps=40)
    original = wrapper.train_dataset
    wrapper.dataset = list(range(78))  # Looks identical but is not audited object.
    assert wrapper.dataset is not original
    with pytest.raises(RuntimeError, match="cached dataset differs"):
        target.recover_train_save(
            wrapper, project_dir=fixture, model_revision="a" * 40,
            optimizer_steps=40, archive_path=fixture / "must_not_exist.zip",
        )
    assert wrapper.trainer.state.global_step == 0
    assert wrapper.data_reload_calls == 0
    assert not (fixture / "must_not_exist.zip").exists()


def test_native40_rejects_unexpected_eval_dataset_before_training(fixture):
    wrapper = FakeTrainer(steps=40)
    wrapper.eval_dataset = ["fake evaluation examples"]
    with pytest.raises(RuntimeError, match="Unexpected evaluation dataset"):
        target.recover_train_save(
            wrapper, project_dir=fixture, model_revision="a" * 40,
            optimizer_steps=40, archive_path=fixture / "must_not_exist.zip",
        )
    assert wrapper.trainer.state.global_step == 0
    assert not (fixture / "must_not_exist.zip").exists()


def test_native40_refuses_training_if_sdk_replaces_bound_dataset(fixture):
    class ModelRebindsData(FakeTrainer):
        def train(self):
            result = super().train()
            self.dataset = list(range(78))
            return result
    wrapper = ModelRebindsData(steps=40)
    with pytest.raises(RuntimeError, match="replaced the original checked training dataset"):
        target.recover_train_save(
            wrapper, project_dir=fixture, model_revision="a" * 40,
            optimizer_steps=40, archive_path=fixture / "no_out.zip",
        )
    saved = json.loads(
        (fixture / "artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert saved["status"] == "failed"
    assert not (fixture / "no_out.zip").exists()


def test_adapter_zip_excludes_real_trl_checkpoint_and_extra_tokenizer_files(fixture):
    import zipfile
    class TrainerWithArtifacts(FakeTrainer):
        def train(self):
            result = super().train()
            out = Path(self.config.logging.output_dir)
            checkpoint = out / "checkpoint-40"
            checkpoint.mkdir()
            (checkpoint / "optimizer.pt").write_bytes(b"large checkpoint, not for sharing")
            (out / "trainer_state.json").write_text('{"global_step":40}')
            return result

        def save_model(self, output_dir):
            saved = super().save_model(output_dir)
            (Path(output_dir) / "tokenizer.json").write_text(
                '{"note":"private, extra output"}'
            )
            return saved

    wrapper = TrainerWithArtifacts(steps=40)
    output = fixture / "small_adapter.zip"
    target.recover_train_save(
        wrapper, project_dir=fixture, model_revision="a" * 40,
        optimizer_steps=40, archive_path=output,
    )
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == [
            "adapter/adapter_model.safetensors",
            "adapter/adapter_config.json",
            "run_info.json",
        ]
        assert "checkpoint-40/optimizer.pt" not in archive.namelist()
        assert "trainer_state.json" not in archive.namelist()
        assert "adapter/tokenizer.json" not in archive.namelist()
    original_out = fixture / "artifacts/align_qlora_native40"
    assert (original_out / "checkpoint-40/optimizer.pt").is_file()
    assert (original_out / "adapter/tokenizer.json").is_file()


def test_native40_requires_safe_adapter_not_legacy_pickle_binary(fixture):
    class UnsafeAdapter(FakeTrainer):
        def save_model(self, output_dir):
            folder = Path(output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "adapter_config.json").write_text('{"peft_type":"LORA"}')
            (folder / "adapter_model.bin").write_bytes(b"pickle is not acceptable")
            return str(folder)

    archive = fixture / "unsafe.zip"
    with pytest.raises(RuntimeError, match="safe PEFT adapter"):
        target.recover_train_save(
            UnsafeAdapter(steps=40),
            project_dir=fixture, model_revision="a" * 40,
            optimizer_steps=40, archive_path=archive,
        )
    assert not archive.exists()


def test_native40_never_packages_symlinked_peft_files(fixture):
    class SymlinkAdapter(FakeTrainer):
        def save_model(self, output_dir):
            folder = Path(output_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "adapter_config.json").write_text('{"peft_type":"LORA"}')
            outside = folder.parent / "outside-model.safetensors"
            outside.write_bytes(b"do-not-package")
            (folder / "adapter_model.safetensors").symlink_to(outside)
            return str(folder)

    archive = fixture / "symlink.zip"
    with pytest.raises(RuntimeError, match="regular files"):
        target.recover_train_save(
            SymlinkAdapter(steps=40),
            project_dir=fixture, model_revision="a" * 40,
            optimizer_steps=40, archive_path=archive,
        )
    assert not archive.exists()



def test_native40_training_result_and_trl_state_must_both_agree(fixture):
    class FalseResult(FakeTrainer):
        def train(self):
            super().train()
            return {"steps": 39}
    wrapper = FalseResult(steps=40)
    output = fixture / "not_published.zip"
    with pytest.raises(RuntimeError, match="training-result steps"):
        target.recover_train_save(
            wrapper, project_dir=fixture, model_revision="a"*40,
            optimizer_steps=40, archive_path=output,
        )
    note = json.loads(
        (fixture/"artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert note["status"] == "failed"
    assert note["optimizer_steps_observed"] == 40
    assert not output.exists()


def test_native40_refuses_missing_or_string_training_result_steps(fixture):
    class WrongResult(FakeTrainer):
        def train(self):
            super().train()
            return {"steps": "40"}
    output = fixture / "wrong_step_type.zip"
    with pytest.raises(RuntimeError, match="training-result steps"):
        target.recover_train_save(
            WrongResult(steps=40), project_dir=fixture,
            model_revision="a"*40, optimizer_steps=40, archive_path=output,
        )
    assert not output.exists()


def test_native40_atomic_zip_has_matching_completed_receipt(fixture):
    import zipfile
    wrapper = FakeTrainer(steps=40)
    output = fixture/"published.zip"
    target.recover_train_save(
        wrapper, project_dir=fixture, model_revision="a"*40,
        optimizer_steps=40, archive_path=output,
    )
    local = json.loads(
        (fixture/"artifacts/align_qlora_native40/run_info.json").read_text()
    )
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        inside = json.loads(archive.read("run_info.json"))
        assert local == inside
        assert archive.namelist() == [
            "adapter/adapter_model.safetensors",
            "adapter/adapter_config.json",
            "run_info.json",
        ]
    assert local["status"] == "completed"
    assert local["aligntune_train_result_steps"] == 40
    assert local["artifact_publish_method"] == "validated_temporary_zip_then_atomic_link"
    assert not list(fixture.glob(".policyproof-adapter-*.partial"))
    assert not list(
        (fixture/"artifacts/align_qlora_native40").glob(
            ".policyproof-receipt-*.partial"
        )
    )


def test_atomic_publish_failure_does_not_leave_partial_final_zip(fixture, monkeypatch):
    def fail_publish(_src, _dest):
        raise OSError("simulated disk-full during final publish")
    monkeypatch.setattr(target.os, "link", fail_publish)
    output = fixture/"should_not_exist.zip"
    with pytest.raises(OSError, match="simulated disk-full"):
        target.recover_train_save(
            FakeTrainer(steps=40),
            project_dir=fixture, model_revision="a"*40,
            optimizer_steps=40, archive_path=output,
        )
    note = json.loads(
        (fixture/"artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert note["status"] == "failed"
    assert note["optimizer_steps_observed"] == 40
    assert not output.exists()
    assert not list(fixture.glob(".policyproof-adapter-*.partial"))
    assert not list(
        (fixture/"artifacts/align_qlora_native40").glob(
            ".policyproof-receipt-*.partial"
        )
    )


def test_atomic_publish_collision_does_not_overwrite_new_owner_zip(
    fixture, monkeypatch,
):
    real_link = target.os.link
    output = fixture / "owner_claimed.zip"
    def another_writer_arrived(src, dst):
        output.write_bytes(b"OWNER'S EXISTING ARCHIVE")
        real_link(src, dst)
    monkeypatch.setattr(target.os, "link", another_writer_arrived)
    with pytest.raises(FileExistsError):
        target.recover_train_save(
            FakeTrainer(steps=40),
            project_dir=fixture, model_revision="a"*40,
            optimizer_steps=40, archive_path=output,
        )
    assert output.read_bytes() == b"OWNER'S EXISTING ARCHIVE"
    assert not list(fixture.glob(".policyproof-adapter-*.partial"))
    note = json.loads(
        (fixture/"artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert note["status"] == "failed"


def test_existing_dangling_archive_symlink_rejected_before_optimizer(fixture):
    output = fixture/"dangling.zip"
    output.symlink_to(fixture/"nonexistent.zip")
    wrapper = FakeTrainer(steps=40)
    with pytest.raises(FileExistsError, match="already exists"):
        target.recover_train_save(
            wrapper, project_dir=fixture, model_revision="a"*40,
            optimizer_steps=40, archive_path=output,
        )
    assert output.is_symlink()
    assert wrapper.trainer.state.global_step == 0


def test_failing_zip_crc_check_leaves_no_final_archive(fixture, monkeypatch):
    import zipfile
    original_zip = target.zipfile.ZipFile

    class BrokenCRC(original_zip):
        def testzip(self):
            return "adapter/adapter_model.safetensors"

    monkeypatch.setattr(target.zipfile, "ZipFile", BrokenCRC)
    output = fixture/"bad_crc.zip"
    with pytest.raises(RuntimeError, match="Corrupt or unexpected ZIP"):
        target.recover_train_save(
            FakeTrainer(steps=40),
            project_dir=fixture, model_revision="a"*40,
            optimizer_steps=40, archive_path=output,
        )
    assert not output.exists()
    assert not list(fixture.glob(".policyproof-adapter-*.partial"))
    note = json.loads(
        (fixture/"artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert note["status"] == "failed"


def test_receipt_atomic_write_failure_does_not_create_partial_file(
    fixture, monkeypatch,
):
    # Fail at the first atomic receipt update; portable ZIP is not published.
    def no_replace(_src, _dest):
        raise OSError("simulated receipt write failure")
    monkeypatch.setattr(target.os, "replace", no_replace)
    output = fixture/"receipt_write_failed.zip"
    with pytest.raises(OSError, match="simulated receipt write failure"):
        target.recover_train_save(
            FakeTrainer(steps=40),
            project_dir=fixture, model_revision="a"*40,
            optimizer_steps=40, archive_path=output,
        )
    assert not output.exists()
    assert not list(
        (fixture/"artifacts/align_qlora_native40").glob(
            ".policyproof-receipt-*.partial"
        )
    )



def test_keyboard_interrupt_leaves_preoptimizer_receipt_not_success(fixture):
    class InterruptedByColab(FakeTrainer):
        def train(self):
            assert self.trainer.state.global_step == 0
            raise KeyboardInterrupt("simulated Colab runtime stopped")

    output = fixture / "no_zip_keyboard_interrupt.zip"
    with pytest.raises(KeyboardInterrupt, match="Colab runtime"):
        target.recover_train_save(
            InterruptedByColab(steps=40), project_dir=fixture,
            model_revision="a" * 40, optimizer_steps=40, archive_path=output,
        )
    receipt = json.loads(
        (fixture / "artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert receipt["status"] == "in_progress"
    assert receipt["requested_optimizer_steps"] == 40
    assert receipt["claims_improved_safety"] is False
    assert not output.exists()


def test_postpublish_local_receipt_failure_is_not_mislabeled_as_failed_training(
    fixture, monkeypatch
):
    original = target._write_receipt_atomic
    calls = []

    def fails_completed_only(path, metadata):
        calls.append(metadata["status"])
        if metadata["status"] == "completed":
            raise OSError("simulated final receipt sync failure")
        return original(path, metadata)

    monkeypatch.setattr(target, "_write_receipt_atomic", fails_completed_only)
    output = fixture / "archive_exists_local_receipt_bad.zip"
    with pytest.raises(OSError, match="simulated final receipt sync failure"):
        target.recover_train_save(
            FakeTrainer(steps=40), project_dir=fixture,
            model_revision="a" * 40, optimizer_steps=40, archive_path=output,
        )
    assert calls == [
        "in_progress", "packaging", "completed",
        "archive_published_local_receipt_failed",
    ]
    assert output.is_file()  # Preserve the *already published* complete ZIP.
    receipt = json.loads(
        (fixture / "artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert receipt["status"] == "archive_published_local_receipt_failed"
    assert receipt["optimizer_steps_observed"] == 40
    with zipfile.ZipFile(output) as z:
        assert z.testzip() is None
        assert json.loads(z.read("run_info.json"))["status"] == "completed"


def test_postpublish_double_receipt_write_error_preserves_packaging_evidence(
    fixture, monkeypatch
):
    original = target._write_receipt_atomic

    def cannot_finalize(path, metadata):
        if metadata["status"] in {
            "completed", "archive_published_local_receipt_failed",
        }:
            raise OSError("simulated permanent local receipt write failure")
        return original(path, metadata)

    monkeypatch.setattr(target, "_write_receipt_atomic", cannot_finalize)
    output = fixture / "archive_still_exists.zip"
    with pytest.raises(OSError, match="permanent local receipt write failure"):
        target.recover_train_save(
            FakeTrainer(steps=40), project_dir=fixture,
            model_revision="a"*40, optimizer_steps=40, archive_path=output,
        )
    assert output.is_file()
    local = json.loads(
        (fixture / "artifacts/align_qlora_native40/run_info.json").read_text()
    )
    assert local["status"] == "packaging"
    assert local["requested_optimizer_steps"] == 40
