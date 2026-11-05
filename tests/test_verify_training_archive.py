"""CPU-only integrity checks; fake fixtures are NEVER inference/training evidence."""
import hashlib
import json
import struct
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.verify_training_archive import (
    CURATOR_TRAIN_SHA256, digest, inspect_training_archive,
    inspect_training_directory, validate_safetensors,
)


def _safetensors(*, bad_range=False, bad_name=False):
    keys = ("base_model.model.a.lora_A.weight", "base_model.model.a.lora_B.weight")
    if bad_name:
        keys = ("model.a.weight", "model.b.weight")
    header = {
        keys[0]: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        keys[1]: {"dtype": "F32", "shape": [1], "data_offsets": [3 if bad_range else 4, 8]},
    }
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + bytes(range(8))


def _fixture(tmp_path, *, backend="aligntune_wrapper", revision="a" * 40):
    folder = tmp_path / "run"
    adapter = folder / "adapter"
    adapter.mkdir(parents=True)
    model = adapter / "adapter_model.safetensors"
    model.write_bytes(_safetensors())
    config = adapter / "adapter_config.json"
    config.write_text(json.dumps({"peft_type": "LORA", "r": 16}))
    receipt = {
        "status": "completed", "model": "CohereLabs/tiny-aya-global",
        "optimizer_steps": 10, "examples": 78,
        "dataset_sha256": CURATOR_TRAIN_SHA256,
        "trainer_backend": backend,
        "aligntune_trainer_completed": backend == "aligntune_wrapper",
        "model_revision": revision,
        "code_commit": "b" * 40,
        "adapter_sha256": {model.name: digest(model)},
        "adapter_config_sha256": digest(config),
    }
    (folder / "run_info.json").write_text(json.dumps(receipt))
    return folder


def _edit_info(directory, modify):
    path = directory / "run_info.json"
    data = json.loads(path.read_text())
    modify(data)
    path.write_text(json.dumps(data))


def test_peft_header_and_declared_sha_consistency(tmp_path):
    directory = _fixture(tmp_path)
    result = inspect_training_directory(directory)
    assert result["status"] == "ARTIFACT_INTEGRITY_CHECKED"
    assert result["file_hashes_match"]
    assert result["tensor_count"] == 2
    assert result["track1_submission_evidence_candidate"] is False
    assert result["expected_tiny_aya_lora_geometry"] is False
    assert result["training_independently_verified"] is False
    assert result["model_improvement_verified"] is False


def test_manual_fallback_is_not_aligntune_track1_completion(tmp_path):
    directory = _fixture(tmp_path, backend="manual_torch_smoke_fallback")
    result = inspect_training_directory(directory)
    assert result["backend"] == "manual_torch_smoke_fallback"
    assert result["track1_submission_evidence_candidate"] is False
    assert result["file_hashes_match"] is True


def test_unpinned_base_revision_cannot_qualify_as_release_candidate(tmp_path):
    directory = _fixture(tmp_path, revision=None)
    result = inspect_training_directory(directory)
    assert not result["base_revision_pinned"]
    assert not result["track1_submission_evidence_candidate"]


def test_rejects_missing_or_malformed_checkpoint(tmp_path):
    directory = _fixture(tmp_path)
    weights = directory / "adapter/adapter_model.safetensors"
    weights.write_text("not an adapter")
    with pytest.raises(ValueError, match="safetensors"):
        inspect_training_directory(directory)


def test_rejects_modified_tensor_bytes_even_if_header_valid(tmp_path):
    directory = _fixture(tmp_path)
    weights = directory / "adapter/adapter_model.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0x01
    weights.write_bytes(data)
    with pytest.raises(ValueError, match="weights hash"):
        inspect_training_directory(directory)


def test_rejects_mutated_config_hash(tmp_path):
    directory = _fixture(tmp_path)
    config = directory / "adapter/adapter_config.json"
    config.write_text(json.dumps({"peft_type":"LORA","r":8}))
    with pytest.raises(ValueError, match="config hash"):
        inspect_training_directory(directory)


@pytest.mark.parametrize("edit", [
    lambda d: d.update(status="failed"),
    lambda d: d.update(optimizer_steps=9),
    lambda d: d.update(optimizer_steps=True),
    lambda d: d.update(model="other"),
    lambda d: d.update(examples=60),
    lambda d: d.update(dataset_sha256="0" * 64),
    lambda d: d.update(code_commit="not-a-commit"),
    lambda d: d.update(adapter_sha256={}),
    lambda d: d.update(trainer_backend="mock"),
])
def test_rejects_untrustworthy_run_receipts(tmp_path, edit):
    directory = _fixture(tmp_path)
    _edit_info(directory, edit)
    with pytest.raises(ValueError):
        inspect_training_directory(directory)


def test_manual_fallback_requires_explicit_disclaimer(tmp_path):
    directory = _fixture(tmp_path, backend="manual_torch_smoke_fallback")
    _edit_info(directory, lambda d: d.update(aligntune_trainer_completed=True))
    with pytest.raises(ValueError, match="disclaim"):
        inspect_training_directory(directory)


@pytest.mark.parametrize("bad_range,bad_name", [(True, False), (False, True)])
def test_forged_safetensors_headers_fail(tmp_path, bad_range, bad_name):
    path = tmp_path / "adapter_model.safetensors"
    path.write_bytes(_safetensors(bad_range=bad_range, bad_name=bad_name))
    with pytest.raises(ValueError):
        validate_safetensors(path)


def test_zip_inspection_and_hash_are_read_only(tmp_path):
    directory = _fixture(tmp_path)
    zpath = tmp_path / "valid.zip"
    with ZipFile(zpath, "w") as z:
        for f in directory.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(directory))
    result = inspect_training_archive(zpath)
    assert result["archive_sha256"] == digest(zpath)
    assert result["archive_member_count"] == 3
    assert result["track1_submission_evidence_candidate"] is False


@pytest.mark.parametrize("unsafe", ["../outside.txt", "adapter/../../evil", "/etc/passwd"])
def test_zip_blocks_path_traversal(tmp_path, unsafe):
    directory = _fixture(tmp_path)
    zpath = tmp_path / "unsafe.zip"
    with ZipFile(zpath, "w") as z:
        for f in directory.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(directory))
        z.writestr(unsafe, "not allowed")
    with pytest.raises(ValueError, match="Unsafe"):
        inspect_training_archive(zpath)


def test_zip_does_not_accept_fake_nonpeft_weights(tmp_path):
    directory = _fixture(tmp_path)
    zpath = tmp_path / "bad.zip"
    (directory / "adapter/adapter_model.safetensors").write_bytes(b"bad")
    with ZipFile(zpath, "w") as z:
        for f in directory.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(directory))
    with pytest.raises(ValueError):
        inspect_training_archive(zpath)



def test_expected_tiny_aya_parameter_geometry_is_required_for_release(tmp_path):
    """Format-valid tiny fake adapters cannot pass as genuine Tiny Aya LoRA."""
    directory = _fixture(tmp_path)
    path = directory / "adapter/adapter_model.safetensors"
    header = {}
    offset = 0
    total = 0
    for index in range(144):
        for side in ("A", "B"):
            name = f"base_model.model.layers.{index}.lora_{side}.weight"
            count = 16 * 832  # 13,312 elements per tensor
            size = count * 4
            header[name] = {
                "dtype": "F32", "shape": [16, 832],
                "data_offsets": [offset, offset + size],
            }
            offset += size
            total += count
    assert len(header) == 288
    assert total == 3_833_856
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    _edit_info(directory, lambda d: d.update(
        adapter_sha256={path.name: digest(path)}
    ))
    result = inspect_training_directory(directory)
    assert result["expected_tiny_aya_lora_geometry"]
    assert result["track1_submission_evidence_candidate"]
    # A forged but internally consistent file still cannot prove actual GPU
    # optimizer steps. The release gate explicitly avoids that claim.
    assert result["training_independently_verified"] is False



def _mark_extended_receipt(directory):
    from scripts.t4_torch_fallback import training_plan
    plan = training_plan(steps=40)
    path = directory / "run_info.json"
    receipt = json.loads(path.read_text())
    receipt.update({
        "optimizer_steps": 40,
        "loaded_revision_evidence": {
            "hub_revision_claim": receipt["model_revision"],
            "loaded_model_revision_observed": receipt["model_revision"],
            "loaded_model_revision_verified": True,
            "validation_method": "loaded_config__commit_hash_if_available",
        },
        "training_plan": {
            "train_indices": plan["train_indices"],
            "validation_indices": plan["validation_indices"],
            "steps": 40,
            "seed": 42,
            "validation_on_training_source_only": True,
            "final_benchmark_or_challenge_used_for_training": False,
        },
        "training_only_validation_losses": [
            {"step": step, "training_only_validation_loss": float(step) / 50}
            for step in (10, 20, 30, 40)
        ],
    })
    path.write_text(json.dumps(receipt))


def test_extended_pytorch_artifact_is_integrity_checkable_but_not_aligntune(tmp_path):
    folder = _fixture(tmp_path, backend="manual_torch_extended_fallback")
    _mark_extended_receipt(folder)
    proof = inspect_training_directory(folder)
    assert proof["optimizer_steps_self_reported"] == 40
    assert proof["backend"] == "manual_torch_extended_fallback"
    assert proof["base_revision_pinned"] is True
    assert proof["track1_trainer_path_declared"] is False
    assert proof["track1_submission_evidence_candidate"] is False
    assert proof["training_independently_verified"] is False
    assert proof["model_improvement_verified"] is False


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(optimizer_steps=39), "valid training-only schedule"),
    (lambda r: r["training_plan"].update(
        {"final_benchmark_or_challenge_used_for_training": True}
    ), "valid training-only schedule"),
    (lambda r: r["training_plan"].update({"validation_indices":[1, 2, 3]}),
     "split does not match"),
    (lambda r: r.update(training_only_validation_losses=[
        {"step":10,"training_only_validation_loss":float("nan")},
    ]), "validation losses"),
])
def test_extended_adapter_receipt_rejects_false_training_claims(tmp_path, mutation, match):
    folder = _fixture(tmp_path, backend="manual_torch_extended_fallback")
    _mark_extended_receipt(folder)
    _edit_info(folder, mutation)
    with pytest.raises(ValueError, match=match):
        inspect_training_directory(folder)



def test_unobserved_loaded_revision_is_not_misrepresented_as_pinned(tmp_path):
    folder = _fixture(tmp_path, backend="manual_torch_extended_fallback")
    _mark_extended_receipt(folder)
    _edit_info(folder, lambda r: r["loaded_revision_evidence"].update({
        "loaded_model_revision_observed": None,
        "loaded_model_revision_verified": False,
    }))
    result = inspect_training_directory(folder)
    assert result["base_revision_pinned"] is False
    assert result["track1_submission_evidence_candidate"] is False


def test_conflicting_observed_loaded_revision_is_rejected(tmp_path):
    folder = _fixture(tmp_path, backend="manual_torch_extended_fallback")
    _mark_extended_receipt(folder)
    _edit_info(folder, lambda r: r["loaded_revision_evidence"].update({
        "loaded_model_revision_observed": "c"*40,
    }))
    with pytest.raises(ValueError, match="conflicts with declared base"):
        inspect_training_directory(folder)



def _mark_native40_receipt(directory, *, revision_confirmed=True):
    def update(info):
        revision = info["model_revision"]
        info.update({
            "status": "completed",
            "kind": "40-step native AlignTune QLoRA trial",
            "requested_optimizer_steps": 40,
            "optimizer_steps": 40,
            "training_engine": "AlignTune TRL wrapper.train()",
            "trainer_backend": "aligntune_wrapper",
            "aligntune_trainer_completed": True,
            "final_benchmark_or_challenge_used_for_training": False,
            "independent_gpu_execution_attested": False,
            "aligntune_train_result_steps": 40,
            "aligntune_cached_train_dataset_reused": True,
            "second_data_manager_load_requested": False,
            "portable_adapter_members": [
                "adapter/adapter_model.safetensors",
                "adapter/adapter_config.json",
                "run_info.json",
            ],
            "artifact_publish_method": "validated_temporary_zip_then_atomic_link",
            "loaded_revision_evidence": {
                "loaded_model_revision_observed": (
                    revision if revision_confirmed else None
                ),
                "loaded_model_revision_verified": revision_confirmed,
            },
        })
    _edit_info(directory, update)


def test_native40_receipt_stays_cpu_integrity_evidence_not_independent_gpu_proof(
    tmp_path,
):
    directory = _fixture(tmp_path)
    _mark_native40_receipt(directory)
    receipt = inspect_training_directory(directory)
    assert receipt["optimizer_steps_self_reported"] == 40
    assert receipt["backend"] == "aligntune_wrapper"
    assert receipt["native_aligntune_40step_receipt_checked"] is True
    assert receipt["base_revision_pinned"] is True
    assert receipt["base_revision_loaded_evidence_verified"] is True
    # This fixture's two tiny tensors are not genuine Tiny Aya 288-tensor LoRA.
    assert receipt["track1_submission_evidence_candidate"] is False
    assert receipt["training_independently_verified"] is False
    assert receipt["model_improvement_verified"] is False


def test_native40_hub_model_sha_alone_is_not_a_verified_loaded_base(tmp_path):
    directory = _fixture(tmp_path)
    _mark_native40_receipt(directory, revision_confirmed=False)
    audit = inspect_training_directory(directory)
    assert audit["native_aligntune_40step_receipt_checked"] is True
    assert audit["base_revision_pinned"] is False
    assert audit["base_revision_loaded_evidence_verified"] is False
    assert audit["track1_submission_evidence_candidate"] is False


@pytest.mark.parametrize("mutation,message", [
    (lambda info: info.pop("kind"), "approved provenance shape"),
    (lambda info: info.update(requested_optimizer_steps=10), "approved provenance shape"),
    (lambda info: info.update(training_engine="direct PyTorch"), "approved provenance shape"),
    (lambda info: info.update(final_benchmark_or_challenge_used_for_training=True),
     "approved provenance shape"),
    (lambda info: info.update(independent_gpu_execution_attested=True),
     "approved provenance shape"),
    (lambda info: info.update(aligntune_train_result_steps=39),
     "approved provenance shape"),
    (lambda info: info.update(aligntune_train_result_steps="40"),
     "approved provenance shape"),
    (lambda info: info.update(aligntune_cached_train_dataset_reused=False),
     "approved provenance shape"),
    (lambda info: info.update(second_data_manager_load_requested=True),
     "approved provenance shape"),
    (lambda info: info.update(portable_adapter_members=["full_model.bin"]),
     "approved provenance shape"),
    (lambda info: info.update(artifact_publish_method="unsafe_direct_zip"),
     "approved provenance shape"),
    (lambda info: info["loaded_revision_evidence"].update(
        loaded_model_revision_observed="c"*40,
    ), "conflicts with declared base"),
    (lambda info: info["loaded_revision_evidence"].update(
        loaded_model_revision_verified=True,
        loaded_model_revision_observed=None,
    ), "lacks matching observed SHA"),
])
def test_native40_does_not_accept_forged_trainer_or_revision_claims(
    tmp_path, mutation, message,
):
    directory = _fixture(tmp_path)
    _mark_native40_receipt(directory)
    _edit_info(directory, mutation)
    with pytest.raises(ValueError, match=message):
        inspect_training_directory(directory)


def test_native40_cannot_inherit_a_10step_smoke_label(tmp_path):
    directory = _fixture(tmp_path)
    _edit_info(directory, lambda info: info.update(optimizer_steps=40))
    with pytest.raises(ValueError, match="approved provenance shape"):
        inspect_training_directory(directory)
