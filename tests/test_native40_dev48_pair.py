"""CPU-only native AlignTune dev48 source protection tests; NO GPU or inference."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts import dev48_gpu_pair as dev
from tests.test_dev48_gpu_pair import fake_root, fake_training_zip, fake_archive_check

NATIVE_SHA = "dcefc5b0babf708aa1b48ac0349ddb3bc4745405c72e9636c5860d7719d8f0e9"


def fake_native_zip(tmp_path):
    archive = tmp_path / "fake_native_NOT_REAL.zip"
    with ZipFile(archive, "w") as z:
        z.writestr("run_info.json", json.dumps({
            "kind": "40-step native AlignTune QLoRA trial",
            "trainer_backend": "aligntune_wrapper",
            "model": "CohereLabs/tiny-aya-global",
            "base_revision": "a" * 40,
            "aligntune_trainer_completed": True,
            "independent_gpu_execution_attested": False,
            "claims_improved_safety": False,
            "loaded_revision_independently_verified": False,
        }))
    return archive, dev.digest(archive)


def fake_native_structural_verifier(monkeypatch):
    monkeypatch.setattr(dev, "inspect_training_archive", lambda archive: {
        "archive_sha256": dev.digest(archive),
        "backend": "aligntune_wrapper",
        "optimizer_steps_self_reported": 40,
        "expected_tiny_aya_lora_geometry": True,
        "native_aligntune_40step_receipt_checked": True,
        "file_hashes_match": True,
        "adapter_sha256": "c" * 64,
        "base_revision_pinned": False,
    })


def test_native40_accepts_exact_pinned_backend_and_warns_about_base_revision(
    tmp_path, monkeypatch,
):
    root, _ = fake_root(tmp_path)
    z, stamp = fake_native_zip(tmp_path)
    fake_native_structural_verifier(monkeypatch)
    report = dev.preflight(root, z, stamp, expected_backend="aligntune_wrapper")
    assert report["training_zip"]["backend"] == "aligntune_wrapper"
    assert report["training_zip"]["optimizer_steps_self_reported"] == 40
    assert report["training_zip"]["loaded_model_revision_verified_from_training"] is False
    assert report["model_inference_performed"] is False
    assert report["model_improvement_verified"] is False
    assert report["cases"] == 48
    assert report["generation"]["do_sample"] is False


def test_native40_cannot_pretend_to_be_older_manual_fallback(tmp_path, monkeypatch):
    root, _ = fake_root(tmp_path)
    z, stamp = fake_native_zip(tmp_path)
    fake_native_structural_verifier(monkeypatch)
    with pytest.raises(ValueError, match="requested backend"):
        dev.preflight(root, z, stamp)
    with pytest.raises(ValueError, match="requested backend"):
        dev.preflight(
            root, z, stamp, expected_backend="manual_torch_extended_fallback"
        )


def test_manual_fallback_cannot_be_relabelled_as_native_aligntune(tmp_path, monkeypatch):
    root, _ = fake_root(tmp_path)
    z, stamp = fake_training_zip(tmp_path)
    fake_archive_check(monkeypatch)
    with pytest.raises(ValueError, match="requested backend"):
        dev.preflight(root, z, stamp, expected_backend="aligntune_wrapper")


@pytest.mark.parametrize("modified", [
    {"independent_gpu_execution_attested": True},
    {"claims_improved_safety": True},
    {"loaded_revision_independently_verified": True},
    {"aligntune_trainer_completed": False},
    {"trainer_backend": "manual_torch_extended_fallback"},
    {"kind": "40 steps on GPU maybe"},
])
def test_native40_false_training_or_improvement_claims_fail(tmp_path,monkeypatch,modified):
    root, _ = fake_root(tmp_path)
    z, stamp = fake_native_zip(tmp_path)
    fake_native_structural_verifier(monkeypatch)
    with ZipFile(z, "r") as archive:
        receipt=json.loads(archive.read("run_info.json"))
    receipt.update(modified)
    with ZipFile(z, "w") as archive:
        archive.writestr("run_info.json",json.dumps(receipt))
    with pytest.raises(ValueError, match="truthful disclaimers"):
        dev.preflight(root, z, dev.digest(z), expected_backend="aligntune_wrapper")


@pytest.mark.parametrize("bad", ["missing", "0"*64, "invalid", NATIVE_SHA])
def test_native40_wrong_source_sha_is_rejected_before_any_gpu(tmp_path,monkeypatch,bad):
    root, _ = fake_root(tmp_path)
    z, stamp = fake_native_zip(tmp_path)
    fake_native_structural_verifier(monkeypatch)
    if bad == stamp:
        pytest.skip("Generated synthetic ZIP unexpectedly matches a test negative hash")
    with pytest.raises(ValueError):
        dev.preflight(root, z, bad, expected_backend="aligntune_wrapper")


def test_native40_gpu_is_never_enabled_in_committed_notebook():
    nb = (Path(__file__).resolve().parents[1] /
          "notebooks/13_native40_dev48_pair_t4.ipynb")
    notebook = json.loads(nb.read_text())
    code = "\n".join(
        "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "RUN_DEV48_GPU_EVAL = False" in code
    assert "RUN_CPU_AUDIT_RECOVERY = False" in code
    assert "expected_backend='aligntune_wrapper'" in code
    assert "dcefc5b0babf708aa1b48ac0349ddb3bc4745405c72e9636c5860d7719d8f0e9" in code
    assert "policyproof_align_native40.zip" in code
    assert "policyproof_native40_dev48_paired_gpu_results.zip" in code
    assert "native40_dev48_paired_base.json" in code
    assert "native40_dev48_paired_tuned.json" in code
    assert 'ROOT/"artifacts/dev48_paired_base.json"' not in code
    assert "verify_tiny_aya_gated_access()" in code
    assert "train_existing_t4(" not in code
    assert "trainer.train(" not in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "native40_gpu_notebook", "exec")


def test_old_dev48_evaluation_notebook_still_pins_old_manual_archive():
    nb = (Path(__file__).resolve().parents[1] /
          "notebooks/06_dev48_pair_eval_t4.ipynb")
    source=nb.read_text()
    assert "policyproof_t4_torch_extended.zip" in source
    assert "f76750c31e26db8603dab7f0eaef481929403ff96fd3ec6aaea1361a842fba3f" in source
    assert "RUN_DEV48_GPU_EVAL = False" in source
