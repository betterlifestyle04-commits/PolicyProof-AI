"""CPU-only Stage 13 regression tests for code objects cached after git pull."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.check_native40_live_code import (
    assert_live_pinned_snapshot_inference,
)


def test_current_pinned_source_code_is_loaded_without_any_gpu_calls():
    actual = assert_live_pinned_snapshot_inference()
    assert actual["status"] == "LIVE_STAGE13_PINNED_SNAPSHOT_CODE_CONFIRMED"
    assert actual["model_loaded"] is False
    assert actual["gpu_inference_performed"] is False
    assert actual["training_performed"] is False


def test_old_verifier_rejected_even_after_source_file_updated():
    from scripts import frozen_pair_inference as frozen
    from scripts import dev48_gpu_pair as dev

    def old_verify_loaded_model_revision(model, expected):
        return expected
    stale = SimpleNamespace(
        verify_loaded_model_revision=old_verify_loaded_model_revision,
        _predict_on_gpu=frozen._predict_on_gpu,
    )
    with pytest.raises(RuntimeError, match="STALE_STAGE13_PYTHON_CODE"):
        assert_live_pinned_snapshot_inference(
            frozen_module=stale, dev_module=dev,
        )


def test_dev48_cached_old_predictor_rejected_if_frozen_import_was_refreshed():
    from scripts import frozen_pair_inference as frozen

    def old_predict_on_gpu(rows, revision, adapter_directory=None):
        raise AssertionError("Synthetic old predictor must never be executed")
    stale_dev = SimpleNamespace(_predict_on_gpu=old_predict_on_gpu)
    with pytest.raises(RuntimeError, match="Runtime > Restart session"):
        assert_live_pinned_snapshot_inference(
            frozen_module=frozen, dev_module=stale_dev,
        )


def test_old_frozen_predictor_rejected_even_if_dev_points_to_it():
    from scripts import frozen_pair_inference as frozen

    def old_predict_on_gpu(rows, revision, adapter_directory=None):
        raise AssertionError("Synthetic old predictor must never be executed")
    stale = SimpleNamespace(
        verify_loaded_model_revision=frozen.verify_loaded_model_revision,
        _predict_on_gpu=old_predict_on_gpu,
    )
    with pytest.raises(RuntimeError, match="STALE_STAGE13_PYTHON_CODE"):
        assert_live_pinned_snapshot_inference(
            frozen_module=stale,
            dev_module=SimpleNamespace(_predict_on_gpu=old_predict_on_gpu),
        )


def test_stage13_notebook_preflight_and_execution_both_guard_against_stale_code():
    import json
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] /
         "notebooks/13_native40_dev48_pair_t4.ipynb").read_text()
    )
    preflight = "".join(notebook["cells"][8]["source"])
    gpu = "".join(notebook["cells"][12]["source"])
    setup = "".join(notebook["cells"][2]["source"])
    assert "assert_live_pinned_snapshot_inference()" in preflight
    assert "assert_live_pinned_snapshot_inference()" in gpu
    assert gpu.index("assert_live_pinned_snapshot_inference()") < gpu.index("execute_pair(")
    assert "_stale_imports_at_checkout" in setup
    assert "_updated_source != _prior_source" in setup
    assert "RUN_DEV48_GPU_EVAL = False" in gpu
    assert "RESUME_SAVED_BASE = False" in gpu
    assert "train(" not in gpu


def test_stale_code_error_never_requires_deleting_private_training_evidence():
    from scripts.check_native40_live_code import FAILURE_GUIDANCE
    assert "Runtime > Restart session" in FAILURE_GUIDANCE
    assert "NOT factory reset" in FAILURE_GUIDANCE
    assert "Do not rerun training" in FAILURE_GUIDANCE
    assert "delete saved evidence" in FAILURE_GUIDANCE
