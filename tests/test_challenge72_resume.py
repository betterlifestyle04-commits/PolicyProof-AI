"""Synthetic CPU-only recovery tests: NO CUDA, Tiny Aya model, or genuine inference.

Exercise immutable per-case journals and revised model loader callback API.
The fake outputs are an oracle fixture and never qualify as research evidence.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from policyproof.core import ROOT, make_gold
from scripts import challenge72_gpu_inference as runner
from scripts.challenge72_checkpoint import read_checkpoint
from tests.test_challenge72_gpu_inference import REVISION, frozen_fixture

JOURNAL = "artifacts/challenge72_progress"


def _runtime(revision):
    return {
        "gpu": "SYNTHETIC_CPU_TEST_NOT_A_GPU",
        "torch_version": "SYNTHETIC_TEST_ONLY",
        "loaded_model_revision_observed": revision,
        "loaded_model_revision_verified": True,
    }


def _mock_generator(items, revision, *, progress_callback=None,
                    fail_after=None):
    assert revision == REVISION
    outputs = {}
    for row in items:
        raw = make_gold(row["expected_answer"], row["expected_source_id"])
        outputs[row["id"]] = raw
        if progress_callback is not None:
            progress_callback(row, raw, _runtime(revision))
        if fail_after is not None and len(outputs) == fail_after:
            raise RuntimeError("SYNTHETIC_INTERRUPT_AFTER_CHECKPOINT")
    return outputs, _runtime(revision)


def test_interrupted_gpu_generation_reuses_original_three_cases_and_saves_full_72(
    tmp_path, monkeypatch,
):
    dataset, rows = frozen_fixture(tmp_path)
    dest = "artifacts/final_72_predictions.json"
    monkeypatch.setattr(
        runner, "_predict_on_gpu",
        lambda items, revision, *, progress_callback=None: _mock_generator(
            items, revision, progress_callback=progress_callback, fail_after=3
        ),
    )
    state = runner.preflight(
        tmp_path, dataset=dataset, revision=REVISION, output=dest,
        checkpoint_dir=JOURNAL,
    )
    assert state["status"] == "READY_FOR_EXPLICIT_GPU_EXECUTION"
    with pytest.raises(RuntimeError, match="SYNTHETIC_INTERRUPT"):
        runner.execute_gpu(
            tmp_path, dataset, REVISION, dest, checkpoint_dir=JOURNAL,
        )
    folder = tmp_path / JOURNAL
    assert (folder / "manifest.json").exists()
    case_files = sorted(p for p in folder.iterdir() if p.name.startswith("case_"))
    assert len(case_files) == 3
    original = {p: p.read_bytes() for p in case_files}
    assert not (tmp_path / dest).exists()
    state = runner.preflight(
        tmp_path, dataset=dataset, revision=REVISION, output=dest,
        checkpoint_dir=JOURNAL, resume=True,
    )
    assert state["status"] == "READY_FOR_EXPLICIT_GPU_EXECUTION"
    assert "3/72" in state["checks"]["write_once_checkpoint"]["details"]
    called_with = []
    def resume_gpu(items, revision, *, progress_callback=None):
        called_with.extend(item["id"] for item in items)
        return _mock_generator(
            items, revision, progress_callback=progress_callback,
        )
    monkeypatch.setattr(runner, "_predict_on_gpu", resume_gpu)
    result = runner.execute_gpu(
        tmp_path, dataset, REVISION, dest,
        checkpoint_dir=JOURNAL, resume=True,
    )
    assert result["new_cases_generated_in_this_process"] == 69
    assert result["cases_recovered_from_write_once_checkpoint"] == 3
    assert called_with == [row["id"] for row in rows[3:]]
    assert all(p.read_bytes() == data for p, data in original.items())
    report = json.loads((tmp_path / dest).read_text(encoding="utf-8"))
    assert report["resumed_partial_checkpoint"] is True
    assert report["cases_restored_from_checkpoint"] == 3
    assert report["human_semantic_review_completed"] is False
    assert len(report["predictions"]) == 72
    assert [p["id"] for p in report["predictions"]] == [row["id"] for row in rows]
    assert result["independent_gpu_attestation"] is False
    assert len([p for p in folder.iterdir() if p.name.startswith("case_")]) == 72


def test_resume_fully_checkpointed_run_does_not_generate_any_new_text(
    tmp_path, monkeypatch,
):
    dataset, rows = frozen_fixture(tmp_path)
    output = "artifacts/complete_from_checkpoint.json"
    def interrupted_after_last(items, revision, *, progress_callback=None):
        return _mock_generator(
            items, revision, progress_callback=progress_callback, fail_after=72,
        )
    monkeypatch.setattr(runner, "_predict_on_gpu", interrupted_after_last)
    with pytest.raises(RuntimeError, match="SYNTHETIC_INTERRUPT"):
        runner.execute_gpu(
            tmp_path, dataset, REVISION, output, checkpoint_dir=JOURNAL,
        )
    assert not (tmp_path / output).exists()
    monkeypatch.setattr(
        runner, "_predict_on_gpu",
        lambda *a, **kw: pytest.fail(
            "A complete original write-once journal must not reload a model"
        ),
    )
    result = runner.execute_gpu(
        tmp_path, dataset, REVISION, output, checkpoint_dir=JOURNAL, resume=True
    )
    assert result["new_cases_generated_in_this_process"] == 0
    assert result["base_model_inference_performed_in_this_process"] is False
    report = json.loads((tmp_path / output).read_text())
    assert report["model_inference_executed_in_this_process"] is False
    assert report["cases_restored_from_checkpoint"] == 72
    assert report["actual_gpu_inference_metadata"]["loaded_model_revision_observed"] == REVISION


def _three_saved_cases(tmp_path, monkeypatch):
    dataset, rows = frozen_fixture(tmp_path)
    monkeypatch.setattr(
        runner, "_predict_on_gpu",
        lambda items, revision, *, progress_callback=None: _mock_generator(
            items, revision, progress_callback=progress_callback, fail_after=3
        ),
    )
    with pytest.raises(RuntimeError, match="SYNTHETIC_INTERRUPT"):
        runner.execute_gpu(
            tmp_path, dataset, REVISION,
            "artifacts/unchanged_predictions.json",
            checkpoint_dir=JOURNAL,
        )
    return dataset, rows, tmp_path / JOURNAL


@pytest.mark.parametrize("mutation", [
    "raw", "source", "revision", "case_id", "case_gap", "unexpected_file",
    "symlink_case", "manifest",
])
def test_tampered_original_checkpoint_fails_without_loading_gpu(
    tmp_path, monkeypatch, mutation,
):
    dataset, rows, folder = _three_saved_cases(tmp_path, monkeypatch)
    file = folder / "case_001.json"
    case = json.loads(file.read_text(encoding="utf-8"))
    if mutation == "raw":
        case["raw"] = "FORGED TEXT WITHOUT UPDATED SHA"
    elif mutation == "source":
        case["frozen_dataset_sha256"] = "0" * 64
    elif mutation == "revision":
        case["runtime"]["loaded_model_revision_observed"] = "b" * 40
    elif mutation == "case_id":
        case["id"] = "ANOTHER_CASE"
    elif mutation == "case_gap":
        file.rename(folder / "case_004.json")
    elif mutation == "unexpected_file":
        (folder / "extra.json").write_text("{}")
    elif mutation == "symlink_case":
        copy = tmp_path / "case_copy.json"
        copy.write_bytes(file.read_bytes())
        file.unlink()
        file.symlink_to(copy)
    elif mutation == "manifest":
        x = json.loads((folder / "manifest.json").read_text())
        x["base_model_revision"] = "b" * 40
        (folder / "manifest.json").write_text(json.dumps(x))
    if mutation in ("raw", "source", "revision", "case_id"):
        file.write_text(json.dumps(case), encoding="utf-8")
    monkeypatch.setattr(
        runner, "_predict_on_gpu",
        lambda *a, **kw: pytest.fail("Tampered checkpoint must block before GPU"),
    )
    status = runner.preflight(
        tmp_path, dataset=dataset, revision=REVISION,
        output="artifacts/unchanged_predictions.json",
        checkpoint_dir=JOURNAL, resume=True,
    )
    assert status["status"] == "BLOCKED"
    with pytest.raises(ValueError, match="GPU inference blocked"):
        runner.execute_gpu(
            tmp_path, dataset, REVISION,
            "artifacts/unchanged_predictions.json",
            checkpoint_dir=JOURNAL, resume=True,
        )
    assert not (tmp_path / "artifacts/unchanged_predictions.json").exists()


def test_checkpoint_resume_rejects_different_revision(tmp_path, monkeypatch):
    dataset, rows, folder = _three_saved_cases(tmp_path, monkeypatch)
    state = runner.preflight(
        tmp_path, dataset=dataset, revision="b" * 40,
        output="artifacts/replacement_output.json",
        checkpoint_dir=JOURNAL, resume=True,
    )
    assert state["status"] == "BLOCKED"
    assert "manifest" in state["checks"]["write_once_checkpoint"]["details"]


@pytest.mark.parametrize("path", [
    "data/forged", "docs/frozen_benchmark", "../escape",
    "artifacts/../data", "artifacts/nested/foo", "artifacts//double",
    "/tmp/unrelated_checkpoint", "artifacts/unsafe:filename",
])
def test_journal_destination_stays_directly_inside_artifacts(tmp_path, path):
    dataset, rows = frozen_fixture(tmp_path)
    state = runner.preflight(
        tmp_path, dataset=dataset, revision=REVISION,
        output="artifacts/final.json", checkpoint_dir=path,
    )
    assert state["status"] == "BLOCKED"
    assert state["checks"]["write_once_checkpoint"]["passed"] is False
    assert not (tmp_path / "artifacts").exists()


def test_resume_is_never_silently_enabled_or_allowed_without_checkpoint(tmp_path):
    dataset, rows = frozen_fixture(tmp_path)
    state = runner.preflight(
        tmp_path, dataset=dataset, revision=REVISION,
        output="artifacts/final.json", checkpoint_dir=JOURNAL, resume=True,
    )
    assert state["status"] == "BLOCKED"
    assert "existing checkpoint" in state["checks"]["write_once_checkpoint"]["details"]
    no_source = runner.preflight(
        tmp_path, dataset=dataset, revision=REVISION,
        output="artifacts/final.json", resume=True,
    )
    assert no_source["status"] == "BLOCKED"
    cmd = subprocess.run(
        [sys.executable, "-m", "scripts.challenge72_gpu_inference", "--resume"],
        cwd=ROOT, text=True, capture_output=True, timeout=20,
    )
    assert cmd.returncode != 0
    assert "--resume requires --checkpoint-dir" in cmd.stderr
