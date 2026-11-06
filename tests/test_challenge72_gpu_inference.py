"""Synthetic CPU-only tests. Fake GPU responses below are NOT model output."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from policyproof.core import ROOT, make_gold, sha256
from scripts import challenge72_gpu_inference as runner
from scripts.generate_track2a_challenge import build_challenge
from scripts.score_track2a_72 import (
    KNOWN_CHALLENGE_SHA256, validate_model_report,
)

REVISION = "a" * 40


def frozen_fixture(tmp_path):
    dataset = tmp_path / "data" / "challenge_v1.jsonl"
    dataset.parent.mkdir(parents=True)
    rows = build_challenge()
    dataset.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows),
        encoding="utf-8",
    )
    assert sha256(dataset) == KNOWN_CHALLENGE_SHA256
    return dataset, rows


def fake_gpu(rows, revision):
    assert revision == REVISION
    return ({
        row["id"]: make_gold(row["expected_answer"], row["expected_source_id"])
        for row in rows
    }, {
        "gpu": "FAKE_UNIT_TEST_ONLY_NOT_T4",
        "torch_version": "SIMULATED_NOT_TORCH",
        "loaded_model_revision_observed": revision,
        "loaded_model_revision_verified": True,
    })


def test_default_read_only_preflight_reports_missing_revision_without_writes(tmp_path):
    dataset, rows = frozen_fixture(tmp_path)
    before = dataset.read_bytes()
    result = runner.preflight(tmp_path, dataset=dataset, revision=None)
    assert result["status"] == "BLOCKED"
    assert result["checks"]["frozen_challenge"]["passed"]
    assert result["checks"]["pinned_base_revision"]["passed"] is False
    assert result["gpu_execution_performed"] is False
    assert dataset.read_bytes() == before
    assert not (tmp_path / "artifacts").exists()


def test_valid_pinned_72_preflight_is_still_not_an_executed_experiment(tmp_path):
    dataset, rows = frozen_fixture(tmp_path)
    result = runner.preflight(tmp_path, dataset=dataset, revision=REVISION)
    assert result["status"] == "READY_FOR_EXPLICIT_GPU_EXECUTION"
    assert result["case_count"] == 72
    assert result["dataset_sha256"] == KNOWN_CHALLENGE_SHA256
    assert result["gpu_execution_performed"] is False
    assert result["human_semantic_review_performed"] is False
    assert "GPU access" in result["warning"]


def test_simulated_gpu_response_can_be_written_once_and_scored_later(
    tmp_path, monkeypatch,
):
    dataset, rows = frozen_fixture(tmp_path)
    monkeypatch.setattr(runner, "_predict_on_gpu", fake_gpu)
    output = "artifacts/unchanged_72.json"
    result = runner.execute_gpu(tmp_path, dataset, REVISION, output)
    file = tmp_path / output
    saved = json.loads(file.read_text(encoding="utf-8"))
    assert result["status"] == "CHALLENGE72_GPU_INFERENCE_SAVED"
    assert saved["predictions"][0]["id"] == rows[0]["id"]
    assert len(validate_model_report(saved, rows, KNOWN_CHALLENGE_SHA256)) == 72
    assert result["training_performed"] is False
    assert result["independent_gpu_attestation"] is False
    assert saved["human_semantic_review_completed"] is False
    assert saved["actual_gpu_inference_metadata"]["gpu"] == "FAKE_UNIT_TEST_ONLY_NOT_T4"
    original = file.read_bytes()
    with pytest.raises(ValueError, match="CPU prerequisites"):
        runner.execute_gpu(tmp_path, dataset, REVISION, output)
    assert file.read_bytes() == original


def test_preflight_rejects_tampered_or_symlinked_challenge(tmp_path):
    dataset, rows = frozen_fixture(tmp_path)
    dataset.write_bytes(dataset.read_bytes() + b"x")
    assert runner.preflight(tmp_path, dataset=dataset, revision=REVISION)["status"] == "BLOCKED"
    dataset.unlink()
    other, rows = frozen_fixture(tmp_path / "original")
    dataset.symlink_to(other)
    report = runner.preflight(tmp_path, dataset=dataset, revision=REVISION)
    assert report["status"] == "BLOCKED"
    assert "symlinked" in report["checks"]["frozen_challenge"]["details"]


@pytest.mark.parametrize("revision", [
    None, "", "main", "a" * 39, "A" * 40, "a" * 41,
])
def test_unpinned_or_ambiguous_model_revision_never_permits_execution(
    tmp_path, revision,
):
    dataset, rows = frozen_fixture(tmp_path)
    state = runner.preflight(tmp_path, dataset=dataset, revision=revision)
    assert state["checks"]["pinned_base_revision"]["passed"] is False


@pytest.mark.parametrize("destination", [
    "data/challenge_v1.jsonl",
    "docs/final_benchmark_review_signoff.json",
    "../outside.json",
    "artifacts/subfolder/result.json",
    "artifacts/../data/challenge_v1.jsonl",
])
def test_unsafe_prediction_output_rejected_before_gpu_load(
    tmp_path, monkeypatch, destination,
):
    dataset, rows = frozen_fixture(tmp_path)
    monkeypatch.setattr(
        runner, "_predict_on_gpu",
        lambda *_args: pytest.fail("No GPU calls permitted for unsafe output"),
    )
    with pytest.raises(ValueError, match="CPU prerequisites"):
        runner.execute_gpu(tmp_path, dataset, REVISION, destination)


@pytest.mark.parametrize("bad_metadata", [
    {"loaded_model_revision_observed": "b" * 40,
     "loaded_model_revision_verified": True},
    {"loaded_model_revision_observed": REVISION,
     "loaded_model_revision_verified": False},
    {},
])
def test_unverified_gpu_loaded_sha_does_not_save_any_predictions(
    tmp_path, monkeypatch, bad_metadata,
):
    dataset, rows = frozen_fixture(tmp_path)
    def not_trusted(items, revision):
        responses, metadata = fake_gpu(items, revision)
        metadata.update(bad_metadata)
        if bad_metadata == {}:
            metadata.pop("loaded_model_revision_observed")
        return responses, metadata
    monkeypatch.setattr(runner, "_predict_on_gpu", not_trusted)
    with pytest.raises(RuntimeError, match="matching observed loaded-model"):
        runner.execute_gpu(tmp_path, dataset, REVISION, "artifacts/predictions.json")
    assert not (tmp_path / "artifacts/predictions.json").exists()


def test_mutated_frozen_cases_after_simulated_gpu_pass_fail_without_publication(
    tmp_path, monkeypatch,
):
    dataset, rows = frozen_fixture(tmp_path)
    def fake_mutating_gpu(items, revision):
        response = fake_gpu(items, revision)
        dataset.write_bytes(dataset.read_bytes() + b" ")
        return response
    monkeypatch.setattr(runner, "_predict_on_gpu", fake_mutating_gpu)
    with pytest.raises(ValueError, match="changed during GPU inference"):
        runner.execute_gpu(tmp_path, dataset, REVISION, "artifacts/predictions.json")
    assert not (tmp_path / "artifacts/predictions.json").exists()


def test_existing_output_is_preserved_before_any_gpu_request(tmp_path, monkeypatch):
    dataset, rows = frozen_fixture(tmp_path)
    dest = tmp_path / "artifacts"
    dest.mkdir()
    report = dest / "predictions.json"
    report.write_text("IMPORTED ORIGINAL PREDICTIONS")
    monkeypatch.setattr(
        runner, "_predict_on_gpu",
        lambda *_args: pytest.fail("Never run GPU for preexisting output"),
    )
    with pytest.raises(ValueError, match="CPU prerequisites"):
        runner.execute_gpu(tmp_path, dataset, REVISION, "artifacts/predictions.json")
    assert report.read_text() == "IMPORTED ORIGINAL PREDICTIONS"


@pytest.mark.parametrize("flags", [
    ["--execute"],
    ["--execute", "--confirm-gpu-run"],
    ["--execute", "--acknowledge-gated-model-license"],
])
def test_execution_requires_all_operator_flags_without_importing_torch(flags):
    r = subprocess.run(
        [sys.executable, "-m", "scripts.challenge72_gpu_inference", *flags],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert r.returncode != 0
    assert "requires --execute --confirm-gpu-run" in r.stderr


def test_help_and_default_do_not_run_gpu():
    r = subprocess.run(
        [sys.executable, "-m", "scripts.challenge72_gpu_inference"],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert r.returncode == 0
    assert '"gpu_execution_performed": false' in r.stdout
    assert "Read-only CPU preflight" in r.stdout
