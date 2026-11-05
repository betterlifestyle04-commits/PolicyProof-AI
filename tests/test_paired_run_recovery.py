"""Recovery safety tests use made-up bytes and stubbed GPU results only.

These are NEVER real model outputs or scientifically valid model measurements.
They run on local temporary directories and do not publish any evidence.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import pytest

from policyproof.core import sha256, write_jsonl
from scripts.frozen_pair_inference import (
    GENERATION, MODEL_ID, execute_gpu_pair, make_report
)
from scripts.paired_run_recovery import (
    validate_saved_base, write_evidence_once
)


REV = "a" * 40
DIGEST = "b" * 64
OUTPUT = '{"answer":"Three.","source_id":"POL-F101","abstain":false}'


def sample_rows():
    return [
        {
            "id": "CASE-1", "language": "en", "category": "final_supported",
            "question": "How many?", "context": "[POL-F101] Three.",
            "expected_answer": "Three.", "expected_source_id": "POL-F101",
            "should_abstain": False,
        },
        {
            "id": "CASE-2", "language": "no", "category": "final_unknown",
            "question": "Unknown?", "context": "[POL-F101] Three.",
            "expected_answer": "INSUFFICIENT_EVIDENCE",
            "expected_source_id": None, "should_abstain": True,
        },
    ]


def dummy_predictions():
    return {
        "CASE-1": OUTPUT,
        "CASE-2": '{"answer":"INSUFFICIENT_EVIDENCE","source_id":null,"abstain":true}',
    }


def demo_report(rows=None):
    rows = rows or sample_rows()
    return make_report(
        rows, dummy_predictions(), digest=DIGEST, revision=REV,
        adapter_sha=None,
        gpu_details={
            "gpu": "TEST_FIXTURE_NOT_A_REAL_GPU",
            "torch_version": "FAKE_FIXTURE_ONLY",
            "loaded_model_revision_observed": REV,
            "loaded_model_revision_verified": True,
        },
    )


def validated(path, rows=None):
    return validate_saved_base(
        path, rows or sample_rows(),
        dataset_sha256=DIGEST, revision=REV,
        generation_config=GENERATION, model_id=MODEL_ID,
    )


def test_exclusive_complete_base_write_and_round_trip(tmp_path):
    path = tmp_path / "base.json"
    write_evidence_once(path, demo_report())
    assert path.is_file()
    assert validated(path)["metrics"]["samples"] == 2
    with pytest.raises(FileExistsError):
        write_evidence_once(path, {"overwritten": True})
    assert validated(path)["model_revision"] == REV


def test_writer_cleans_only_new_incomplete_file(tmp_path):
    path = tmp_path / "broken.json"
    with pytest.raises(TypeError):
        write_evidence_once(path, {"not_json_serializable": object()})
    assert not path.exists()


@pytest.mark.parametrize("field,value", [
    ("model", "different-model"),
    ("model_revision", "0" * 40),
    ("dataset_sha256", "wrong"),
    ("is_real_model_result", False),
    ("is_real_model_result", 1),
    ("is_finetuned", True),
    ("adapter_sha256", "c" * 64),
    ("split", "unseen-policy"),
    ("generation_config", {"do_sample": True}),
    ("human_semantic_review_completed", True),
    ("created_utc", "yesterday"),
    ("actual_gpu_inference_metadata", {"gpu": "fake", "torch_version": ""}),
    ("metrics", {"samples": 2}),
    ("scores", []),
    ("predictions", []),
])
def test_resume_rejects_mismatched_or_corrupted_base(tmp_path, field, value):
    report = demo_report()
    report[field] = value
    path = tmp_path / "base.json"
    write_evidence_once(path, report)
    with pytest.raises(ValueError):
        validated(path)


def test_reject_duplicate_ids_and_raw_text_corruption(tmp_path):
    p = demo_report()
    p["predictions"][1]["id"] = p["predictions"][0]["id"]
    path = tmp_path / "badids.json"
    write_evidence_once(path, p)
    with pytest.raises(ValueError, match="duplicated"):
        validated(path)
    another = demo_report()
    another["predictions"][0]["raw"] = "bad model prediction"
    path2 = tmp_path / "badraw.json"
    write_evidence_once(path2, another)
    with pytest.raises(ValueError, match="scores or metrics"):
        validated(path2)


def test_resume_rejects_duplicate_top_level_json_keys(tmp_path):
    path = tmp_path / "dup.json"
    source = json.dumps(demo_report())
    path.write_text(source.replace('"engine":', '"engine":"mock","engine":', 1))
    with pytest.raises(ValueError, match="ambiguous JSON"):
        validated(path)


def test_resume_rejects_missing_or_oversized_file(tmp_path):
    with pytest.raises(ValueError, match="Missing"):
        validated(tmp_path / "never_created.json")
    huge = tmp_path / "huge.json"
    huge.write_bytes(b"{" + b"x" * 2_000_001)
    with pytest.raises(ValueError, match="oversized"):
        validated(huge)


def temporary_runner(monkeypatch, tmp_path):
    from scripts import frozen_pair_inference as module
    rows = sample_rows()
    data = tmp_path / "data" / "final_holdout_v1.jsonl"
    write_jsonl(data, rows)
    data_sha = sha256(data)
    def preflight(root, adapter):
        return {
            "status": "READY_FOR_EXPLICIT_GPU_EXECUTION",
            "checks": {}, "dataset_sha256": data_sha,
            "base_revision": REV, "adapter_sha256": "c" * 64,
        }
    monkeypatch.setattr(module, "build_preflight", preflight)
    # The fake adapter path has no real PEFT tensor evidence. The new
    # on-disk binding is regression-tested separately with real test bytes.
    monkeypatch.setattr(module, "verify_adapter_still_unchanged",
                        lambda *_args: None)
    return rows, data_sha


def test_tuned_failure_preserves_complete_base_for_explicit_resume(monkeypatch, tmp_path):
    rows, digest = temporary_runner(monkeypatch, tmp_path)
    from scripts import frozen_pair_inference as module
    base_path = tmp_path / "base.json"
    tuned_path = tmp_path / "tuned.json"
    paired_path = tmp_path / "paired.json"
    calls = []

    def initial_gpu_run(items, revision, adapter_directory=None):
        calls.append("base" if adapter_directory is None else "tuned")
        if adapter_directory is not None:
            raise RuntimeError("SIMULATED tuned model load failure")
        return dummy_predictions(), {
            "gpu": "TEST_FIXTURE_NOT_A_REAL_GPU",
            "torch_version": "FAKE_FIXTURE_ONLY",
            "loaded_model_revision_observed": REV,
            "loaded_model_revision_verified": True,
        }

    monkeypatch.setattr(module, "_predict_on_gpu", initial_gpu_run)
    with pytest.raises(RuntimeError, match="SIMULATED"):
        execute_gpu_pair(tmp_path, tmp_path/"adapter", base_path, tuned_path, paired_path)

    assert calls == ["base", "tuned"]
    assert base_path.is_file()
    assert not tuned_path.exists() and not paired_path.exists()
    base_report = validate_saved_base(
        base_path, rows, dataset_sha256=digest,
        revision=REV, generation_config=GENERATION, model_id=MODEL_ID,
    )
    assert base_report["metrics"]["samples"] == 2
    with pytest.raises(FileExistsError, match="Use --resume-base"):
        execute_gpu_pair(tmp_path, tmp_path/"adapter", base_path, tuned_path, paired_path)

    def resume_gpu_run(items, revision, adapter_directory=None):
        assert adapter_directory is not None, "Base must not run twice on explicit resume"
        calls.append("tuned_resumed")
        return dummy_predictions(), {
            "gpu": "TEST_FIXTURE_NOT_A_REAL_GPU",
            "torch_version": "FAKE_FIXTURE_ONLY",
            "loaded_model_revision_observed": REV,
            "loaded_model_revision_verified": True,
        }

    monkeypatch.setattr(module, "_predict_on_gpu", resume_gpu_run)
    from scripts import auditkit_compare
    monkeypatch.setattr(
        auditkit_compare, "comparison",
        lambda b, t, r, d: {"status": "TEST_FIXTURE_PAIRED_ONLY"},
    )
    result = execute_gpu_pair(
        tmp_path, tmp_path/"adapter", base_path, tuned_path, paired_path,
        resume_base=True,
    )
    assert result["base_pass_reused_from_previously_saved_evidence"]
    assert calls == ["base", "tuned", "tuned_resumed"]
    assert tuned_path.is_file() and paired_path.is_file()
    saved_pair = json.loads(paired_path.read_text())
    assert saved_pair["resumed_existing_base_pass"] is True
    assert saved_pair["base_gpu_executed_in_this_process"] is False
    assert saved_pair["tuned_gpu_executed_in_this_process"] is True
    with pytest.raises(FileExistsError, match="already exists"):
        execute_gpu_pair(
            tmp_path, tmp_path/"adapter", base_path, tuned_path, paired_path,
            resume_base=True,
        )


def test_resume_fails_before_any_gpu_call_on_mutated_file(monkeypatch, tmp_path):
    rows, digest = temporary_runner(monkeypatch, tmp_path)
    from scripts import frozen_pair_inference as module
    base_path = tmp_path / "base.json"
    report = demo_report(rows)
    report["dataset_sha256"] = "0" * 64
    write_evidence_once(base_path, report)
    monkeypatch.setattr(module, "_predict_on_gpu", lambda *a, **k: pytest.fail("Unexpected GPU call"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        execute_gpu_pair(
            tmp_path, tmp_path/"adapter", base_path,
            tmp_path/"tuned.json", tmp_path/"paired.json", resume_base=True,
        )
    assert not (tmp_path/"tuned.json").exists()


def test_resume_flag_never_enabled_by_read_only_cli():
    import subprocess
    import sys
    from policyproof.core import ROOT
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.frozen_pair_inference", "--resume-base"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode != 0
    assert "--resume-base requires --execute" in completed.stderr


@pytest.mark.parametrize("mutation", [
    lambda info: info.pop("loaded_model_revision_observed"),
    lambda info: info.update(loaded_model_revision_observed="b" * 40),
    lambda info: info.update(loaded_model_revision_verified=False),
    lambda info: info.update(loaded_model_revision_verified="true"),
])
def test_saved_60_case_report_rejects_requested_only_or_conflicting_revision(
    tmp_path, mutation,
):
    report = demo_report()
    meta = report["actual_gpu_inference_metadata"]
    mutation(meta)
    path = tmp_path / "tampered_commit_evidence.json"
    write_evidence_once(path, report)
    with pytest.raises(ValueError, match="matching observed loaded-base commit"):
        validated(path)


def test_gpu_pair_rechecks_adapter_binding_before_and_after_tuned_pass(
    monkeypatch, tmp_path,
):
    rows, digest = temporary_runner(monkeypatch, tmp_path)
    from scripts import frozen_pair_inference as runner
    from scripts import auditkit_compare
    seen = []
    monkeypatch.setattr(
        runner, "verify_adapter_still_unchanged",
        lambda adapter, receipt: seen.append(receipt["adapter_sha256"]),
    )
    def simulated_gpu(items, revision, adapter_directory=None):
        assert revision == REV
        return dummy_predictions(), {
            "gpu": "NOT_REAL_T4",
            "torch_version": "UNIT-TEST_ONLY",
            "loaded_model_revision_observed": REV,
            "loaded_model_revision_verified": True,
        }
    monkeypatch.setattr(runner, "_predict_on_gpu", simulated_gpu)
    monkeypatch.setattr(
        auditkit_compare, "comparison",
        lambda *_: {"status": "SYNTHETIC_AUDIT_ONLY"},
    )
    base, tuned, paired = (
        tmp_path / "bound_base.json",
        tmp_path / "bound_tuned.json",
        tmp_path / "bound_paired.json",
    )
    result = execute_gpu_pair(tmp_path, tmp_path / "adapter", base, tuned, paired)
    assert len(seen) == 4
    assert seen == ["c" * 64] * 4
    assert result["training_performed"] is False
    assert base.is_file() and tuned.is_file() and paired.is_file()
