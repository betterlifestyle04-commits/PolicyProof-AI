"""CPU-only failure/recovery tests with synthetic, never-published outputs.

All apparent GPU metadata in fixtures is deliberately fake. No model loads,
optimizer, real inference, or claimed empirical performance occurs here.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys

import pytest

from policyproof.core import ROOT, make_gold, read_jsonl, sha256
from scripts.frozen_pair_inference import (
    GENERATION, MODEL_ID, execute_gpu_pair, make_report,
)
from scripts.paired_run_recovery import validate_saved_tuned, write_evidence_once
from scripts.finish_saved_pair import finish_saved_pair


def fixture_pack(monkeypatch, tmp_path):
    from scripts import frozen_pair_inference as gpu
    from scripts import finish_saved_pair as cpu
    path = tmp_path / "data" / "final_holdout_v1.jsonl"
    path.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "data/final_holdout_v1.jsonl", path)
    rows = read_jsonl(path)
    assert len(rows) == 60
    digest = sha256(path)
    rev, adapter_sha = "a" * 40, "c" * 64
    pred = {
        r["id"]: make_gold(r["expected_answer"], r["expected_source_id"])
        for r in rows
    }
    meta = {"gpu": "FAKE_TEST_GPU_NOT_REAL_INFERENCE",
            "torch_version": "NOT_A_REAL_TORCH_RUNTIME"}
    def preflight(_root, _adapter):
        return {
            "status": "READY_FOR_EXPLICIT_GPU_EXECUTION",
            "checks": {}, "dataset_sha256": digest,
            "base_revision": rev, "adapter_sha256": adapter_sha,
        }
    monkeypatch.setattr(gpu, "build_preflight", preflight)
    monkeypatch.setattr(cpu, "build_preflight", preflight)
    base, tuned, paired = (
        tmp_path / "base.json", tmp_path / "tuned.json", tmp_path / "paired.json"
    )
    return rows, pred, meta, digest, rev, adapter_sha, base, tuned, paired


def reports(pack):
    rows, pred, meta, digest, rev, adapter_sha, *_ = pack
    base = make_report(rows, pred, digest=digest, revision=rev,
                       adapter_sha=None, gpu_details=meta)
    tuned = make_report(rows, pred, digest=digest, revision=rev,
                        adapter_sha=adapter_sha, gpu_details=meta)
    return base, tuned


def test_tuned_output_survives_real_auditkit_failure_simulation(monkeypatch, tmp_path):
    pack = fixture_pack(monkeypatch, tmp_path)
    rows, pred, meta, digest, rev, adapter_sha, base, tuned, paired = pack
    from scripts import frozen_pair_inference as gpu
    from scripts import auditkit_compare
    calls = []
    def fake_gpu(*args, **kwargs):
        calls.append("tuned" if kwargs.get("adapter_directory") else "base")
        return pred, meta
    monkeypatch.setattr(gpu, "_predict_on_gpu", fake_gpu)
    def fail_audit(*args, **kwargs):
        raise RuntimeError("SIMULATED CPU AuditKIT failure")
    monkeypatch.setattr(auditkit_compare, "comparison", fail_audit)
    with pytest.raises(RuntimeError, match="SIMULATED CPU AuditKIT"):
        execute_gpu_pair(tmp_path, tmp_path / "adapter", base, tuned, paired)
    assert calls == ["base", "tuned"]
    assert base.exists() and tuned.exists()
    assert not paired.exists()
    assert validate_saved_tuned(
        tuned, rows, dataset_sha256=digest, revision=rev,
        generation_config=GENERATION, model_id=MODEL_ID,
        adapter_sha256=adapter_sha
    )["metrics"]["samples"] == 60


def test_cpu_replay_recovers_without_any_gpu_rerun(monkeypatch, tmp_path):
    pack = fixture_pack(monkeypatch, tmp_path)
    rows, pred, meta, digest, rev, adapter_sha, base, tuned, paired = pack
    first, second = reports(pack)
    write_evidence_once(base, first)
    write_evidence_once(tuned, second)
    from scripts import frozen_pair_inference as gpu
    from scripts import auditkit_compare
    monkeypatch.setattr(gpu, "_predict_on_gpu", lambda *a, **k: pytest.fail("Unexpected GPU call"))
    monkeypatch.setattr(auditkit_compare, "comparison", lambda b, t, r, d: {
        "status": "paired_deterministic_audit",
        "sample_count": len(r),
        "verified_human_semantic_improvement": False,
    })
    result = finish_saved_pair(tmp_path, tmp_path / "adapter", base, tuned, paired)
    assert result["status"] == "PAIRED_CPU_AUDITKIT_REPLAY_COMPLETED"
    assert result["real_model_inference_in_this_process"] is False
    original = json.loads(paired.read_text())
    assert original["offline_replayed_from_saved_predictions"] is True
    assert original["base_gpu_executed_in_this_process"] is False
    assert original["tuned_gpu_executed_in_this_process"] is False
    assert original["base_report_sha256"] == sha256(base)
    assert original["tuned_report_sha256"] == sha256(tuned)
    with pytest.raises(FileExistsError):
        finish_saved_pair(tmp_path, tmp_path / "adapter", base, tuned, paired)


@pytest.mark.parametrize("target,field,value", [
    ("tuned", "dataset_sha256", "other"),
    ("tuned", "adapter_sha256", "d" * 64),
    ("tuned", "model_revision", "b" * 40),
    ("tuned", "generation_config", {"do_sample": True}),
    ("tuned", "is_finetuned", False),
    ("base", "is_finetuned", True),
    ("tuned", "predictions", []),
    ("tuned", "scores", []),
    ("tuned", "metrics", {}),
    ("base", "predictions", []),
])
def test_cpu_replay_rejects_tampered_saved_evidence(
    monkeypatch, tmp_path, target, field, value
):
    pack = fixture_pack(monkeypatch, tmp_path)
    *_, base, tuned, paired = pack
    a, b = reports(pack)
    (a if target == "base" else b)[field] = value
    write_evidence_once(base, a)
    write_evidence_once(tuned, b)
    from scripts import auditkit_compare
    monkeypatch.setattr(auditkit_compare, "comparison",
                        lambda *args: pytest.fail("Invalid reports reached AuditKIT"))
    with pytest.raises(ValueError):
        finish_saved_pair(tmp_path, tmp_path / "adapter", base, tuned, paired)
    assert not paired.exists()


def test_cpu_replay_does_not_claim_success_if_auditkit_errors(monkeypatch, tmp_path):
    pack = fixture_pack(monkeypatch, tmp_path)
    *_, base, tuned, paired = pack
    a, b = reports(pack)
    write_evidence_once(base, a)
    write_evidence_once(tuned, b)
    from scripts import auditkit_compare
    monkeypatch.setattr(auditkit_compare, "comparison",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("AuditKIT unavailable")))
    with pytest.raises(RuntimeError, match="AuditKIT unavailable"):
        finish_saved_pair(tmp_path, tmp_path / "adapter", base, tuned, paired)
    assert not paired.exists() and base.exists() and tuned.exists()


def test_cpu_replay_refuses_to_replace_input_or_existing_report(monkeypatch, tmp_path):
    pack = fixture_pack(monkeypatch, tmp_path)
    *_, base, tuned, paired = pack
    a, b = reports(pack)
    write_evidence_once(base, a)
    write_evidence_once(tuned, b)
    for destination in (base, tuned, tmp_path / "data/final_holdout_v1.jsonl"):
        with pytest.raises((ValueError, FileExistsError)):
            finish_saved_pair(tmp_path, tmp_path / "adapter", base, tuned, destination)
    assert not paired.exists()


def test_cpu_replay_fails_without_genuine_signed_review_and_adapter(tmp_path):
    # NO patched preflight: real repo preflight must remain blocked.
    from scripts.finish_saved_pair import finish_saved_pair
    with pytest.raises(ValueError, match="prerequisites missing"):
        finish_saved_pair(
            ROOT, ROOT / "artifacts/align_qlora_smoke",
            tmp_path / "base.json", tmp_path / "tuned.json", tmp_path / "paired.json"
        )


def test_cli_defaults_to_read_only_and_finalize_requires_confirmation():
    root = ROOT
    command = [sys.executable, "-m", "scripts.finish_saved_pair"]
    preflight = subprocess.run(command, cwd=root, capture_output=True,
                               text=True, timeout=30)
    assert preflight.returncode == 0
    assert '"status": "BLOCKED"' in preflight.stdout
    assert "No model, GPU, or saved predictions loaded" in preflight.stdout
    incomplete = subprocess.run(
        command + ["--finalize"], cwd=root, capture_output=True,
        text=True, timeout=30,
    )
    assert incomplete.returncode != 0
    assert "requires --finalize --confirm-saved-inference" in incomplete.stderr


def test_saved_tuned_report_rejects_non_hex_adapter_sha(tmp_path):
    rows = read_jsonl(ROOT / "data/final_holdout_v1.jsonl")
    with pytest.raises(ValueError, match="complete adapter SHA"):
        validate_saved_tuned(
            tmp_path / "absent.json", rows, dataset_sha256="b" * 64,
            revision="a" * 40, generation_config=GENERATION, model_id=MODEL_ID,
            adapter_sha256="not-a-sha",
        )
