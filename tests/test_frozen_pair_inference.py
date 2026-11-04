"""CPU-only regression coverage for a future genuine paired GPU evaluation.

Mocked or hand-authored responses never count as real inference; this suite
tests only PRECONDITIONS and metadata and does not execute the model.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from policyproof.core import ROOT, sha256
from scripts.frozen_pair_inference import (
    FINAL_REL, GENERATION, build_preflight, verify_review_signoff,
    execute_gpu_pair,
)


def test_default_project_is_blocked_without_review_or_adapter():
    report = build_preflight(ROOT)
    assert report["status"] == "BLOCKED"
    assert report["inference_performed"] is False
    assert report["model_improvement_proven"] is False
    assert report["checks"]["frozen_dataset"]["passed"] is True
    assert report["checks"]["independent_review_declared"]["passed"] is False
    assert report["checks"]["adapter_integrity"]["passed"] is False
    assert report["case_count"] == 60
    assert report["base_revision"] is None


def test_default_command_is_read_only_and_cannot_claim_inference():
    r = subprocess.run(
        [sys.executable, "-m", "scripts.frozen_pair_inference"],
        cwd=ROOT, text=True, capture_output=True, timeout=30
    )
    assert r.returncode == 0, r.stderr
    assert '"status": "BLOCKED"' in r.stdout
    assert '"inference_performed": false' in r.stdout
    assert "No GPU, model loading, or inference" in r.stdout


@pytest.mark.parametrize("flags", [
    ["--execute"],
    ["--execute", "--confirm-review"],
    ["--execute", "--confirm-gpu-run"],
])
def test_gpu_execution_requires_both_explicit_confirmations(flags):
    r = subprocess.run(
        [sys.executable, "-m", "scripts.frozen_pair_inference", *flags],
        cwd=ROOT, text=True, capture_output=True, timeout=30
    )
    assert r.returncode != 0
    assert "requires --execute --confirm-review --confirm-gpu-run" in r.stderr


def _signoff(tmp_path, *, complete=True, digest="a"*64, reviewers=None, dataset=FINAL_REL):
    path = tmp_path / "signoff.json"
    path.write_text(json.dumps({
        "review_complete": complete,
        "dataset_file": dataset,
        "dataset_sha256": digest,
        "reviewers": ["reviewer_A", "reviewer_B"] if reviewers is None else reviewers,
    }), encoding="utf-8")
    return path


def test_declared_two_reviewers_is_not_independent_identity_proof(tmp_path):
    path = _signoff(tmp_path)
    report = verify_review_signoff(path, "a"*64)
    assert report["review_declared"] is True
    assert report["reviewers_declared"] == 2
    assert report["reviewer_identity_independently_verified"] is False
    assert report["translation_validity_independently_verified"] is False
    assert report["signoff_sha256"] == sha256(path)


@pytest.mark.parametrize("bad", [
    {"complete":False},
    {"digest":"0"*64},
    {"reviewers":["reviewer_A"]},
    {"reviewers":["reviewer_A","reviewer_A"]},
    {"reviewers":["reviewer_A",None]},
    {"reviewers":["reviewer_A", "name@email.com"]},
    {"reviewers":"reviewer_A,reviewer_B"},
    {"dataset":"data/unseen_policy_test.jsonl"},
])
def test_signoff_refuses_fake_or_mismatched_review(tmp_path, bad):
    path = _signoff(tmp_path, **bad)
    with pytest.raises(ValueError):
        verify_review_signoff(path, "a"*64)


def test_signoff_rejects_corrupted_and_missing_input(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="missing"):
        verify_review_signoff(missing, "a"*64)
    corrupted = tmp_path / "bad.json"
    corrupted.write_text('{"review_complete":true')
    with pytest.raises(ValueError, match="JSON"):
        verify_review_signoff(corrupted, "a"*64)


def test_preflight_cannot_be_overridden_with_confirmation_only():
    # Even full explicit authorization cannot invent the missing adapter/review.
    with pytest.raises(ValueError, match="Refusing GPU inference"):
        execute_gpu_pair(
            ROOT, ROOT/"artifacts"/"align_qlora_smoke",
            ROOT/"artifacts"/"fake_base.json",
            ROOT/"artifacts"/"fake_tuned.json",
            ROOT/"artifacts"/"fake_paired.json",
        )


def test_frozen_generation_settings_are_deterministic():
    assert GENERATION == {"do_sample":False, "num_beams":1, "max_new_tokens":160}


def test_no_test_inference_can_enter_submission_outputs():
    """Never add files or pretend mock predictions were a completed model run."""
    assert not (ROOT/"artifacts"/"final_base_inference.json").exists()
    assert not (ROOT/"artifacts"/"final_tuned_inference.json").exists()
    assert not (ROOT/"artifacts"/"paired_auditkit.json").exists()



def test_rejects_duplicate_signoff_keys_even_when_last_value_claims_complete(tmp_path):
    path = tmp_path / "double.json"
    path.write_text(
        '{"review_complete":false,"review_complete":true,'
        '"dataset_file":"data/final_holdout_v1.jsonl",'
        '"dataset_sha256":"' + "a" * 64 + '",'
        '"reviewers":["reviewer_A","reviewer_B"]}',
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Invalid review signoff JSON"):
        verify_review_signoff(path, "a" * 64)
