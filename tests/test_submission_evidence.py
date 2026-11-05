"""Adversarial local release gate tests. All model outputs are SIMULATED."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from policyproof.core import ROOT, make_gold, read_jsonl, sha256
from scripts.frozen_pair_inference import make_report
from scripts.final_pair_diagnostics import diagnostics_from_raw_reports
from scripts.submission_evidence import (
    _paired_deltas, paired_evaluation_from_sources,
    public_hf_evidence, source_linked_review,
)


def fake_final_reports(tmp_path: Path, monkeypatch):
    from scripts import submission_evidence as gate
    root = tmp_path
    data = root / "data"
    artifacts = root / "artifacts"
    data.mkdir()
    artifacts.mkdir()
    shutil.copy2(ROOT / "data/final_holdout_v1.jsonl", data / "final_holdout_v1.jsonl")
    rows = read_jsonl(data / "final_holdout_v1.jsonl")
    digest = sha256(data / "final_holdout_v1.jsonl")
    predictions = {
        r["id"]: make_gold(r["expected_answer"], r["expected_source_id"])
        for r in rows
    }
    rev, digest_adapter = "a" * 40, "b" * 64
    gpu = {"gpu": "FAKE_SYNTHETIC_GPU", "torch_version": "NOT_REAL_GPU",
           "loaded_model_revision_observed": rev,
           "loaded_model_revision_verified": True}
    base = make_report(rows, predictions, digest=digest, revision=rev,
                       adapter_sha=None, gpu_details=gpu)
    tuned = make_report(rows, predictions, digest=digest, revision=rev,
                        adapter_sha=digest_adapter, gpu_details=gpu)
    baseline_file = artifacts / "final_base_inference.json"
    tuned_file = artifacts / "final_tuned_inference.json"
    baseline_file.write_text(json.dumps(base), encoding="utf-8")
    tuned_file.write_text(json.dumps(tuned), encoding="utf-8")
    # Monkeypatch only the adapter lookup; source validators run against all
    # preserved fake outputs, proper scores and exact benchmark bytes.
    monkeypatch.setattr(gate, "_verified_model_adapter",
                        lambda root_, receipts: (rev, digest_adapter))
    paired = {
        "status": "paired_deterministic_audit",
        "sample_count": 60,
        "evaluation_checksum": digest,
        "baseline_model_revision": rev,
        "adapter_sha256": digest_adapter,
        "generation_config": {
            "do_sample": False, "num_beams": 1, "max_new_tokens": 160
        },
        "verified_human_semantic_improvement": False,
        "semantics_independently_reviewed": False,
        "baseline_auditkit_fingerprint": "SYNTHETIC-NOT-A-REAL-AUDITKIT-RUN",
        "candidate_auditkit_fingerprint": "SYNTHETIC-NOT-A-REAL-AUDITKIT-RUN",
        "case_matched_deterministic_counts": _paired_deltas(rows, base, tuned),
        "case_matched_exploratory_diagnostics": diagnostics_from_raw_reports(
            rows, base, tuned
        ),
        "base_report_sha256": sha256(baseline_file),
        "tuned_report_sha256": sha256(tuned_file),
    }
    pair_file = artifacts / "paired_auditkit.json"
    pair_file.write_text(json.dumps(paired), encoding="utf-8")
    return root, rows, base, tuned, pair_file


def test_source_linked_review_rejects_boolean_only(tmp_path):
    path = tmp_path / "docs"
    path.mkdir()
    (path / "final_benchmark_review_signoff.json").write_text(
        json.dumps({"review_complete": True, "reviewers": ["alice", "bob"],
                    "dataset_sha256": "0" * 64}), encoding="utf-8"
    )
    checked, reason = source_linked_review(tmp_path)
    assert checked is False
    assert "BLOCKED" in reason


def test_standalone_forged_paired_receipt_is_not_accepted(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "paired_auditkit.json").write_text(
        json.dumps({"status": "paired_deterministic_audit", "sample_count": 60,
                    "adapter_sha256": "a" * 64,
                    "case_matched_deterministic_counts": {"correct": 60}}),
        encoding="utf-8",
    )
    assert paired_evaluation_from_sources(tmp_path, [], review_is_valid=False)[0] is False
    assert paired_evaluation_from_sources(tmp_path, [], review_is_valid=True)[0] is False


def test_synthetic_correctly_linked_pair_is_just_structural(monkeypatch, tmp_path):
    root, rows, base, tuned, pair_file = fake_final_reports(tmp_path, monkeypatch)
    passed, details = paired_evaluation_from_sources(
        root, [], review_is_valid=True
    )
    assert passed is True
    assert "semantic improvements still not independently authenticated" in details
    assert json.loads(pair_file.read_text())["verified_human_semantic_improvement"] is False


@pytest.mark.parametrize("field,value", [
    ("sample_count", 48),
    ("evaluation_checksum", "0" * 64),
    ("adapter_sha256", "c" * 64),
    ("generation_config", {"do_sample": True}),
    ("verified_human_semantic_improvement", True),
    ("semantics_independently_reviewed", True),
    ("baseline_model_revision", "c" * 40),
    ("case_matched_deterministic_counts", {"correct": {"baseline_count": 60}}),
    ("case_matched_exploratory_diagnostics", {"status": "FALSIFIED"}),
    ("base_report_sha256", "f" * 64),
    ("candidate_auditkit_fingerprint", None),
])
def test_pair_metadata_tampering_never_passes(monkeypatch, tmp_path, field, value):
    root, rows, base, tuned, pair_file = fake_final_reports(tmp_path, monkeypatch)
    paired = json.loads(pair_file.read_text())
    paired[field] = value
    pair_file.write_text(json.dumps(paired), encoding="utf-8")
    passed, reason = paired_evaluation_from_sources(
        root, [], review_is_valid=True
    )
    assert passed is False
    assert "BLOCKED" in reason


def test_raw_output_mutation_fails_even_if_paired_receipt_unchanged(
    monkeypatch, tmp_path
):
    root, rows, base, tuned, pair_file = fake_final_reports(tmp_path, monkeypatch)
    tuned_path = root / "artifacts/final_tuned_inference.json"
    tampered = json.loads(tuned_path.read_text())
    tampered["predictions"][0]["raw"] = "FAKE"
    tuned_path.write_text(json.dumps(tampered), encoding="utf-8")
    passed, reason = paired_evaluation_from_sources(root, [], review_is_valid=True)
    assert passed is False
    assert "BLOCKED" in reason


def test_manifest_owner_self_verification_never_passes(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "hf_publication_manifest.json").write_text(json.dumps({
        "dataset_url": "https://huggingface.co/datasets/demo/dataset",
        "model_collection_url": "https://huggingface.co/collections/demo/model-1111",
        "dataset_collection_url": "https://huggingface.co/collections/demo/dataset-2222",
        "adapter_url": "https://huggingface.co/demo/adapter",
        "logs_url": "https://huggingface.co/datasets/demo/logs",
        "external_access_verified": True,
        "public_items_ungated_verified": True,
    }), encoding="utf-8")
    published, collections, why = public_hf_evidence(tmp_path)
    assert (published, collections) == (False, False)
    assert "receipt" in why


def test_fake_manifest_bound_receipt_without_adapter_repo_still_fails(
    monkeypatch, tmp_path
):
    from scripts import submission_evidence as gate
    docs, artifacts = tmp_path / "docs", tmp_path / "artifacts"
    docs.mkdir()
    artifacts.mkdir()
    (docs / "hf_publication_manifest.json").write_text(json.dumps({
        "dataset_url": "https://huggingface.co/datasets/demo/dataset",
        "model_collection_url": "https://huggingface.co/collections/demo/model-1111",
        "dataset_collection_url": "https://huggingface.co/collections/demo/data-2222",
        "adapter_url": "https://huggingface.co/demo/adapter",
        "logs_url": "https://huggingface.co/datasets/demo/logs",
    }), encoding="utf-8")
    (artifacts / "hf_anonymous_access_receipt.json").write_text(json.dumps({
        "model_collection_requirement_verified": True,
        "model_repositories": [],
        "dataset_repositories": [{"id": "demo/dataset",
                                   "anonymous_access_verified": True}],
    }), encoding="utf-8")
    monkeypatch.setattr(gate, "receipt_is_current", lambda receipt, manifest: True)
    published, collections, why = public_hf_evidence(tmp_path)
    assert (published, collections) == (False, False)
    assert "Adapter" in why


def test_tampering_language_slice_discordance_rejected_even_if_totals_match(monkeypatch, tmp_path):
    root, rows, base, tuned, pair_file = fake_final_reports(tmp_path, monkeypatch)
    record = json.loads(pair_file.read_text(encoding="utf-8"))
    # Aggregate deltas remain untouched, but a fake language-slice claim fails.
    report = record["case_matched_exploratory_diagnostics"]
    language = next(iter(report["by_language"]))
    report["by_language"][language]["metrics"]["strict_exact"]["improved_cases"] += 1
    pair_file.write_text(json.dumps(record), encoding="utf-8")
    accepted, reason = paired_evaluation_from_sources(
        root, [], review_is_valid=True
    )
    assert accepted is False
    assert "diagnostics do not reproduce" in reason


def test_missing_paired_uncertainty_receipt_fails_closed(monkeypatch, tmp_path):
    root, rows, base, tuned, pair_file = fake_final_reports(tmp_path, monkeypatch)
    record = json.loads(pair_file.read_text(encoding="utf-8"))
    del record["case_matched_exploratory_diagnostics"]
    pair_file.write_text(json.dumps(record), encoding="utf-8")
    accepted, reason = paired_evaluation_from_sources(root, [], review_is_valid=True)
    assert accepted is False
    assert "diagnostics do not reproduce" in reason
