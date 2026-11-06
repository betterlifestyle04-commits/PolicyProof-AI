"""Standalone AuditKIT CLI safety tests; all reports are synthetic, never GPU output."""
from __future__ import annotations

import json
import shutil

import pytest

from policyproof.core import ROOT
from scripts import auditkit_compare as cli


def fake_development_inputs(tmp_path):
    """Two fake metadata files; AuditKIT is intentionally mocked in these tests."""
    root = tmp_path
    (root / "data").mkdir()
    (root / "artifacts").mkdir()
    dataset = root / "data/development.jsonl"
    dataset.write_text(json.dumps({
        "id": "synthetic-only", "language": "en", "category": "mock",
        "question": "Fake?", "context": "[POL-F101] Fiction.",
        "expected_answer": "Fiction.", "expected_source_id": "POL-F101",
        "should_abstain": False,
    }) + "\n", encoding="utf-8")
    baseline = root / "artifacts/fake_base.json"
    candidate = root / "artifacts/fake_tuned.json"
    baseline.write_text(json.dumps({"split": "unseen-policy", "engine": "SIMULATED"}))
    candidate.write_text(json.dumps({"split": "unseen-policy", "engine": "SIMULATED"}))
    return root, baseline, candidate, dataset


def _fake_compare(monkeypatch):
    def compare(b, c, rows, digest):
        assert len(rows) == 1
        assert len(digest) == 64
        return {
            "status": "SYNTHETIC_UNIT_TEST_NOT_A_REAL_COMPARISON",
            "not_model_evidence": True,
        }
    monkeypatch.setattr(cli, "comparison", compare)


def test_cli_creates_fresh_report_without_mutating_original_evidence(tmp_path, monkeypatch):
    root, baseline, candidate, dataset = fake_development_inputs(tmp_path)
    _fake_compare(monkeypatch)
    originals = {file: file.read_bytes() for file in (baseline, candidate, dataset)}
    result = cli.compare_saved_files(
        baseline, candidate, dataset, "artifacts/paired_test.json", root=root
    )
    report = root / "artifacts/paired_test.json"
    assert report.is_file()
    assert json.loads(report.read_text()) == result
    assert all(file.read_bytes() == data for file, data in originals.items())
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.compare_saved_files(
            baseline, candidate, dataset, "artifacts/paired_test.json", root=root
        )
    assert json.loads(report.read_text()) == result


@pytest.mark.parametrize("destination", [
    "data/development.jsonl", "docs/final_benchmark_review_signoff.json",
    "reviews/private_votes.csv", "README.md",
    "artifacts/../data/development.jsonl", "artifacts/nested/result.json",
    "artifacts//result.json", "artifacts/result.csv",
    "/tmp/result.json", "artifacts\\result.json",
])
def test_comparison_output_cannot_escape_exclusive_artifacts_report_dir(
    tmp_path, monkeypatch, destination
):
    root, baseline, candidate, dataset = fake_development_inputs(tmp_path)
    _fake_compare(monkeypatch)
    originals = {file: file.read_bytes() for file in (baseline, candidate, dataset)}
    with pytest.raises(ValueError, match="Report output"):
        cli.compare_saved_files(baseline, candidate, dataset, destination, root=root)
    assert all(file.read_bytes() == data for file, data in originals.items())


@pytest.mark.parametrize("target", ["base", "candidate", "dataset"])
def test_cli_refuses_symlinked_original_source(tmp_path, monkeypatch, target):
    root, baseline, candidate, dataset = fake_development_inputs(tmp_path)
    _fake_compare(monkeypatch)
    paths = {"base": baseline, "candidate": candidate, "dataset": dataset}
    original = paths[target]
    saved = original.read_bytes()
    elsewhere = root / "original_unmodified.txt"
    elsewhere.write_bytes(saved)
    original.unlink()
    original.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlinked"):
        cli.compare_saved_files(baseline, candidate, dataset,
                                "artifacts/paired_test.json", root=root)
    assert elsewhere.read_bytes() == saved
    assert not (root / "artifacts/paired_test.json").exists()


@pytest.mark.parametrize("target", ["base", "candidate"])
def test_cli_refuses_duplicate_key_or_nonfinite_original_json(tmp_path, monkeypatch, target):
    root, baseline, candidate, dataset = fake_development_inputs(tmp_path)
    _fake_compare(monkeypatch)
    input_file = baseline if target == "base" else candidate
    input_file.write_text('{"engine":"SIMULATED","engine":"REAL","bad":NaN}')
    with pytest.raises(ValueError, match="Malformed or ambiguous"):
        cli.compare_saved_files(baseline, candidate, dataset,
                                "artifacts/paired_test.json", root=root)
    assert not (root / "artifacts/paired_test.json").exists()


@pytest.mark.parametrize("target", ["base", "candidate", "dataset"])
def test_source_mutation_during_scoring_refuses_publication(tmp_path, monkeypatch, target):
    root, baseline, candidate, dataset = fake_development_inputs(tmp_path)
    paths = {"base": baseline, "candidate": candidate, "dataset": dataset}
    def tampering_compare(*_):
        paths[target].write_bytes(paths[target].read_bytes() + b"x")
        return {"status": "SIMULATED_ONLY"}
    monkeypatch.setattr(cli, "comparison", tampering_compare)
    with pytest.raises(ValueError, match="changed during AuditKIT comparison"):
        cli.compare_saved_files(baseline, candidate, dataset,
                                "artifacts/paired_test.json", root=root)
    assert not (root / "artifacts/paired_test.json").exists()


def test_cli_fails_if_input_files_identical(tmp_path, monkeypatch):
    root, baseline, candidate, dataset = fake_development_inputs(tmp_path)
    _fake_compare(monkeypatch)
    with pytest.raises(ValueError, match="distinct source reports"):
        cli.compare_saved_files(baseline, baseline, dataset,
                                "artifacts/paired_test.json", root=root)


def test_final60_direct_cli_requires_complete_observed_gpu_report_and_scores(
    tmp_path, monkeypatch,
):
    root, baseline, candidate, _ = fake_development_inputs(tmp_path)
    # Exact 60-case benchmark bytes are real repository fixtures, but the two
    # model report files are explicitly incomplete synthetic metadata only.
    dataset = root / "data/final_holdout_v1.jsonl"
    shutil.copy2(ROOT / "data/final_holdout_v1.jsonl", dataset)
    baseline.write_text(json.dumps({
        "split": "final_candidate_holdout_v1",
        "model_revision": "a" * 40,
        "is_real_model_result": True,
        "is_finetuned": False,
    }))
    candidate.write_text(json.dumps({
        "split": "final_candidate_holdout_v1",
        "model_revision": "a" * 40,
        "adapter_sha256": "b" * 64,
        "is_real_model_result": True,
        "is_finetuned": True,
    }))
    monkeypatch.setattr(
        cli, "comparison",
        lambda *_: pytest.fail("Incomplete final evidence must fail before AuditKIT"),
    )
    with pytest.raises(ValueError, match="Saved base"):
        cli.compare_saved_files(baseline, candidate, dataset,
                                "artifacts/paired_test.json", root=root)
    assert not (root / "artifacts/paired_test.json").exists()
