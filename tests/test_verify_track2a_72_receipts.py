"""Adversarial local consistency checks on SYNTHETIC model-report fixtures.

These tests do NOT execute the language model, attest to GPU work, review
answers, or produce evidence of genuine safety improvement.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from policyproof.core import ROOT, sha256
from scripts import score_track2a_72 as scorer
from scripts import verify_track2a_72_receipts as verifier
from tests.test_score_track2a_72_handoff import evidence_sources


def completed_fixture(tmp_path):
    pytest.importorskip("auditkit")
    _, _, dataset, source = evidence_sources(tmp_path)
    metrics, audit = scorer.score_saved_72(
        dataset, source, "artifacts/completed_fixture", root=tmp_path
    )
    folder = tmp_path / "artifacts/completed_fixture"
    assert folder.is_dir()
    return folder, dataset, source


def verify_fixture(folder, dataset, source, root):
    return verifier.verify_completed_72(
        folder, dataset, source, root=root,
    )


def update_marker_hash(folder, field, source_path):
    marker_path = folder / "EVIDENCE_COMPLETE.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker[field] = sha256(source_path)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


def test_completed_native_auditkit_receipts_are_reproducible_without_gpu(tmp_path):
    folder, dataset, source = completed_fixture(tmp_path)
    originals = {
        path: path.read_bytes()
        for path in (folder / "measured_72_report.json",
                     folder / "auditkit_72_receipt.json",
                     folder / "EVIDENCE_COMPLETE.json", dataset, source)
    }
    verified = verify_fixture(folder, dataset, source, tmp_path)
    assert verified["status"] == "VERIFIED_LOCAL_72_CASE_SOURCE_REPLAY"
    assert verified["cases"] == 72
    assert verified["languages"] == {"en": 24, "no": 24, "ar": 24}
    assert verified["original_model_report_sha256"] == sha256(source)
    assert verified["dataset_sha256"] == sha256(dataset)
    assert verified["original_gpu_provenance_independently_authenticated"] is False
    assert verified["human_semantic_quality_verified"] is False
    assert verified["training_performed"] is False
    assert all(path.read_bytes() == content for path, content in originals.items())


@pytest.mark.parametrize("file", [
    "measured_72_report.json",
    "auditkit_72_receipt.json",
    "EVIDENCE_COMPLETE.json",
])
def test_missing_or_incomplete_saved_receipts_cannot_pass(tmp_path, file):
    folder, dataset, source = completed_fixture(tmp_path)
    (folder / file).unlink()
    with pytest.raises(ValueError, match="Incomplete or unexpected"):
        verify_fixture(folder, dataset, source, tmp_path)


def test_extra_file_in_evidence_directory_is_not_silently_accepted(tmp_path):
    folder, dataset, source = completed_fixture(tmp_path)
    (folder / "extra_private_token.txt").write_text("SYNTHETIC_ONLY")
    with pytest.raises(ValueError, match="unexpected"):
        verify_fixture(folder, dataset, source, tmp_path)


def test_symlinked_saved_receipt_refused_even_if_bytes_match(tmp_path):
    folder, dataset, source = completed_fixture(tmp_path)
    original = folder / "measured_72_report.json"
    backup = tmp_path / "original_metrics.txt"
    backup.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(backup)
    with pytest.raises(ValueError, match="symbolic"):
        verify_fixture(folder, dataset, source, tmp_path)


@pytest.mark.parametrize("file, field, patch", [
    ("measured_72_report.json", "measured_report_sha256", "language"),
    ("measured_72_report.json", "measured_report_sha256", "overall"),
    ("auditkit_72_receipt.json", "auditkit_receipt_sha256", "fingerprint"),
])
def test_rewritten_completion_hash_cannot_authorize_tampered_results(
    tmp_path, file, field, patch,
):
    folder, dataset, source = completed_fixture(tmp_path)
    path = folder / file
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if patch == "language":
        receipt["by_language"]["ar"]["metrics"]["strict_accuracy_pct"] = 100.0             if receipt["by_language"]["ar"]["metrics"]["strict_accuracy_pct"] != 100.0 else 0.0
    elif patch == "overall":
        receipt["overall"]["strict_accuracy_pct"] = -123.0
    else:
        receipt["fingerprint"] = "fake_native_fingerprint_NOT_A_REAL_RUN"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    update_marker_hash(folder, field, path)
    with pytest.raises(ValueError, match="do not reproduce|differs from current actual"):
        verify_fixture(folder, dataset, source, tmp_path)


def test_modified_source_report_invalidates_existing_completion(tmp_path):
    folder, dataset, source = completed_fixture(tmp_path)
    report = json.loads(source.read_text(encoding="utf-8"))
    report["predictions"][0]["raw"] = "ALTERED_OUTPUT"
    source.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Completed marker"):
        verify_fixture(folder, dataset, source, tmp_path)


def test_changed_challenge_bytes_rejected_before_any_replay(tmp_path):
    folder, dataset, source = completed_fixture(tmp_path)
    dataset.write_bytes(dataset.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Frozen 72-case challenge checksum"):
        verify_fixture(folder, dataset, source, tmp_path)


def test_strict_duplicate_keys_in_completion_marker_are_not_trusted(tmp_path):
    folder, dataset, source = completed_fixture(tmp_path)
    marker = folder / "EVIDENCE_COMPLETE.json"
    marker.write_text('{"status":"SAVED_72_CPU_AUDIT_COMPLETED",'
                      '"status":"FAKE_DUPLICATE"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed or ambiguous"):
        verify_fixture(folder, dataset, source, tmp_path)


def test_source_replacement_mid_replay_cannot_pass(tmp_path, monkeypatch):
    folder, dataset, source = completed_fixture(tmp_path)
    original_replay = verifier.evaluate_real_run
    def mutate_while_replaying(*args):
        result = original_replay(*args)
        source.write_bytes(source.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(verifier, "evaluate_real_run", mutate_while_replaying)
    with pytest.raises(ValueError, match="changed while it was being verified"):
        verify_fixture(folder, dataset, source, tmp_path)


@pytest.mark.parametrize("folder_arg", [
    "data/hidden", "artifacts/../data", "artifacts/nested/check",
    "artifacts//nested", "README.md", "/tmp/no_results",
])
def test_verifier_never_reads_unrestricted_directory(tmp_path, folder_arg):
    _, _, dataset, source = evidence_sources(tmp_path)
    with pytest.raises(ValueError, match="direct artifacts"):
        verify_fixture(folder_arg, dataset, source, tmp_path)


def test_normal_cli_is_read_only_blocked_on_missing_real_results():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.verify_track2a_72_receipts"],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0
    result = json.loads(proc.stdout)
    assert result["status"] == "BLOCKED"
    assert result["gpu_inference_verified"] is False
    assert result["semantic_accuracy_verified"] is False


def test_strict_cli_exits_nonzero_for_missing_real_results():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.verify_track2a_72_receipts", "--strict"],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["status"] == "BLOCKED"
