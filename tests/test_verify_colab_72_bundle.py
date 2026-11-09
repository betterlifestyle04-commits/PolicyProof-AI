"""CPU-only strict Colab ZIP verification tests; saved GPU responses are ORACLE FIXTURES.

No model was loaded or run in these tests. Fake self-recorded GPU metadata
is acceptable only for demonstrating that local integrity CAN be recomputed,
and DOES NOT authenticate any actual GPU experiment.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import stat
import subprocess
import sys
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import pytest

from policyproof.core import ROOT, sha256
from scripts import verify_colab_72_bundle as receiver
from scripts.score_track2a_72 import score_saved_72
from tests.test_score_track2a_72_handoff import evidence_sources

MODEL_SHA = "f" * 40
NOT_REAL_GPU_FIXTURE = True


def _synthetic_bundle(tmp_path):
    pytest.importorskip("auditkit")
    _, original, dataset, model_report = evidence_sources(tmp_path)
    # The original report is a known ORACLE test fixture with fabricated
    # "is_real_model_result" metadata; never describe it as a GPU run.
    # Stage 12 always records this explicit false declaration; normalize
    # the older generic scorer fixture to the notebook's exact ZIP schema.
    original["human_semantic_review_completed"] = False
    model_report.write_text(
        json.dumps(original, ensure_ascii=False), encoding="utf-8"
    )
    result, receipt = score_saved_72(
        dataset, model_report, "artifacts/cpu_72_analysis", root=tmp_path
    )
    folder = tmp_path / "artifacts/cpu_72_analysis"
    contents = {
        "frozen_72_challenge.jsonl": dataset.read_bytes(),
        "original_72_raw_predictions.json": model_report.read_bytes(),
        "auditkit_72/measured_72_report.json":
            (folder / "measured_72_report.json").read_bytes(),
        "auditkit_72/auditkit_72_receipt.json":
            (folder / "auditkit_72_receipt.json").read_bytes(),
        "auditkit_72/EVIDENCE_COMPLETE.json":
            (folder / "EVIDENCE_COMPLETE.json").read_bytes(),
        "OWNER_SCOPE_MANIFEST.json": json.dumps({
            "status": "OWNER_SAVED_ORIGINAL_72_MODEL_OUTPUTS_REPLAYED_NOT_EXTERNALLY_ATTESTED",
            "dataset_sha256": sha256(dataset),
            "code_commit": MODEL_SHA,
            "original_gpu_provenance_independently_verified": False,
            "human_semantic_review_complete": False,
            "training_performed": False,
            "official_lexsi_submission_completed": False,
        }).encode(),
    }
    return contents, dataset, model_report, folder


def _zip_bytes(contents: dict, *, extra=None, duplicate=None, symlink=None):
    out = BytesIO()
    with ZipFile(out, "w", compression=ZIP_STORED) as z:
        for name, data in contents.items():
            info = ZipInfo(name, (2026, 11, 8, 0, 0, 0))
            info.compress_type = ZIP_STORED
            info.external_attr = ((stat.S_IFLNK if name == symlink else stat.S_IFREG) | 0o644) << 16
            z.writestr(info, data)
        if extra is not None:
            z.writestr(extra, b"EXTRA_OR_PRIVATE_DATA")
        if duplicate is not None:
            z.writestr(duplicate, contents[duplicate])
    return out.getvalue()


def test_real_native_cpu_replay_from_synthetic_owner_zip_is_read_only(tmp_path):
    contents, dataset, report, folder = _synthetic_bundle(tmp_path)
    before = {str(p): p.read_bytes() for p in [dataset, report, *folder.iterdir()]}
    original = tmp_path / "simulated_owner_bundle.zip"
    original.write_bytes(_zip_bytes(contents))
    original_bytes = original.read_bytes()
    output = receiver.inspect_bundle(original)
    assert output["status"] == "VERIFIED_PRIVATE_STAGE12_72_LOCAL_SOURCE_REPLAY_NOT_GPU_ATTESTATION"
    assert output["original_raw_saved_responses_source_replayed"] == 72
    assert output["original_frozen_dataset_sha256"] == sha256(dataset)
    assert output["owner_zip_sha256"] == sha256(original)
    assert output["native_auditkit_version"] == "1.1.3"
    assert output["zip_members_verified"] == 6
    assert output["original_model_report_sha256"] == sha256(report)
    assert all(output[flag] is False for flag in (
        "original_gpu_execution_independently_authenticated",
        "human_semantic_quality_verified",
        "native_aligntune_training_performed",
        "public_hf_publication_performed",
        "official_lexsi_submission_completed",
        "project_evidence_files_overwritten",
    ))
    assert original.read_bytes() == original_bytes
    assert {str(p): p.read_bytes() for p in [dataset, report, *folder.iterdir()]} == before
    assert not (ROOT / "artifacts/actual_72_case_evaluation").exists()


@pytest.mark.parametrize("change", [
    "missing", "extra", "duplicate", "symlink", "edited_gold", "edited_model",
    "edited_receipt", "upgraded_human_claim", "forged_done_marker", "uncompressed_override",
])
def test_invalid_bundle_contents_are_rejected(tmp_path, change):
    contents, dataset, report, folder = _synthetic_bundle(tmp_path)
    items = dict(contents)
    extra, dup, symlink = None, None, None
    if change == "missing":
        items.pop("auditkit_72/EVIDENCE_COMPLETE.json")
    if change == "extra":
        extra = "../secret.txt"
    if change == "duplicate":
        dup = "original_72_raw_predictions.json"
    if change == "symlink":
        symlink = "original_72_raw_predictions.json"
    if change == "edited_gold":
        items["frozen_72_challenge.jsonl"] += b"\n"
    if change == "edited_model":
        original_report = json.loads(items["original_72_raw_predictions.json"])
        original_report["predictions"][0]["raw"] = "FORGED_ORIGINAL_MODEL_RESPONSE"
        items["original_72_raw_predictions.json"] = json.dumps(original_report).encode()
    if change == "edited_receipt":
        scores = json.loads(items["auditkit_72/measured_72_report.json"])
        scores["overall"]["strict_accuracy_pct"] = 100000
        items["auditkit_72/measured_72_report.json"] = json.dumps(scores).encode()
    if change == "upgraded_human_claim":
        manifest = json.loads(items["OWNER_SCOPE_MANIFEST.json"])
        manifest["human_semantic_review_complete"] = True
        items["OWNER_SCOPE_MANIFEST.json"] = json.dumps(manifest).encode()
    if change == "forged_done_marker":
        marker = json.loads(items["auditkit_72/EVIDENCE_COMPLETE.json"])
        marker["original_model_report_sha256"] = "0" * 64
        items["auditkit_72/EVIDENCE_COMPLETE.json"] = json.dumps(marker).encode()
    if change == "uncompressed_override":
        # Declared entry sizes are enforced prior to reading ZIP member data.
        items["OWNER_SCOPE_MANIFEST.json"] = b"x" * 20001
    raw = _zip_bytes(items, extra=extra, duplicate=dup, symlink=symlink)
    destination = tmp_path / "original_owner.zip"
    destination.write_bytes(raw)
    with pytest.raises((ValueError, KeyError)):
        receiver.inspect_bundle(destination)
    assert destination.read_bytes() == raw
    assert not (tmp_path / "artifacts/actual_72_case_evaluation").exists()


@pytest.mark.parametrize("bad", [
    "{}", '{"status":"fake"}',
    '{"x":1,"x":2}',
    '{"status":NaN}',
])
def test_owner_manifest_must_have_exact_original_scope(bad):
    with pytest.raises(ValueError, match="manifest|scope"):
        receiver._metadata(bad.encode())


def test_unsafe_external_symlink_zip_is_rejected(tmp_path):
    target = tmp_path / "original.zip"
    real = tmp_path / "real.zip"
    real.write_bytes(b"not archive")
    target.symlink_to(real)
    with pytest.raises(ValueError, match="symlink"):
        receiver.inspect_bundle(target)


def test_cli_is_explicit_read_only_and_does_not_generate_evidence():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.verify_colab_72_bundle", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0
    assert "--archive" in result.stdout
