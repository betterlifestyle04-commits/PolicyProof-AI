"""Offline-only CI metadata and original ZIP anti-forgery tests.

A successful fake GitHub run is a SYNTHETIC TEST FIXTURE. These tests do not
call external APIs, publish Hugging Face files, prove original GPU execution,
or authorize model rights.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from policyproof.core import ROOT, sha256
from scripts.verify_ci_hf_handoff import verify_owner_handoff, REPOSITORY, ARCHIVE
from tests.test_publish_track2a_candidate import candidate

SHA = "f" * 40


def original_run(run_id: int = 2345) -> dict:
    return {
        "id": run_id,
        "repository": {"full_name": REPOSITORY},
        "head_repository": {"full_name": REPOSITORY},
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_branch": "main",
        "status": "completed",
        "conclusion": "success",
        "head_sha": SHA,
    }


def handoff(tmp_path, *, layout: str = "plain"):
    archive, digest = candidate(tmp_path)
    root = tmp_path / "downloaded"
    root.mkdir()
    output = root / ARCHIVE if layout == "plain" else root / "artifacts" / ARCHIVE
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive, output)
    metadata = tmp_path / "run.json"
    metadata.write_text(json.dumps(original_run()), encoding="utf-8")
    return root, output, metadata, digest


@pytest.mark.parametrize("layout", ["plain", "artifacts"])
def test_verified_successful_main_research_zip_is_read_only(tmp_path, layout):
    root, zipfile, meta, digest = handoff(tmp_path, layout=layout)
    original = [meta.read_bytes(), zipfile.read_bytes()]
    report = verify_owner_handoff(
        meta, root, run_id="2345", expected_sha256=digest,
    )
    assert report["status"] == "VERIFIED_OWNER_SELECTED_MAIN_CI_ZIP_NO_PUBLICATION"
    assert report["ci_run_id"] == "2345"
    assert report["ci_head_sha"] == SHA
    assert report["approved_archive_sha256"] == digest
    assert report["files_strictly_verified"] == 15
    assert report["dataset_uploaded"] is False
    assert report["model_collection_created"] is False
    assert report["official_submission_completed"] is False
    assert report["original_gpu_inference_independently_authenticated"] is False
    assert [meta.read_bytes(), zipfile.read_bytes()] == original


def test_inspection_without_preapproved_sha_discovers_original_exact_archive(tmp_path):
    """An owner should see the SHA before deciding whether to publish."""
    root, source, metadata, digest = handoff(tmp_path)
    before = [source.read_bytes(), metadata.read_bytes()]
    discovered = verify_owner_handoff(metadata, root, run_id="2345")
    assert discovered["status"] == "INSPECTED_MAIN_CI_ZIP_UNAPPROVED_NO_PUBLICATION"
    assert discovered["measured_original_archive_sha256"] == digest
    assert discovered["approved_archive_sha256"] is None
    assert discovered["owner_approved_archive_digest_provided"] is False
    assert discovered["dataset_uploaded"] is False
    assert discovered["files_strictly_verified"] == 15
    assert [source.read_bytes(), metadata.read_bytes()] == before
    # A later separately pinned check can confirm the EXACT bytes.
    pinned = verify_owner_handoff(
        metadata, root, run_id="2345", expected_sha256=discovered[
            "measured_original_archive_sha256"
        ],
    )
    assert pinned["status"] == "VERIFIED_OWNER_SELECTED_MAIN_CI_ZIP_NO_PUBLICATION"
    assert pinned["owner_approved_archive_digest_provided"] is True
    assert pinned["approved_archive_sha256"] == digest


def test_cli_inspect_only_does_not_require_a_sha(tmp_path):
    root, source, metadata, digest = handoff(tmp_path)
    cli = subprocess.run(
        [sys.executable, "-m", "scripts.verify_ci_hf_handoff",
         "--metadata", str(metadata), "--download-dir", str(root),
         "--run-id", "2345"],
        cwd=ROOT, capture_output=True, text=True, timeout=25,
    )
    assert cli.returncode == 0, cli.stderr
    record = json.loads(cli.stdout)
    assert record["status"] == "INSPECTED_MAIN_CI_ZIP_UNAPPROVED_NO_PUBLICATION"
    assert record["measured_original_archive_sha256"] == digest
    assert record["owner_approved_archive_digest_provided"] is False


@pytest.mark.parametrize("invalid", ["", "abc", "0" * 40, "G" * 64])
def test_malformed_or_empty_explicit_pins_never_silently_turn_into_discovery(
    tmp_path, invalid,
):
    root, original, metadata, digest = handoff(tmp_path)
    with pytest.raises(ValueError, match="64-character"):
        verify_owner_handoff(
            metadata, root, run_id="2345", expected_sha256=invalid,
        )
    assert original.is_file()


def test_inspection_workflow_separates_unapproved_discovery_from_any_upload():
    workflow = (ROOT / ".github/workflows/owner-track2a-hf-dataset.yml").read_text()
    assert "required: false" in workflow
    assert "default: ''" in workflow
    assert 'if mode == "publish_public_dataset" and not digest' in workflow
    assert "first inspect and review the original" in workflow
    assert 'if [ -n "$OWNER_SHA256" ]; then' in workflow
    assert "handoff_args+=(--expected-sha256" in workflow
    assert "Retain safe inspection and measured SHA256" in workflow
    assert workflow.count("HF_TOKEN: ${{ secrets.HF_TOKEN }}") == 1
    assert "GITHUB_STEP_SUMMARY" in workflow
    assert "INSPECTION_ONLY" in workflow
    assert "cancel-in-progress: false" in workflow
    assert workflow.index("Verify exact owner-approved ZIP bytes") < workflow.index(
        "PUBLIC owner-approved dataset upload"
    )


@pytest.mark.parametrize("change", [
    {"head_branch": "feature"}, {"event": "pull_request"},
    {"status": "in_progress"}, {"conclusion": "failure"},
    {"path": ".github/workflows/forged.yml"},
    {"id": 2346},
    {"head_repository": {"full_name": "fork/PolicyProof-AI"}},
    {"repository": {"full_name": "forged/PolicyProof-AI"}},
    {"head_sha": "fake-or-short"},
])
def test_untrusted_ci_metadata_is_rejected_before_any_publication(tmp_path, change):
    root, original, meta, digest = handoff(tmp_path)
    run = original_run()
    run.update(change)
    meta.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError, match="successful|successful original|source must|Original source"):
        verify_owner_handoff(meta, root, run_id="2345", expected_sha256=digest)
    assert original.exists()


@pytest.mark.parametrize("run_id", ["", "0", "abc", "2345;echo bad", "-2345"])
def test_run_id_is_not_shell_injectable(tmp_path, run_id):
    root, _, meta, digest = handoff(tmp_path)
    with pytest.raises(ValueError, match="numeric"):
        verify_owner_handoff(meta, root, run_id=run_id, expected_sha256=digest)


def test_original_digest_mismatch_blocks_handoff(tmp_path):
    root, zipfile, meta, digest = handoff(tmp_path)
    with pytest.raises(ValueError, match="approved SHA256|approved SHA|source ZIP"):
        verify_owner_handoff(meta, root, run_id="2345", expected_sha256="0" * 64)
    assert zipfile.exists()


def test_ambiguous_duplicate_downloaded_zip_blocks_handoff(tmp_path):
    root, original, meta, digest = handoff(tmp_path)
    directory = root / "artifacts"
    directory.mkdir()
    shutil.copy2(original, directory / ARCHIVE)
    with pytest.raises(ValueError, match="exactly one"):
        verify_owner_handoff(meta, root, run_id="2345", expected_sha256=digest)


def test_symlinked_archive_is_refused(tmp_path):
    root, original, meta, digest = handoff(tmp_path)
    copy = tmp_path / "preserved_research.zip"
    shutil.copy2(original, copy)
    original.unlink()
    original.symlink_to(copy)
    with pytest.raises(ValueError, match="symlink"):
        verify_owner_handoff(meta, root, run_id="2345", expected_sha256=digest)
    assert copy.is_file()


def test_duplicate_run_metadata_fields_are_refused(tmp_path):
    root, original, meta, digest = handoff(tmp_path)
    meta.write_text('{"id":2345,"id":2346}', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed or ambiguous"):
        verify_owner_handoff(meta, root, run_id="2345", expected_sha256=digest)


def test_owner_action_stays_manual_and_confirms_rights_before_token():
    workflow = (ROOT / ".github/workflows/owner-track2a-hf-dataset.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "inspect_only" in workflow
    assert "publish_public_dataset" in workflow
    assert "confirm_rights" in workflow and "confirm_publication" in workflow
    assert workflow.count("HF_TOKEN: ${{ secrets.HF_TOKEN }}") == 1
    assert workflow.count("if: ${{ inputs.mode == 'publish_public_dataset' }}") == 2
    assert workflow.index("Verify exact owner-approved ZIP bytes") < workflow.index("PUBLIC owner-approved dataset upload")
    assert "--confirm-rights --confirm-publication" in workflow
    assert "actions: read" in workflow
    assert "persist-credentials: false" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "dataset_uploaded" not in workflow


def test_cli_inspects_synthetic_original_run_without_network(tmp_path):
    root, _, meta, digest = handoff(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.verify_ci_hf_handoff",
         "--metadata", str(meta), "--download-dir", str(root),
         "--run-id", "2345", "--expected-sha256", digest],
        cwd=ROOT, capture_output=True, text=True, timeout=25,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["status"] == "VERIFIED_OWNER_SELECTED_MAIN_CI_ZIP_NO_PUBLICATION"
