"""Submission gate must never equate CPU research checks to trained model quality."""
from pathlib import Path
import subprocess
import sys

import pytest

from policyproof.core import ROOT
from scripts.submission_gate import evaluate_readiness, write_new_report
from scripts.build_public_demo import verify_existing_gate_report


def test_current_worktree_not_ready_without_real_checkpoint():
    state=evaluate_readiness(ROOT)
    assert state["status"]=="NOT_READY"
    assert state["ready_for_submission"] is False
    assert state["checks"]["saved_baseline"]["passed"] is True
    assert state["checks"]["trained_adapter"]["passed"] is False
    assert state["checks"]["paired_finetuned_evaluation"]["passed"] is False
    assert state["checks"]["huggingface_artifacts"]["passed"] is False


def test_empty_worktree_is_not_ready(tmp_path):
    report=evaluate_readiness(tmp_path)
    assert report["ready_for_submission"] is False
    assert report["checks"]["saved_baseline"]["passed"] is False


def test_strict_command_fails_when_submission_not_ready():
    result=subprocess.run([sys.executable,"-m","scripts.submission_gate","--strict"],
        cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode==2
    assert '"status": "NOT_READY"' in result.stdout

def test_two_separate_hf_collections_required_even_if_urls_present(tmp_path):
    import json
    d=tmp_path/"docs"
    d.mkdir()
    manifest={
        "dataset_url":"https://huggingface.co/datasets/example/data",
        "adapter_url":"https://huggingface.co/example/lora",
        "logs_url":"https://huggingface.co/datasets/example/logs",
        "model_collection_url":"https://huggingface.co/collections/example/models-123",
        "dataset_collection_url":"https://huggingface.co/collections/example/data-123",
        "external_access_verified":True,
        "public_items_ungated_verified":False,
    }
    (d/"hf_publication_manifest.json").write_text(json.dumps(manifest))
    state=evaluate_readiness(tmp_path)
    assert state["checks"]["huggingface_artifacts"]["passed"] is False
    assert state["checks"]["public_hf_collections"]["passed"] is False
    manifest["public_items_ungated_verified"]=True
    (d/"hf_publication_manifest.json").write_text(json.dumps(manifest))
    state=evaluate_readiness(tmp_path)
    assert state["checks"]["huggingface_artifacts"]["passed"] is False
    assert state["checks"]["public_hf_collections"]["passed"] is False
    assert state["ready_for_submission"] is False
    assert state["owner_manifest_claims_are_not_proof"] is True


def test_abstract_is_less_than_500_words_and_does_not_claim_finished_training():
    draft=(ROOT/"docs/LEXSI_SUBMISSION_ABSTRACT.md").read_text(encoding="utf-8")
    body=draft.split("\n\n",2)[2].split("\n\n---",1)[0]
    assert len(body.split()) < 500
    assert "no model-training success" in body
    assert "48 outputs" in body



def test_fake_nonempty_weights_and_completed_metadata_fail_train_gate(tmp_path):
    """An arbitrary nonempty file plus 'completed' must not satisfy release checks."""
    import hashlib
    import json

    adapter = tmp_path / "artifacts" / "phony" / "adapter"
    adapter.mkdir(parents=True)
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"definitely not safetensors")
    config = adapter / "adapter_config.json"
    config.write_text('{"peft_type":"LORA","r":16}', encoding="utf-8")
    run = adapter.parent / "run_info.json"
    run.write_text(json.dumps({
        "status":"completed", "optimizer_steps":10,
        "model":"CohereLabs/tiny-aya-global", "examples":78,
        "dataset_sha256":"4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c",
        "trainer_backend":"aligntune_wrapper",
        "model_revision":"a"*40, "code_commit":"b"*40,
        "adapter_sha256": {
            weights.name: hashlib.sha256(weights.read_bytes()).hexdigest()
        },
        "adapter_config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),
    }))
    state=evaluate_readiness(tmp_path)
    assert state["checks"]["adapter_artifact_integrity"]["passed"] is False
    assert state["checks"]["trained_adapter"]["passed"] is False
    assert state["ready_for_submission"] is False


def test_readiness_report_is_created_exclusively_and_keeps_original(tmp_path):
    report_path = write_new_report('{"ready_for_submission":false}\\n',
                                   "artifacts/submission_gate_v2.json", tmp_path)
    original = report_path.read_bytes()
    assert report_path == tmp_path / "artifacts/submission_gate_v2.json"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_new_report("different report", "artifacts/submission_gate_v2.json",
                         tmp_path)
    assert report_path.read_bytes() == original


@pytest.mark.parametrize("unsafe_output", [
    "README.md",
    "docs/LEXSI_SUBMISSION_DRAFT.md",
    "reviews/human_votes.csv",
    "data/final_holdout_v1.jsonl",
    "artifacts/../docs/LEXSI_SUBMISSION_DRAFT.md",
    "artifacts/nested/report.json",
    "artifacts//report.json",
    "artifacts/.hidden.json",
    "artifacts/report.csv",
    "artifacts\\\\report.json",
    "/tmp/report.json",
    "artifacts/report:evil.json",
    "artifacts/Réport.json",
])
def test_readiness_output_cannot_escape_safe_report_directory(tmp_path, unsafe_output):
    with pytest.raises(ValueError, match="Report output"):
        write_new_report("no overwrite", unsafe_output, tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_readiness_report_refuses_existing_symlink_target(tmp_path):
    folder = tmp_path / "artifacts"
    folder.mkdir()
    protected = tmp_path / "protected.json"
    protected.write_text("KEEP")
    (folder / "link.json").symlink_to(protected)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_new_report("overwrite", "artifacts/link.json", tmp_path)
    assert protected.read_text() == "KEEP"


def test_readiness_report_refuses_symlinked_artifacts_directory(tmp_path):
    elsewhere = tmp_path / "protected_folder"
    elsewhere.mkdir()
    (tmp_path / "artifacts").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe report output directory"):
        write_new_report("overwrite", "artifacts/report.json", tmp_path)
    assert list(elsewhere.iterdir()) == []


def test_public_demo_reuses_identical_immutable_gate_snapshot(tmp_path):
    record = {"status": "NOT_READY", "ready_for_submission": False}
    path = write_new_report(
        '{"status":"NOT_READY","ready_for_submission":false}',
        "artifacts/submission_gate.json", tmp_path,
    )
    before = path.read_bytes()
    assert verify_existing_gate_report(path, record) == record
    assert path.read_bytes() == before


def test_public_demo_rejects_stale_or_tampered_gate_report_without_rewriting(tmp_path):
    path = write_new_report(
        '{"status":"NOT_READY","ready_for_submission":false}',
        "artifacts/submission_gate.json", tmp_path,
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="differs from current source evidence"):
        verify_existing_gate_report(
            path, {"status": "NOT_READY", "ready_for_submission": False,
                   "extra_evidence_check": "new"},
        )
    assert path.read_bytes() == before


def test_public_demo_rejects_duplicate_keys_in_existing_gate_snapshot(tmp_path):
    path = write_new_report(
        '{"status":"NOT_READY","status":"READY_FOR_MANUAL_REVIEW"}',
        "artifacts/submission_gate.json", tmp_path,
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        verify_existing_gate_report(path, {"status": "READY_FOR_MANUAL_REVIEW"})


def test_public_demo_rejects_existing_symlinked_gate_snapshot(tmp_path):
    elsewhere = tmp_path / "original.json"
    elsewhere.write_text('{"status":"NOT_READY"}')
    path = tmp_path / "artifacts" / "submission_gate.json"
    path.parent.mkdir()
    path.symlink_to(elsewhere)
    with pytest.raises(ValueError, match="bounded regular file"):
        verify_existing_gate_report(path, {"status": "NOT_READY"})
    assert elsewhere.read_text() == '{"status":"NOT_READY"}'


def _fake_baseline_audit_source(root, monkeypatch):
    """Only source-reader tests: no model inference or actual AuditKIT proof."""
    import json
    from policyproof.core import sha256
    from scripts import submission_gate as gate
    baseline = root / "artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text('{"model_revision":"' + "a" * 40 + '"}')
    dataset = root / "data/unseen_policy_test.jsonl"
    dataset.parent.mkdir()
    dataset.write_text("synthetic parser unit-test data")
    monkeypatch.setattr(gate, "verify_baseline", lambda *_: {"sample_count": 48})
    audit = root / "artifacts/auditkit_actual_baseline_replay.json"
    record = {
        "status": "complete",
        "sample_count": 48,
        "dataset_sha256": sha256(dataset),
        "model": "CohereLabs/tiny-aya-global",
        "model_revision": "a" * 40,
        "source": "previously_saved_real_model_predictions",
        "is_post_training_evaluation": False,
        "model_inference_executed": False,
        "model_accuracy_semantically_verified": False,
        "human_review_completed": False,
        "auditkit_version": "UNIT-TEST-ONLY",
        "auditkit_fingerprint": "SYNTHETIC-NOT-REAL",
    }
    audit.write_text(json.dumps(record))
    return audit, record


def test_release_gate_does_not_trust_duplicate_audit_status(tmp_path, monkeypatch):
    import json
    audit, record = _fake_baseline_audit_source(tmp_path, monkeypatch)
    assert evaluate_readiness(tmp_path)["checks"]["baseline_auditkit"]["passed"] is True
    audit.write_text(
        json.dumps(record).replace(
            '"status": "complete"',
            '"status":"complete","status":"interrupted"',
            1,
        )
    )
    result = evaluate_readiness(tmp_path)
    assert result["checks"]["saved_baseline"]["passed"] is True
    assert result["checks"]["baseline_auditkit"]["passed"] is False
    assert result["ready_for_submission"] is False


def test_release_gate_does_not_trust_symlinked_audit_receipt(tmp_path, monkeypatch):
    audit, _ = _fake_baseline_audit_source(tmp_path, monkeypatch)
    original = tmp_path / "outside.json"
    original.write_bytes(audit.read_bytes())
    audit.unlink()
    audit.symlink_to(original)
    report = evaluate_readiness(tmp_path)
    assert report["checks"]["saved_baseline"]["passed"] is True
    assert report["checks"]["baseline_auditkit"]["passed"] is False
    assert original.is_file()


def test_release_gate_does_not_trust_nonfinite_audit_receipt(tmp_path, monkeypatch):
    import json
    audit, record = _fake_baseline_audit_source(tmp_path, monkeypatch)
    audit.write_text(
        json.dumps(record).replace('"status": "complete"',
                                   '"status":"complete","bogus":Infinity', 1)
    )
    result = evaluate_readiness(tmp_path)
    assert result["checks"]["baseline_auditkit"]["passed"] is False
