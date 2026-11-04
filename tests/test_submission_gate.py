"""Submission gate must never equate CPU research checks to trained model quality."""
from pathlib import Path
import subprocess
import sys

from policyproof.core import ROOT
from scripts.submission_gate import evaluate_readiness


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
    assert state["checks"]["huggingface_artifacts"]["passed"] is True
    assert state["checks"]["public_hf_collections"]["passed"] is False
    manifest["public_items_ungated_verified"]=True
    (d/"hf_publication_manifest.json").write_text(json.dumps(manifest))
    state=evaluate_readiness(tmp_path)
    assert state["checks"]["public_hf_collections"]["passed"] is True
    assert state["ready_for_submission"] is False


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
