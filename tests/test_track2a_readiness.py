"""Alternative safety benchmark has a release gate distinct from model tuning."""
import json
from pathlib import Path
from scripts.track2a_readiness import readiness
from policyproof.core import ROOT


def test_default_repo_cannot_claim_track2a_submission():
    state=readiness(ROOT)
    assert state["ready_to_submit"] is False
    assert state["training_claimed"] is False
    assert state["new_model_evaluation_claimed_for_72_cases"] is False


def test_missing_evidence_is_not_research_complete(tmp_path):
    state=readiness(tmp_path)
    assert state["status"]=="RESEARCH_INCOMPLETE"
    assert not state["ready_to_submit"]


def test_actual_chain_and_challenge_require_public_hf_collections(tmp_path):
    a=tmp_path/"artifacts/track2a_safety"
    b=tmp_path/"artifacts/track2a_challenge_unscored"
    docs=tmp_path/"docs"
    for path in (a,b,docs): path.mkdir(parents=True)
    (a/"track2a_chain_report.json").write_text(json.dumps({
        "status":"curatorkit_auditkit_chain_complete",
        "case_count":48,
        "curatorkit_version":"0.1.22",
        "auditkit_version":"1.1.3",
        "human_labels_independently_reviewed":False,
        "model":{"new_inference_performed":False},
        "curation":{"label_and_case_id_integrity_verified":True},
    }))
    (b/"challenge_manifest.json").write_text(json.dumps({
        "status":"curated_unscored","cases":72,"curated_cases":72,
        "model_predictions_available":False,
        "independent_human_review":False,
    }))
    (docs/"TRACK2A_SUBMISSION_ABSTRACT.md").write_text("A transparent abstract")
    (docs/"TRACK2A_SAFETY_BENCHMARK_2026-11-03.md").write_text("A transparent report")
    stage=readiness(tmp_path)
    # Superficially plausible JSON flags WITHOUT their 48 saved predictions,
    # native toolkit receipts or original 72 curated files are NOT research.
    assert stage["status"]=="RESEARCH_INCOMPLETE"
    assert stage["ready_to_submit"] is False
    assert stage["checks"]["real_curatorkit_and_auditkit_chain"]["passed"] is False
    assert stage["checks"]["unscored_challenge_curated"]["passed"] is False
    (docs/"hf_publication_manifest.json").write_text(json.dumps({
        "model_collection_url":"https://huggingface.co/collections/team/model-test-1",
        "dataset_collection_url":"https://huggingface.co/collections/team/data-test-1",
        "external_access_verified":True,"public_items_ungated_verified":False,
    }))
    assert readiness(tmp_path)["ready_to_submit"] is False


def test_abstract_is_under_500_words():
    path=ROOT/"docs/TRACK2A_SUBMISSION_ABSTRACT.md"
    assert len(path.read_text(encoding="utf-8").split())<500



def test_owner_entered_verified_flags_without_live_receipt_are_not_publication_proof(tmp_path):
    """Manual manifest booleans never unlock the release gate."""
    path = tmp_path / "docs"
    path.mkdir(parents=True)
    (path / "hf_publication_manifest.json").write_text(json.dumps({
        "dataset_url":"https://huggingface.co/datasets/owner/repo",
        "model_collection_url":"https://huggingface.co/collections/owner/models",
        "dataset_collection_url":"https://huggingface.co/collections/owner/datasets",
        "external_access_verified":True,
        "public_items_ungated_verified":True,
    }))
    result = readiness(tmp_path)
    assert result["ready_to_submit"] is False
    assert result["checks"]["public_ungated_hf_collections"]["passed"] is False


def test_fake_toolkit_status_is_not_scientific_source_evidence(tmp_path):
    """Neither a forged success receipt nor a fabricated release is proof."""
    import json
    source = tmp_path / "artifacts/track2a_safety"
    challenge = tmp_path / "artifacts/track2a_challenge_unscored"
    source.mkdir(parents=True)
    challenge.mkdir(parents=True)
    (source / "track2a_chain_report.json").write_text(json.dumps({
        "status": "curatorkit_auditkit_chain_complete",
        "case_count": 48,
        "curatorkit_version": "0.1.22",
        "auditkit_version": "1.1.3",
        "human_labels_independently_reviewed": False,
        "model": {"new_inference_performed": False},
        "curation": {"label_and_case_id_integrity_verified": True},
    }))
    (challenge / "challenge_manifest.json").write_text(json.dumps({
        "status": "curated_unscored",
        "cases": 72, "curated_cases": 72,
        "model_predictions_available": False,
        "independent_human_review": False,
    }))
    state = readiness(tmp_path)
    assert state["status"] == "RESEARCH_INCOMPLETE"
    assert state["checks"]["real_curatorkit_and_auditkit_chain"]["passed"] is False
    assert state["checks"]["unscored_challenge_curated"]["passed"] is False


def test_track2a_readiness_is_read_only_and_handles_duplicate_status_json(tmp_path):
    import json
    from scripts.track2a_source_evidence import verify_unscored_challenge
    challenge = tmp_path / "artifacts/track2a_challenge_unscored"
    challenge.mkdir(parents=True)
    (challenge / "challenge_manifest.json").write_text(
        '{"status":"curated_unscored","status":"fake"}'
    )
    state = readiness(tmp_path)
    assert state["checks"]["unscored_challenge_curated"]["passed"] is False
    with __import__("pytest").raises(ValueError, match="Missing"):
        verify_unscored_challenge(tmp_path)


def test_readiness_with_original_repo_sources_not_yet_curated_is_not_publishable():
    report = readiness(ROOT)
    assert report["checks"]["public_ungated_hf_collections"]["passed"] is False
    assert report["official_submission_confirmed"] is False
