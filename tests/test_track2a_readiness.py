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
    assert stage["status"]=="RESEARCH_READY_HF_PUBLICATION_BLOCKED"
    assert stage["ready_to_submit"] is False
    (docs/"hf_publication_manifest.json").write_text(json.dumps({
        "model_collection_url":"https://huggingface.co/collections/team/model-test-1",
        "dataset_collection_url":"https://huggingface.co/collections/team/data-test-1",
        "external_access_verified":True,"public_items_ungated_verified":False,
    }))
    assert readiness(tmp_path)["ready_to_submit"] is False


def test_abstract_is_under_500_words():
    path=ROOT/"docs/TRACK2A_SUBMISSION_ABSTRACT.md"
    assert len(path.read_text(encoding="utf-8").split())<500
