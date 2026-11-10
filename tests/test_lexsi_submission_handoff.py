"""Offline handoff tests: source text and local verifier results, NOT form submission.

All real GPU evidence, actual reviewer identity, public HF access, and Lexsi
organizer form interactions are OUT OF SCOPE and explicitly not simulated.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from policyproof.core import ROOT, sha256
from scripts import lexsi_submission_handoff as module

ABSTRACT = ROOT / "docs/TRACK2A_SUBMISSION_ABSTRACT.md"


def fake_root(tmp_path, monkeypatch, *, public=False, measured72=False,
              scientific=True, conflict=False):
    docs = tmp_path / "docs"
    docs.mkdir()
    shutil.copy2(ABSTRACT, docs / ABSTRACT.name)
    def receipt(_):
        return {
            "status": (
                "SOURCE_EVIDENCE_CONFLICT_BLOCKED" if conflict else
                "READY_FOR_MANUAL_SUBMISSION_REVIEW" if public and scientific else
                "RESEARCH_READY_HF_PUBLICATION_BLOCKED" if scientific else
                "RESEARCH_INCOMPLETE"
            ),
            "ready_to_submit": public and scientific and not conflict,
            "checks": {
                "real_curatorkit_and_auditkit_chain": {"passed":scientific},
                "unscored_challenge_curated": {"passed":scientific},
                "public_research_writeup": {"passed":scientific},
                "public_ungated_hf_collections": {"passed":public},
            },
            "new_model_evaluation_claimed_for_72_cases": measured72,
            "saved_72_source_replay": {"cases":72 if measured72 else 0},
            "conflicting_optional_evidence_blocks_release":conflict,
        }
    monkeypatch.setattr(module, "readiness", receipt)
    return tmp_path


def test_official_submission_draft_has_valid_under_500_word_text():
    text, count, digest = module._abstract(ROOT)
    assert 50 <= count < 500
    assert count == len(text.split())
    assert "48 previously saved Tiny Aya" in text
    assert "No model was run on these new cases" in text
    assert digest == sha256(ABSTRACT)
    assert not text.startswith("# PolicyProof")
    assert module.OFFICIAL_FORM == "https://hackathon.lexsi.ai/submission/"


def test_no_public_collections_means_draft_only_no_fake_urls_email_or_submission(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch)
    packet = module.handoff(root)
    assert packet["status"] == "DRAFT_ONLY_MISSING_PUBLICATION_OR_RESEARCH"
    assert packet["eligible_for_manual_form_review_locally"] is False
    assert packet["form_fields"]["hf_model_collection"] is None
    assert packet["form_fields"]["hf_dataset_collection"] is None
    assert packet["form_fields"]["hf_dataset_repository"] is None
    assert packet["registered_email"] is None
    assert packet["form_fields"]["abstract_under_500_words"] is True
    assert packet["form_fields"]["toolkits_used_in_verified_chain"] == [
        "CuratorKIT", "AuditKIT",
    ]
    assert all(packet[name] is False for name in (
        "official_form_opened_by_script", "email_lookup_performed",
        "registered_email_verified", "owner_license_and_rights_approved_by_script",
        "official_submission_completed", "organizer_confirmation_verified",
        "external_network_used",
    ))
    assert "owner enters registration email" in " ".join(packet["required_actions"]).lower()
    assert not (tmp_path / "artifacts").exists()
    assert "model improvement" not in packet["status"].lower()


def test_unverified_owner_manifest_url_is_never_exposed(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch)
    (root / "docs/hf_publication_manifest.json").write_text(json.dumps({
        "model_collection_url":"https://huggingface.co/collections/unsafe/fake-model",
        "dataset_collection_url":"https://huggingface.co/collections/unsafe/fake-dataset",
        "external_access_verified":True,
        "official_submission_confirmed":True,
    }))
    output = module.handoff(root)
    assert output["form_fields"]["hf_model_collection"] is None
    assert output["form_fields"]["hf_dataset_collection"] is None
    assert output["official_submission_completed"] is False


def test_verified_public_hf_links_are_only_read_after_source_receipt(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch, public=True)
    monkeypatch.setattr(module, "validate_manifest", lambda p:{
        "model_collection_url": "https://huggingface.co/collections/owner/models",
        "dataset_collection_url": "https://huggingface.co/collections/owner/datasets",
        "dataset_url": "https://huggingface.co/datasets/owner/research",
    })
    report = module.handoff(root)
    # Existing abstract states public collections are still pending,
    # so it cannot be copied in a form once actual verified public repos exist.
    assert report["status"] == "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED"
    assert report["form_fields"]["abstract_text"] is None
    assert report["form_fields"]["hf_model_collection"].endswith("/models")
    assert report["form_fields"]["hf_dataset_collection"].endswith("/datasets")
    assert report["eligible_for_manual_form_review_locally"] is False
    assert report["official_submission_completed"] is False
    assert "Rewrite" in " ".join(report["required_actions"])


def test_complete_original_72_makes_old_unscored_claim_stale(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch, measured72=True)
    report = module.handoff(root)
    assert report["status"] == "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED"
    assert report["form_fields"]["abstract_text"] is None
    assert report["research_evidence"]["unscored_72_claim_must_not_be_pasted_if_new_results_exist"] is True
    assert report["research_evidence"]["original_72_model_evaluation_locally_source_replayed"] is True
    assert report["official_submission_completed"] is False


def test_source_evidence_conflict_blocks_handoff(tmp_path, monkeypatch):
    root = fake_root(tmp_path, monkeypatch, conflict=True)
    packet = module.handoff(root)
    assert packet["eligible_for_manual_form_review_locally"] is False
    assert any("invalid optional 72" in item for item in packet["required_actions"])


@pytest.mark.parametrize("mutation", [
    "oversized", "too_many_words", "new_unsupported_scores",
    "missing_historical_boundary", "html", "empty", "symlink",
])
def test_abstract_does_not_permit_fake_or_unbounded_form_claims(tmp_path,mutation):
    target = tmp_path / "docs" / ABSTRACT.name
    target.parent.mkdir()
    text = ABSTRACT.read_text(encoding="utf-8")
    if mutation == "oversized":
        text += " " + "x"*15000
    elif mutation == "too_many_words":
        text += " " + ("unsafe " * 500)
    elif mutation == "new_unsupported_scores":
        text = text.replace("33/48", "48/48")
    elif mutation == "missing_historical_boundary":
        text = text.replace(
            "No model was run on these new cases",
            "A Tiny Aya model was successfully trained and beat all gold answers",
        )
    elif mutation == "html":
        text += "<script>alert(1)</script>"
    elif mutation == "empty":
        text = ""
    elif mutation == "symlink":
        copy = tmp_path / "original.txt"
        copy.write_text(text)
        target.symlink_to(copy)
    if mutation != "symlink":
        target.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        module._abstract(tmp_path)


def test_cli_is_read_only_without_output_and_supports_strict_gate():
    result = subprocess.run(
        [sys.executable,"-m","scripts.lexsi_submission_handoff","--help"],
        cwd=ROOT,capture_output=True,text=True,timeout=20,
    )
    assert result.returncode == 0
    assert "--strict" in result.stdout and "--output" in result.stdout


def test_no_network_or_submission_or_email_access_in_handoff_source():
    original = (ROOT / "scripts/lexsi_submission_handoff.py").read_text()
    forbidden = [
        "requests.get(", "requests.post(", "httpx.", "urlopen(",
        "HfApi(", ".upload_file(", ".upload_folder(", "smtplib",
        "gmail", "notebook_login(", "trainer.train(", "issueNewTurn(",
    ]
    assert all(token not in original for token in forbidden)


def test_revised_72_case_abstract_allowed_only_after_actual_source_replay(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch, measured72=True)
    src = root / "docs/TRACK2A_SUBMISSION_ABSTRACT.md"
    text = src.read_text()
    text = text.replace(
        "a separate, unscored 72-case multilingual adversarial challenge",
        "a separate 72-case multilingual adversarial challenge",
    )
    text = text.replace(
        "No model was run on these new cases",
        "Saved original model outputs were source-replayed locally on CPU",
    )
    src.write_text(text)
    result = module.handoff(root)
    assert result["form_fields"]["abstract_text"] is not None
    assert "source-replayed" in result["form_fields"]["abstract_text"]
    assert result["research_evidence"]["original_72_model_evaluation_locally_source_replayed"] is True
    assert result["eligible_for_manual_form_review_locally"] is False
    assert result["official_submission_completed"] is False


def test_unreviewed_claim_of_72_source_replay_without_original_outputs_is_blocked(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch, measured72=False)
    src = root / "docs/TRACK2A_SUBMISSION_ABSTRACT.md"
    text = src.read_text().replace(
        "No model was run on these new cases",
        "Saved original model outputs were source-replayed locally on CPU",
    )
    src.write_text(text)
    result = module.handoff(root)
    assert result["status"] == "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED"
    assert result["form_fields"]["abstract_text"] is None
    assert result["research_evidence"]["new_72_model_results_cannot_be_claimed_without_source_replay"] is True


def test_new_owner_native40_result_blocks_historical_abstract_from_form_copy(tmp_path, monkeypatch):
    root = fake_root(tmp_path, monkeypatch, scientific=True)
    report = root / module.NEW_NATIVE40_REPORT
    report.write_text(
        "Private owner-supplied native AlignTune paired dev48 results: "
        "abstention 4/9 to 2/9, supported exact 0/39; no safety improvement."
    )
    packet = module.handoff(root)
    assert packet["status"] == "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED"
    assert packet["form_fields"]["abstract_text"] is None
    assert packet["research_evidence"]["native40_private_paired_dev48_report_present"] is True
    assert packet["research_evidence"]["historical_abstract_blocked_by_new_native40_results"] is True
    assert packet["research_evidence"]["native_aligntune_training_improvement_claimed"] is False
    assert packet["official_submission_completed"] is False


def test_new_owner_results_cannot_create_fictitious_public_model_collection(tmp_path, monkeypatch):
    root = fake_root(tmp_path, monkeypatch, scientific=True, public=False)
    report = root / module.NEW_NATIVE40_REPORT
    report.write_text("Only owner-supplied private native40 outputs, never public model weights")
    packet = module.handoff(root)
    assert packet["form_fields"]["hf_model_collection"] is None
    assert packet["form_fields"]["hf_dataset_collection"] is None
    assert packet["eligible_for_manual_form_review_locally"] is False
