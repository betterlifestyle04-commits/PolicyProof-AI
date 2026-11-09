"""Offline Lexsi owner-form HTML checks. No network, email, GPU or submission."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re

import pytest

from scripts import lexsi_submission_handoff as source
from scripts.lexsi_submission_offline_html import (
    OUTPUT_REL, render_owner_form_html, write_offline_html,
)
from tests.test_lexsi_submission_handoff import fake_root


def test_source_checked_unsubmitted_form_is_readable_and_contains_no_secrets(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch)
    packet = source.handoff(root)
    html = render_owner_form_html(packet)
    assert html.startswith("<!doctype html>")
    assert "BLOCKED / DRAFT ONLY" in html
    assert "PolicyProof AI" in html
    assert "33/48" in html and "20/39" in html
    assert "Hugging Face model collection" in html
    assert "Hugging Face dataset collection" in html
    assert "MISSING — DO NOT INVENT A LINK" in html
    assert "2026-11-14 end-of-day AoE" in html
    assert "https://hackathon.lexsi.ai/submission/" in html
    assert packet["form_fields"]["abstract_source_sha256"] in html
    assert "registered_email" not in html
    assert "<script" not in html and "onclick" not in html
    assert "fetch(" not in html and "localStorage" not in html
    assert "<form" not in html
    assert "form-action &#39;none&#39;" in html
    assert "No token, registration email, private reviewer judgment" in html


def test_html_escapes_owner_controlled_text_and_disables_markup_execution(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch)
    packet = source.handoff(root)
    forged = deepcopy(packet)
    forged["form_fields"]["project_title"] = '"><svg onload=alert(1)>'
    forged["required_actions"].append('</textarea><script>alert("bad")</script>')
    html = render_owner_form_html(forged)
    assert "<svg" not in html and "<script" not in html
    assert "&lt;svg" in html
    assert "&lt;script&gt;" in html
    assert "NOT an organizer receipt" in html


def test_stale_abstract_is_withheld_instead_of_being_copy_ready(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch, measured72=True)
    packet = source.handoff(root)
    assert packet["form_fields"]["abstract_text"] is None
    html = render_owner_form_html(packet)
    assert "historical abstract is stale" in html
    assert "No model was run on these new cases" not in html
    assert "BLOCKED / DRAFT ONLY" in html


def test_public_url_only_when_existing_handoff_has_anonymous_readback(
    tmp_path, monkeypatch
):
    root = fake_root(tmp_path, monkeypatch)
    packet = source.handoff(root)
    fake_url = "https://huggingface.co/collections/fake-owner/fake-proof"
    packet["form_fields"]["hf_model_collection"] = fake_url
    with pytest.raises(ValueError, match="unverified HF"):
        render_owner_form_html(packet)

    # A genuine source-checked form handoff may display supplied public links
    # only after its separate receipt verifier has already accepted them.
    folder = tmp_path / "public"
    folder.mkdir()
    root = fake_root(folder, monkeypatch, public=True)
    monkeypatch.setattr(source, "validate_manifest", lambda path: {
        "model_collection_url": "https://huggingface.co/collections/owner/models",
        "dataset_collection_url": "https://huggingface.co/collections/owner/datasets",
        "dataset_url": "https://huggingface.co/datasets/owner/policyproof",
    })
    measured = source.handoff(root)
    assert measured["status"] == "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED"
    html = render_owner_form_html(measured)
    assert "https://huggingface.co/collections/owner/models" in html
    assert "historical abstract is stale" in html
    assert "LOCAL EVIDENCE READY" not in html


@pytest.mark.parametrize("change", [
    "pretend_submission", "private_email", "stale_with_abstract",
    "wrong_schema", "unverified_hf", "wrong_count",
    "missing_owner_actions", "contradictory_status", "numeric_hf_flag",
])
def test_forged_local_handoff_is_rejected(tmp_path, monkeypatch, change):
    root = fake_root(tmp_path, monkeypatch)
    packet = deepcopy(source.handoff(root))
    if change == "pretend_submission":
        packet["official_submission_completed"] = True
    elif change == "private_email":
        packet["registered_email"] = "private@example.invalid"
    elif change == "stale_with_abstract":
        packet["status"] = "DRAFT_ONLY_ABSTRACT_STALE_BLOCKED"
    elif change == "wrong_schema":
        packet["schema"] = "attacker_v1"
    elif change == "unverified_hf":
        packet["form_fields"]["hf_dataset_collection"] = (
            "https://huggingface.co/collections/fake/datasets"
        )
    elif change == "wrong_count":
        packet["form_fields"]["abstract_word_count"] = 1000
    elif change == "missing_owner_actions":
        packet["required_actions"] = [""] * 2
    elif change == "contradictory_status":
        packet["eligible_for_manual_form_review_locally"] = True
    elif change == "numeric_hf_flag":
        packet["form_fields"]["hf_anonymous_access_receipt_current"] = 1
    with pytest.raises(ValueError):
        render_owner_form_html(packet)


def test_create_once_offline_file_without_modifying_sources(tmp_path, monkeypatch):
    root = fake_root(tmp_path, monkeypatch)
    evidence = root / "docs/TRACK2A_SUBMISSION_ABSTRACT.md"
    original = evidence.read_bytes()
    created = write_offline_html(root)
    assert created == root / OUTPUT_REL
    before = created.read_bytes()
    assert original == evidence.read_bytes()
    assert created.stat().st_size < 100_000
    with pytest.raises(ValueError, match="overwrite"):
        write_offline_html(root)
    assert created.read_bytes() == before


def test_output_symlinks_rejected_without_writing_elsewhere(tmp_path, monkeypatch):
    root = fake_root(tmp_path, monkeypatch)
    dest = tmp_path / "outside"
    dest.mkdir()
    (tmp_path / "artifacts").symlink_to(dest, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        write_offline_html(root)
    assert not list(dest.iterdir())


def test_optional_owner_html_is_read_only_and_has_no_external_client():
    path = Path(__file__).resolve().parents[1] / "scripts/lexsi_submission_offline_html.py"
    raw = path.read_text()
    for forbidden in (
        "requests.post(", "httpx.", "urlopen(", ".submit(", "HfApi(",
        "send_email(", "smtplib", "navigator.clipboard", "document.cookie",
        "setTimeout(", "localStorage", "fetch(", "iframe",
    ):
        assert forbidden not in raw
    assert "with target.open(\"x\"" in raw
    assert re.search(r"OUTPUT_REL = Path\(\"artifacts/", raw)
