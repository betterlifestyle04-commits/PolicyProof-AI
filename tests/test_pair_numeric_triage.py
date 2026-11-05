"""The paired numeric audit is private and synthetic-test-only in CI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from policyproof.core import aggregate, prepare_rows, score
from scripts.pair_numeric_triage import make_pair_numeric_triage
from tests.test_export_review_from_bundle import make_fake_bundle
from policyproof.core import ROOT


def test_private_sidecar_checks_two_saved_reports_without_model_calls(tmp_path):
    root, archive, digest = make_fake_bundle(tmp_path)
    summary = make_pair_numeric_triage(root, archive, digest)
    result_path = root / "reviews/dev48_numeric_triage_owner_only.json"
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert summary["samples_per_model"] == 48
    assert summary["model_improvement_claim"] is False
    assert saved["research_scope"] == "already_inspected_synthetic_development_not_final"
    assert saved["do_not_share_with_reviewers"] is True
    assert saved["reviewer_blinding_required"] is True
    assert saved["semantic_model_improvement_verified"] is False
    assert saved["base"]["samples"] == 48
    assert saved["tuned"]["samples"] == 48
    assert saved["model_inference_performed"] is False
    assert saved["human_semantic_review_completed"] is False
    assert archive.exists()


def test_mismatched_zip_sha_fails_before_private_report(tmp_path):
    root, archive, digest = make_fake_bundle(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        make_pair_numeric_triage(root, archive, "0" * 64)
    assert not (root / "reviews/dev48_numeric_triage_owner_only.json").exists()


def test_owner_numeric_report_never_overwrites_existing(tmp_path):
    root, archive, digest = make_fake_bundle(tmp_path)
    make_pair_numeric_triage(root, archive, digest)
    output = root / "reviews/dev48_numeric_triage_owner_only.json"
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        make_pair_numeric_triage(root, archive, digest)
    assert output.read_bytes() == before


def test_newly_invented_amount_is_a_flag_not_semantic_model_accuracy(tmp_path):
    fixture = json.loads((ROOT / "data/policies.json").read_text(encoding="utf-8"))
    _, _, rows = prepare_rows(fixture)
    case = next(r for r in rows if r["id"] == "POL-U003-no-0")

    def change_model(base, tuned, paired):
        output = (
            '{"answer":"Det koster NOK 25 daglig, med minst NOK 100.",'
            '"source_id":"POL-U003","abstain":false}'
        )
        predictions = tuned["predictions"]
        item = next(p for p in predictions if p["id"] == case["id"])
        item["raw"] = output
        responses = {p["id"]: p["raw"] for p in predictions}
        tuned["scores"] = [score(r, responses[r["id"]]) for r in rows]
        tuned["metrics"] = aggregate(tuned["scores"])

    root, archive, digest = make_fake_bundle(tmp_path, edits=change_model)
    summary = make_pair_numeric_triage(root, archive, digest)
    data = json.loads((root / "reviews/dev48_numeric_triage_owner_only.json").read_text())
    flag = next(
        case for case in data["tuned"]["flagged_cases"]
        if case["id"] == "POL-U003-no-0"
    )
    assert flag["unmatched_literals"] == ["100"]
    assert summary["tuned_number_review_flags"] >= 1
    assert summary["newly_flagged"] >= 1
    assert data["semantic_model_improvement_verified"] is False
