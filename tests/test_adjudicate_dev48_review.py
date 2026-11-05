"""CPU-only synthetic human adjudication tests; NEVER simulate real research votes."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from policyproof.core import sha256
from scripts.adjudicate_dev48_review import (
    ADJ_SCHEMA, export_disputes, resolve_disputes,
)
from scripts.paired_semantic_review import FIELDS, summarize_reviews
from tests.test_paired_semantic_review import export, read_packet, write_reviewer


def setup_with_disputes(tmp_path: Path):
    root, _, _, _ = export(tmp_path)
    write_reviewer(root, "judge_a.csv", "reviewer_A")
    def b_disagrees(rows):
        rows[0]["semantic_correct"] = "0"
        rows[1]["evidence_supported"] = "0"
    write_reviewer(root, "judge_b.csv", "reviewer_B", mutate=b_disagrees)
    names = ["reviews/judge_a.csv", "reviews/judge_b.csv"]
    result = summarize_reviews(
        root, "reviews/dev48_blank.csv",
        "reviews/dev48_private_mapping.json", names,
    )
    assert result["disputed_response_rows"] == 2
    assert result["metrics"] is None
    return root, names


def _fill_third(root, *, reviewer_code="reviewer_C", modify=None):
    blank = root / "reviews/dev48_disputes_blank.csv"
    with blank.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.update({
            "reviewer_code": reviewer_code,
            "semantic_correct": "0",
            "evidence_supported": "0",
            # Adjudicator can disagree on an UNDISPUTED field here; this is
            # deliberately ignored during conflict-only reconciliation.
            "unsafe_or_injection_followed": "1",
            "reviewer_notes": "synthetic test placeholder, not a human vote",
        })
    if modify:
        modify(rows)
    out = root / "reviews/dev48_adjudicator_completed.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return out


def test_export_masks_only_disputes_without_review_votes_or_real_case_ids(tmp_path):
    root, reviewer_files = setup_with_disputes(tmp_path)
    original_hashes = {name: sha256(root / name) for name in reviewer_files}
    outcome = export_disputes(root, reviewer_files=reviewer_files)
    assert outcome["rows_needing_independent_adjudication"] == 2
    assert outcome["human_adjudication_completed"] is False
    assert outcome["original_votes_shared_with_adjudicator"] is False
    with (root / "reviews/dev48_disputes_blank.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["category"] for row in rows} == {"review_case"}
    assert all(row["case_id"].startswith("CASE-") for row in rows)
    assert all(row["reviewer_code"] == "" for row in rows)
    assert all(all(row[field] == "" for field in (
        "semantic_correct", "evidence_supported", "unsafe_or_injection_followed"
    )) for row in rows)
    assert all("UNKNOWN" not in row["case_id"] and "INJECT" not in row["case_id"]
               for row in rows)
    manifest = json.loads(
        (root / "reviews/dev48_disputes_private_manifest.json").read_text()
    )
    assert manifest["schema"] == ADJ_SCHEMA
    assert manifest["response_rows"] == 2
    assert manifest["human_adjudication_completed"] is False
    assert manifest["first_reviewers_sha256"] == original_hashes
    assert all(sha256(root / name) == stamp for name, stamp in original_hashes.items())
    with pytest.raises(FileExistsError, match="overwrite"):
        export_disputes(root, reviewer_files=reviewer_files)


def test_resolves_only_disputed_field_values_and_preserves_original_votes(tmp_path):
    root, names = setup_with_disputes(tmp_path)
    export_disputes(root, reviewer_files=names)
    orig_hashes = {name: sha256(root / name) for name in names}
    third = _fill_third(root)
    report = resolve_disputes(
        root, reviewer_files=names, adjudicator="reviews/" + third.name,
    )
    assert report["paired_development_cases"] == 48
    assert report["response_rows"] == 96
    assert report["disputed_response_rows"] == 2
    assert report["adjudicated_verdict_cells"] == 2
    assert report["original_unanimous_verdicts_preserved"] is True
    assert report["original_review_files_sha256"] == orig_hashes
    assert report["adjudicator_identity_or_independence_verified"] is False
    assert report["semantically_validated_model_improvement"] is False
    assert report["human_approval_inferred"] is False
    assert report["final_blind_benchmark"] is False
    assert report["metrics"]["all_cases"]["semantic_correct"]["n"] == 48
    assert report["metrics"]["all_cases"]["semantic_correct"]["base_yes"] + (
        report["metrics"]["all_cases"]["semantic_correct"]["tuned_yes"]
    ) == 95  # one disputed semantic verdict adjudicated as 0
    assert report["metrics"]["all_cases"]["evidence_supported"]["base_yes"] + (
        report["metrics"]["all_cases"]["evidence_supported"]["tuned_yes"]
    ) == 95
    assert report["metrics"]["all_cases"]["unsafe_or_injection_followed"]["base_yes"] + (
        report["metrics"]["all_cases"]["unsafe_or_injection_followed"]["tuned_yes"]
    ) == 0  # third reviewer unsafe=1 on undisputed fields is not an override
    assert report["metrics"]["unsupported_only"]["semantic_correct"]["n"] == 9
    assert report["metrics"]["answerable_including_injection"]["semantic_correct"]["n"] == 39
    assert {lang: report["metrics"]["by_language"][lang]["semantic_correct"]["n"]
            for lang in ("en", "no", "ar")} == {"en": 16, "no": 16, "ar": 16}
    assert all(sha256(root / name) == stamp for name, stamp in orig_hashes.items())
    # Double check the original review does not change into fictional
    # unanimous consensus after exporting/resolving real human disagreements.
    fresh = summarize_reviews(
        root, "reviews/dev48_blank.csv", "reviews/dev48_private_mapping.json",
        names,
    )
    assert fresh["status"] == "DISAGREEMENTS_REQUIRE_HUMAN_ADJUDICATION"
    assert fresh["metrics"] is None


def test_without_disagreements_cannot_create_adjudication_packet(tmp_path):
    root, _, _, _ = export(tmp_path)
    write_reviewer(root, "judge_a.csv", "reviewer_A")
    write_reviewer(root, "judge_b.csv", "reviewer_B")
    with pytest.raises(ValueError, match="No disputed human judgments"):
        export_disputes(root, reviewer_files=[
            "reviews/judge_a.csv", "reviews/judge_b.csv",
        ])
    assert not (root / "reviews/dev48_disputes_blank.csv").exists()


@pytest.mark.parametrize("mutate,pattern", [
    (lambda rows: rows.pop(), "complete every disputed response"),
    (lambda rows: rows[0].update(semantic_correct=""), "judgments must be filled"),
    (lambda rows: rows[0].update(raw_response="FAKE ANSWER"), "changed, duplicated or omitted"),
    (lambda rows: rows[0].update(category="unseen_unanswerable"), "changed, duplicated or omitted"),
    (lambda rows: rows[0].update(response_slot="X"), "changed, duplicated or omitted"),
    (lambda rows: rows[0].update(reviewer_code="reviewer_B"), "one valid reviewer code"),
])
def test_rejects_incomplete_or_tampered_adjudicator_csv(tmp_path, mutate, pattern):
    root, names = setup_with_disputes(tmp_path)
    export_disputes(root, reviewer_files=names)
    third = _fill_third(root, modify=mutate)
    with pytest.raises(ValueError, match=pattern):
        resolve_disputes(
            root, reviewer_files=names,
            adjudicator="reviews/" + third.name,
        )


def test_rejects_adjudicator_using_an_original_reviewer_code(tmp_path):
    root, names = setup_with_disputes(tmp_path)
    export_disputes(root, reviewer_files=names)
    third = _fill_third(root, reviewer_code="reviewer_A")
    with pytest.raises(ValueError, match="must differ from original reviewer codes"):
        resolve_disputes(
            root, reviewer_files=names,
            adjudicator="reviews/" + third.name,
        )


def test_rejects_changed_original_votes_after_packet_export(tmp_path):
    root, names = setup_with_disputes(tmp_path)
    export_disputes(root, reviewer_files=names)
    third = _fill_third(root)
    filename = root / names[0]
    content = filename.read_bytes()
    # A harmless-looking added note is still a mutation of locked votes.
    filename.write_bytes(content + b"\n")
    with pytest.raises(ValueError, match="does not match sealed original review evidence"):
        resolve_disputes(
            root, reviewer_files=names,
            adjudicator="reviews/" + third.name,
        )


@pytest.mark.parametrize("field,value", [
    ("original_votes_in_packet", True),
    ("model_origin_in_packet", True),
    ("human_adjudication_completed", True),
    ("source_packet_sha256", "0" * 64),
    ("adjudication_packet_sha256", "0" * 64),
])
def test_rejects_forged_private_adjudication_manifest(tmp_path, field, value):
    root, names = setup_with_disputes(tmp_path)
    export_disputes(root, reviewer_files=names)
    third = _fill_third(root)
    path = root / "reviews/dev48_disputes_private_manifest.json"
    receipt = json.loads(path.read_text())
    receipt[field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="does not match sealed original review evidence"):
        resolve_disputes(
            root, reviewer_files=names,
            adjudicator="reviews/" + third.name,
        )


def test_rejects_modified_blinded_dispute_packet(tmp_path):
    root, names = setup_with_disputes(tmp_path)
    export_disputes(root, reviewer_files=names)
    third = _fill_third(root)
    path = root / "reviews/dev48_disputes_blank.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="does not match sealed original review evidence"):
        resolve_disputes(
            root, reviewer_files=names,
            adjudicator="reviews/" + third.name,
        )


def test_rejects_public_output_paths_and_review_file_path_reuse(tmp_path):
    root, names = setup_with_disputes(tmp_path)
    with pytest.raises(ValueError, match="Private evidence path"):
        export_disputes(
            root, reviewer_files=names,
            blank="artifacts/leaky-disputes.csv",
        )
    with pytest.raises(FileExistsError, match="overwrite"):
        export_disputes(
            root, reviewer_files=names,
            blank=names[0],
        )
    assert not (root / "artifacts/leaky-disputes.csv").exists()
