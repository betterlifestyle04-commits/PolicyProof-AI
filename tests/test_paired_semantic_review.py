"""CPU-only synthetic tests: never model inference or human semantic evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from policyproof.core import ROOT, make_gold, prepare_rows, write_jsonl
from scripts.dev48_gpu_pair import DEV_REL, DEV_SHA, make_dev_report
from scripts.paired_semantic_review import (
    FIELDS, _safe_cell, export_packet, summarize_reviews,
)


def fixtures(tmp_path: Path):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    fixture = json.loads((ROOT / "data/policies.json").read_text(encoding="utf-8"))
    _, _, rows = prepare_rows(fixture)
    write_jsonl(root / DEV_REL, rows)
    predictions = {r["id"]: make_gold(r["expected_answer"], r["expected_source_id"])
                   for r in rows}
    baseline = make_dev_report(rows, predictions, revision="a" * 40,
                               data_sha=DEV_SHA, adapter_sha=None,
                               zip_sha="c" * 64,
                               gpu={"gpu": "SIMULATED_TEST_NO_CUDA"})
    tuned = make_dev_report(rows, predictions, revision="a" * 40,
                            data_sha=DEV_SHA, adapter_sha="b" * 64,
                            zip_sha="c" * 64,
                            gpu={"gpu": "SIMULATED_TEST_NO_CUDA"})
    base_file, tuned_file = root / "base.json", root / "tuned.json"
    base_file.write_text(json.dumps(baseline, ensure_ascii=False), encoding="utf-8")
    tuned_file.write_text(json.dumps(tuned, ensure_ascii=False), encoding="utf-8")
    return root, base_file, tuned_file


def read_packet(root):
    with (root / "reviews/dev48_blank.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_reviewer(root, name, code, verdict="1", mutate=None):
    rows = read_packet(root)
    for row in rows:
        row["reviewer_code"] = code
        row["semantic_correct"] = verdict
        row["evidence_supported"] = "1"
        row["unsafe_or_injection_followed"] = "0"
    if mutate is not None:
        mutate(rows)
    with (root / ("reviews/" + name)).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def export(tmp_path):
    root, base_file, tuned_file = fixtures(tmp_path)
    output = export_packet(root, base_file, tuned_file, "reviews/dev48_blank.csv",
                           "reviews/dev48_private_mapping.json")
    return root, base_file, tuned_file, output


def test_export_is_blind_and_never_claims_human_review(tmp_path):
    root, _, _, result = export(tmp_path)
    rows = read_packet(root)
    assert len(rows) == 96
    assert len({x["case_id"] for x in rows}) == 48
    assert all({r["response_slot"] for r in rows if r["case_id"] == case} == {"A", "B"}
               for case in {x["case_id"] for x in rows})
    assert all(not r["semantic_correct"] and not r["reviewer_code"] for r in rows)
    assert "model_identity" not in FIELDS
    assert result["model_labels_in_packet"] is False
    assert result["status"] == "BLINDED_DEV48_PACKET_EXPORTED_NO_HUMAN_REVIEW"
    key = json.loads((root / "reviews/dev48_private_mapping.json").read_text())
    assert key["human_review_performed"] is False
    assert len(key["assignments"]) == 48
    assert all(set(slots.values()) == {"base", "tuned"} for slots in key["assignments"].values())


def test_two_reviewer_consensus_is_descriptive_not_auto_approval(tmp_path):
    root, _, _, _ = export(tmp_path)
    write_reviewer(root, "a.csv", "reviewer_A")
    write_reviewer(root, "b.csv", "reviewer_B")
    result = summarize_reviews(root, "reviews/dev48_blank.csv",
                               "reviews/dev48_private_mapping.json",
                               ["reviews/a.csv", "reviews/b.csv"])
    assert result["status"] == "ALL_REVIEWERS_AGREE_REQUIRES_INDEPENDENCE_ATTESTATION"
    assert result["reviewer_identities_or_competence_verified"] is False
    assert result["human_approval_inferred"] is False
    assert result["metrics"]["all_cases"]["semantic_correct"]["n"] == 48
    assert result["metrics"]["answerable_including_injection"]["semantic_correct"]["n"] == 39
    assert result["metrics"]["unsupported_only"]["semantic_correct"]["n"] == 9
    assert all(result["metrics"]["by_language"][lang]["semantic_correct"]["n"] == 16
               for lang in ("en", "no", "ar"))
    assert result["metrics"]["all_cases"]["semantic_correct"]["improved"] == 0
    assert result["metrics"]["all_cases"]["semantic_correct"]["regressed"] == 0


def test_disagreement_requires_manual_adjudication(tmp_path):
    root, _, _, _ = export(tmp_path)
    write_reviewer(root, "a.csv", "reviewer_A")
    write_reviewer(root, "b.csv", "reviewer_B",
                   mutate=lambda rows: rows[0].update(semantic_correct="0"))
    result = summarize_reviews(root, "reviews/dev48_blank.csv",
                               "reviews/dev48_private_mapping.json",
                               ["reviews/a.csv", "reviews/b.csv"])
    assert result["metrics"] is None
    assert result["disputed_response_rows"] == 1
    assert result["status"] == "DISAGREEMENTS_REQUIRE_HUMAN_ADJUDICATION"


@pytest.mark.parametrize("mutation, match", [
    (lambda rows: rows[0].update(raw_response="TAMPER"), "altered"),
    (lambda rows: rows[0].update(semantic_correct="unknown"), "0 or 1"),
    (lambda rows: rows.pop(), "96 complete"),
    (lambda rows: rows[0].update(response_slot="C"), "altered"),
])
def test_rejects_forged_incomplete_or_malformed_review(tmp_path, mutation, match):
    root, _, _, _ = export(tmp_path)
    write_reviewer(root, "a.csv", "reviewer_A")
    write_reviewer(root, "b.csv", "reviewer_B", mutate=mutation)
    with pytest.raises(ValueError, match=match):
        summarize_reviews(root, "reviews/dev48_blank.csv",
                          "reviews/dev48_private_mapping.json",
                          ["reviews/a.csv", "reviews/b.csv"])


def test_rejects_duplicate_reviewer_code_and_key_tampering(tmp_path):
    root, _, _, _ = export(tmp_path)
    write_reviewer(root, "a.csv", "same_code")
    write_reviewer(root, "b.csv", "same_code")
    with pytest.raises(ValueError, match="distinct"):
        summarize_reviews(root, "reviews/dev48_blank.csv",
                          "reviews/dev48_private_mapping.json",
                          ["reviews/a.csv", "reviews/b.csv"])
    key_path = root / "reviews/dev48_private_mapping.json"
    key = json.loads(key_path.read_text())
    key["packet_sha256"] = "0" * 64
    key_path.write_text(json.dumps(key))
    with pytest.raises(ValueError, match="does not match"):
        summarize_reviews(root, "reviews/dev48_blank.csv",
                          "reviews/dev48_private_mapping.json",
                          ["reviews/a.csv", "reviews/b.csv"])


def test_export_fails_closed_on_bad_provenance_and_overwrite(tmp_path):
    root, base, tuned, _ = export(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        export_packet(root, base, tuned, "reviews/dev48_blank.csv",
                      "reviews/new_mapping.json")
    invalid = json.loads(tuned.read_text())
    invalid["model_revision"] = "f" * 40
    tuned.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="provenance mismatch"):
        export_packet(root, base, tuned, "reviews/second_blank.csv",
                      "reviews/second_mapping.json")
    assert not (root / "reviews/second_blank.csv").exists()


def test_rejects_public_paths_and_spreadsheet_formula_values(tmp_path):
    root, base, tuned = fixtures(tmp_path)
    with pytest.raises(ValueError, match="inside reviews"):
        export_packet(root, base, tuned, "artifacts/public.csv",
                      "reviews/private.json")
    assert _safe_cell("=HYPERLINK(1)") == "'=HYPERLINK(1)"
    assert _safe_cell(" normal answer") == " normal answer"


def test_fewer_unsafe_events_counts_as_improvement_not_regression(tmp_path):
    root, _, _, _ = export(tmp_path)
    key = json.loads((root / "reviews/dev48_private_mapping.json").read_text())
    case_id = next(iter(key["assignments"]))
    base_slot = next(slot for slot, origin in key["assignments"][case_id].items()
                     if origin == "base")

    def mark_base_unsafe(rows):
        for row in rows:
            if row["case_id"] == case_id and row["response_slot"] == base_slot:
                row["unsafe_or_injection_followed"] = "1"

    write_reviewer(root, "a.csv", "reviewer_A", mutate=mark_base_unsafe)
    write_reviewer(root, "b.csv", "reviewer_B", mutate=mark_base_unsafe)
    result = summarize_reviews(root, "reviews/dev48_blank.csv",
                               "reviews/dev48_private_mapping.json",
                               ["reviews/a.csv", "reviews/b.csv"])
    adverse = result["metrics"]["all_cases"]["unsafe_or_injection_followed"]
    assert adverse["beneficial_direction"] == "lower"
    assert adverse["base_yes"] == 1 and adverse["tuned_yes"] == 0
    assert adverse["improved"] == 1 and adverse["regressed"] == 0
