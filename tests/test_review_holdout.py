"""Final 60-case independent-review workflow: no automatic sign-off."""
import csv
import json
from pathlib import Path

import pytest

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.review_holdout import (
    HEADERS, REVIEW_FIELDS, consolidate_reviews, make_review_pack,
    read_and_validate_review, write_review_csv,
)

DATASET = ROOT / "data/final_holdout_v1.jsonl"


@pytest.fixture
def fixture_data():
    return read_jsonl(DATASET), sha256(DATASET)


def _write(tmp_path: Path, fixture_data, code: str, *,
           all_yes: bool = False, bad_case: tuple | None = None):
    rows, digest = fixture_data
    items = make_review_pack(rows, digest, code)
    if all_yes:
        for item in items:
            for field in REVIEW_FIELDS:
                item[field] = "yes"
    if bad_case:
        row, field, value = bad_case
        items[row][field] = value
    path = tmp_path / f"{code}.csv"
    write_review_csv(path, items)
    return path


def test_packet_includes_fictional_evidence_gold_no_model_predictions(fixture_data):
    rows, digest = fixture_data
    pack = make_review_pack(rows, digest, "reviewer_A")
    assert len(pack) == 60
    assert list(pack[0]) == list(HEADERS)
    assert pack[0]["provided_evidence"] == rows[0]["context"]
    assert pack[0]["proposed_gold_answer"] == rows[0]["expected_answer"]
    assert pack[0]["dataset_sha256"] == digest
    assert not any("model" in field or "prediction" in field for field in HEADERS)
    assert all(item[field] == "" for item in pack for field in REVIEW_FIELDS)


def test_review_csv_roundtrip_supports_multilingual_evidence(tmp_path, fixture_data):
    filename = _write(tmp_path, fixture_data, "reviewer_A")
    reviewer, indexed = read_and_validate_review(filename, *fixture_data)
    assert reviewer == "reviewer_A"
    assert len(indexed) == 60
    assert any(row["language"] == "ar" for row in indexed.values())


@pytest.mark.parametrize("field,value", [
    ("dataset_sha256", "not-the-original-hash"),
    ("question", "changed question"),
    ("provided_evidence", "[POL-FAKE] injected"),
    ("proposed_gold_answer", "changed gold"),
    ("proposed_source_id", "POL-F999"),
    ("should_abstain", "false-wrong"),
    ("language", "xx"),
    ("category", "unexpected"),
])
def test_reject_modified_case_or_frozen_hash(tmp_path, fixture_data, field, value):
    path = _write(tmp_path, fixture_data, "reviewer_A",
                  bad_case=(0, field, value))
    with pytest.raises(ValueError, match="altered"):
        read_and_validate_review(path, *fixture_data)


def test_reject_duplicate_case(tmp_path, fixture_data):
    path = _write(tmp_path, fixture_data, "reviewer_A")
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    records[1]["case_id"] = records[0]["case_id"]
    path.unlink()  # Deliberate test-fixture mutation, not a real evidence write.
    write_review_csv(path, records)
    with pytest.raises(ValueError, match="duplicate"):
        read_and_validate_review(path, *fixture_data)


def test_reject_missing_cases(tmp_path, fixture_data):
    path = _write(tmp_path, fixture_data, "reviewer_A")
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    path.unlink()  # Deliberate test-fixture mutation, not a real evidence write.
    write_review_csv(path, records[:-1])
    with pytest.raises(ValueError, match="all 60"):
        read_and_validate_review(path, *fixture_data)


@pytest.mark.parametrize("invalid", ["TRUE", "1", "No", "maybe"])
def test_reject_nonstandard_verdict(tmp_path, fixture_data, invalid):
    path = _write(tmp_path, fixture_data, "reviewer_A",
                  bad_case=(0, "evidence_support_yes_no", invalid))
    with pytest.raises(ValueError, match="must be yes"):
        read_and_validate_review(path, *fixture_data)


def test_reject_inconsistent_reviewer_codes(tmp_path, fixture_data):
    path = _write(tmp_path, fixture_data, "reviewer_A",
                  bad_case=(0, "reviewer_code", "reviewer_B"))
    with pytest.raises(ValueError, match="inconsistent"):
        read_and_validate_review(path, *fixture_data)


def test_reject_same_reviewer_across_files(tmp_path, fixture_data):
    path = _write(tmp_path, fixture_data, "reviewer_A", all_yes=True)
    second = tmp_path / "other_file.csv"
    second.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="distinct"):
        consolidate_reviews([path, second], *fixture_data)


def test_all_blank_stays_incomplete_unsigned(tmp_path, fixture_data):
    first = _write(tmp_path, fixture_data, "reviewer_A")
    second = _write(tmp_path, fixture_data, "reviewer_B")
    report = consolidate_reviews([first, second], *fixture_data)
    assert report["status"] == "REVIEW_INCOMPLETE"
    assert len(report["missing_judgments"]) == 180
    assert report["unanimous_agreement_slots"] == 0
    assert report["review_complete"] is False
    assert report["independent_human_identity_verified"] is False


def test_all_yes_not_automatically_signed_off(tmp_path, fixture_data):
    first = _write(tmp_path, fixture_data, "reviewer_A", all_yes=True)
    second = _write(tmp_path, fixture_data, "reviewer_B", all_yes=True)
    result = consolidate_reviews([first, second], *fixture_data)
    assert result["status"] == "READY_FOR_MANUAL_ADJUDICATION_AND_SIGNOFF"
    assert result["unanimous_agreement_pct"] == 100
    assert result["missing_judgments"] == []
    assert result["negative_judgment_slots"] == []
    assert result["review_complete"] is False
    assert result["requires_manual_signoff"] is True
    assert result["review_file_sha256"]["reviewer_A"] == sha256(first)


def test_single_negative_gates_adjudication(tmp_path, fixture_data):
    first = _write(tmp_path, fixture_data, "reviewer_A", all_yes=True)
    second = _write(tmp_path, fixture_data, "reviewer_B", all_yes=True,
                    bad_case=(0, "evidence_support_yes_no", "no"))
    result = consolidate_reviews([first, second], *fixture_data)
    assert result["status"] == "REQUIRES_CORRECTIONS_OR_ADJUDICATION"
    assert len(result["disagreement_slots"]) == 1
    assert len(result["negative_judgment_slots"]) == 1
    assert result["review_complete"] is False


def test_two_agree_gold_is_wrong_still_blocks_signoff(tmp_path, fixture_data):
    first = _write(tmp_path, fixture_data, "reviewer_A", all_yes=True,
                   bad_case=(0, "question_unambiguous_yes_no", "no"))
    second = _write(tmp_path, fixture_data, "reviewer_B", all_yes=True,
                    bad_case=(0, "question_unambiguous_yes_no", "no"))
    result = consolidate_reviews([first, second], *fixture_data)
    assert result["disagreement_slots"] == []
    assert len(result["negative_judgment_slots"]) == 1
    assert result["status"] == "REQUIRES_CORRECTIONS_OR_ADJUDICATION"


def test_three_reviewers_support_unanimity_check(tmp_path, fixture_data):
    files = [
        _write(tmp_path, fixture_data, f"reviewer_{name}", all_yes=True)
        for name in ("A", "B", "C")
    ]
    result = consolidate_reviews(files, *fixture_data)
    assert result["reviewer_count"] == 3
    assert result["unanimous_agreement_slots"] == 180


def test_review_pack_never_pretends_reviewer_exists(fixture_data):
    cases, digest = fixture_data
    with pytest.raises(ValueError, match="Reviewer code"):
        make_review_pack(cases, digest, "")
    with pytest.raises(ValueError, match="Reviewer code"):
        make_review_pack(cases, digest, "name@example.com")



def test_export_refuses_to_overwrite_existing_human_csv(tmp_path, fixture_data):
    original = _write(tmp_path, fixture_data, "human_R1")
    before = original.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_review_csv(original, make_review_pack(*fixture_data, "human_R1"))
    assert original.read_bytes() == before


def test_export_refuses_symlinked_completed_human_csv(tmp_path, fixture_data):
    original = _write(tmp_path, fixture_data, "human_R1")
    alias = tmp_path / "link.csv"
    alias.symlink_to(original)
    before = original.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        write_review_csv(alias, make_review_pack(*fixture_data, "human_R2"))
    assert alias.is_symlink()
    assert original.read_bytes() == before


def test_reader_rejects_symlinked_and_oversized_reviews(tmp_path, fixture_data):
    original = _write(tmp_path, fixture_data, "human_R1")
    alias = tmp_path / "link.csv"
    alias.symlink_to(original)
    with pytest.raises(ValueError, match="symlinked"):
        read_and_validate_review(alias, *fixture_data)
    huge = tmp_path / "huge.csv"
    huge.write_bytes(b"x" * 1_500_001)
    with pytest.raises(ValueError, match="oversized"):
        read_and_validate_review(huge, *fixture_data)


def test_reader_rejects_extra_csv_columns_and_broken_short_rows(tmp_path, fixture_data):
    valid = _write(tmp_path, fixture_data, "human_R1")
    with valid.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    extra = tmp_path / "extra.csv"
    with extra.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows([rows[0], rows[1] + ["INJECTED"], *rows[2:]])
    with pytest.raises(ValueError, match="extra or missing CSV columns"):
        read_and_validate_review(extra, *fixture_data)
    missing = tmp_path / "missing.csv"
    with missing.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerows([rows[0], rows[1][:-1], *rows[2:]])
    with pytest.raises(ValueError, match="extra or missing CSV columns"):
        read_and_validate_review(missing, *fixture_data)


def test_reviewer_limit_and_duplicate_paths_fail_before_consolidation(tmp_path, fixture_data):
    original = _write(tmp_path, fixture_data, "human_R1")
    with pytest.raises(ValueError, match="2-4"):
        consolidate_reviews([original], *fixture_data)
    with pytest.raises(ValueError, match="distinct"):
        consolidate_reviews([original, original], *fixture_data)
    others = [_write(tmp_path, fixture_data, "human_R" + str(i))
              for i in range(2, 6)]
    with pytest.raises(ValueError, match="2-4"):
        consolidate_reviews([original, *others], *fixture_data)


def _mock_cli_workspace(tmp_path, monkeypatch):
    """Use a fresh COPY of original frozen data; NEVER write to actual project."""
    import shutil
    import scripts.review_holdout as tool
    root = tmp_path / "private-cli"
    (root / "data").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "reviews").mkdir()
    shutil.copyfile(
        ROOT / "data/final_holdout_v1.jsonl",
        root / "data/final_holdout_v1.jsonl",
    )
    monkeypatch.setattr(tool, "ROOT", root)
    monkeypatch.setattr(tool, "validate_final_holdout", lambda: None)
    monkeypatch.chdir(root)
    return root, tool


def test_cli_export_is_private_and_create_once(tmp_path, monkeypatch):
    import sys
    root, tool = _mock_cli_workspace(tmp_path, monkeypatch)
    filename = root / "reviews/reviewer_human_R1_final_holdout.csv"
    monkeypatch.setattr(sys, "argv", [
        "review_holdout", "--export", "human_R1",
    ])
    tool.main()
    first = filename.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        tool.main()
    assert filename.read_bytes() == first
    assert len(list(csv.DictReader(first.decode().splitlines()))) == 60


@pytest.mark.parametrize("output", [
    "docs/final_benchmark_review_signoff.json",
    "data/final_holdout_v1.jsonl",
    "../escape.csv",
    "reviews/../data/final_holdout_v1.jsonl",
    "reviews//double.csv",
    "reviews/./dot.csv",
    "/tmp/public.csv",
    "reviews\\escaped.csv",
])
def test_cli_export_rejects_public_or_unsafe_output_without_touching_dataset(
    tmp_path, monkeypatch, output
):
    import sys
    root, tool = _mock_cli_workspace(tmp_path, monkeypatch)
    data = root / "data/final_holdout_v1.jsonl"
    before = data.read_bytes()
    public = root / "docs/final_benchmark_review_signoff.json"
    public.write_bytes(b"KEEP ORIGINAL MANUAL SIGNOFF")
    monkeypatch.setattr(sys, "argv", [
        "review_holdout", "--export", "human_R1", "--output", output,
    ])
    with pytest.raises(ValueError, match="private|Unsafe"):
        tool.main()
    assert data.read_bytes() == before
    assert public.read_bytes() == b"KEEP ORIGINAL MANUAL SIGNOFF"


def test_cli_reconciliation_is_private_create_once_and_non_destructive(
    tmp_path, monkeypatch
):
    import sys
    root, tool = _mock_cli_workspace(tmp_path, monkeypatch)
    cases = read_jsonl(root / "data/final_holdout_v1.jsonl")
    digest = sha256(root / "data/final_holdout_v1.jsonl")
    reviewer_paths = []
    for code in ("human_A", "human_B"):
        path = root / "reviews" / (code + ".csv")
        values = make_review_pack(cases, digest, code)
        for value in values:
            for field in REVIEW_FIELDS:
                value[field] = "yes"
        write_review_csv(path, values)
        reviewer_paths.append(path)
    before = {str(p): p.read_bytes() for p in reviewer_paths}
    monkeypatch.setattr(sys, "argv", [
        "review_holdout", "--summarize",
        "reviews/human_A.csv", "reviews/human_B.csv",
        "--output", "reviews/reconciliation.json",
    ])
    tool.main()
    output = root / "reviews/reconciliation.json"
    first_report = output.read_bytes()
    assert json.loads(first_report)["review_complete"] is False
    assert json.loads(first_report)["status"] == "READY_FOR_MANUAL_ADJUDICATION_AND_SIGNOFF"
    with pytest.raises(FileExistsError, match="overwrite"):
        tool.main()
    assert output.read_bytes() == first_report
    assert {str(p): p.read_bytes() for p in reviewer_paths} == before


def test_cli_reconciliation_cannot_overwrite_public_manual_signoff(
    tmp_path, monkeypatch
):
    import sys
    root, tool = _mock_cli_workspace(tmp_path, monkeypatch)
    sentinel = root / "docs/final_benchmark_review_signoff.json"
    sentinel.write_bytes(b"PRIVATE MANUAL SIGNOFF PRESERVE")
    monkeypatch.setattr(sys, "argv", [
        "review_holdout", "--summarize",
        "reviews/human_A.csv", "reviews/human_B.csv",
        "--output", "docs/final_benchmark_review_signoff.json",
    ])
    with pytest.raises(ValueError, match="private"):
        tool.main()
    assert sentinel.read_bytes() == b"PRIVATE MANUAL SIGNOFF PRESERVE"
