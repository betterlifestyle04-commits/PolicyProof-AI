"""Signoff draft tests are all simulated, NOT human review."""
from __future__ import annotations

import csv
import json

import pytest

from policyproof.core import sha256
from scripts.prepare_review_signoff import prepare_draft
from scripts.frozen_pair_inference import verify_review_signoff
from scripts.review_holdout import write_review_csv
from tests.test_verify_review_provenance import prepared_private_review


def test_completed_synthetic_packs_produce_only_unsigned_draft(tmp_path):
    root, cases, digest, signoff, signoff_path=prepared_private_review(tmp_path)
    untouched=sha256(root/"reviews/reviewer_A.csv")
    result=prepare_draft(
        root, ["reviews/reviewer_A.csv","reviews/reviewer_B.csv"],
        "reviews/reconciliation.json", "reviews/signoff_draft.json",
    )
    assert result["status"]=="MANUAL_SIGNOFF_DRAFT_ONLY_NO_APPROVAL"
    assert result["review_complete"] is False
    assert result["reviewer_approvals_created"] is False
    assert result["human_review_performed_by_this_command"] is False
    assert sha256(root/"reviews/reviewer_A.csv")==untouched
    draft_path=root/"reviews/signoff_draft.json"
    draft=json.loads(draft_path.read_text())
    assert draft["review_complete"] is False
    assert draft["dataset_sha256"]==digest
    assert [x["approved"] for x in draft["review_evidence"]["approvals"]]==[False,False]
    assert all(x["competent_languages"]==[] for x in draft["review_evidence"]["approvals"])
    assert draft["review_evidence"]["review_files"][0]["sha256"]==untouched
    # A draft cannot be used as a GPU signoff.
    with pytest.raises(ValueError,match="No completed human review declaration"):
        verify_review_signoff(draft_path,digest,root=root)
    # The original signoff was not overwritten.
    assert json.loads(signoff_path.read_text())["review_complete"] is True
    with pytest.raises(FileExistsError):
        prepare_draft(
            root, ["reviews/reviewer_A.csv","reviews/reviewer_B.csv"],
            "reviews/reconciliation.json", "reviews/signoff_draft.json",
        )


@pytest.mark.parametrize("output", [
    "../../draft.json",
    "docs/final_benchmark_review_signoff.json",
    "reviews/../docs/final_benchmark_review_signoff.json",
    "reviews//draft.json",
    "reviews/./draft.json",
])
def test_draft_cannot_write_outside_private_review_folder(tmp_path,output):
    root,*_=prepared_private_review(tmp_path)
    with pytest.raises((ValueError,FileExistsError)):
        prepare_draft(root,["reviews/reviewer_A.csv","reviews/reviewer_B.csv"],
                      "reviews/reconciliation.json",output)


def test_negative_vote_prevents_even_private_unsigned_draft(tmp_path):
    root,cases,digest,signoff,path=prepared_private_review(tmp_path)
    csv_file=root/"reviews/reviewer_A.csv"
    with csv_file.open(encoding="utf-8",newline="") as handle:
        rows=list(csv.DictReader(handle))
    rows[0]["question_unambiguous_yes_no"]="no"
    csv_file.unlink()  # Construct an intentionally modified test fixture.
    write_review_csv(csv_file,rows)
    with pytest.raises(ValueError,match="not ready for a manual draft"):
        prepare_draft(
            root,["reviews/reviewer_A.csv","reviews/reviewer_B.csv"],
            "reviews/reconciliation.json","reviews/signoff_draft.json",
        )
    assert not (root/"reviews/signoff_draft.json").exists()


def test_reconciliation_tampering_prevents_draft(tmp_path):
    root,cases,digest,signoff,path=prepared_private_review(tmp_path)
    rec=root/"reviews/reconciliation.json"
    report=json.loads(rec.read_text())
    report["reviewer_count"]=999
    rec.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="reconciliation differs"):
        prepare_draft(
            root,["reviews/reviewer_A.csv","reviews/reviewer_B.csv"],
            "reviews/reconciliation.json","reviews/signoff_draft.json",
        )


def test_no_draft_from_missing_real_reviewer_files(tmp_path):
    root,*_=prepared_private_review(tmp_path)
    (root/"reviews/reviewer_B.csv").unlink()
    with pytest.raises(ValueError,match="missing"):
        prepare_draft(root,
                      ["reviews/reviewer_A.csv","reviews/reviewer_B.csv"],
                      "reviews/reconciliation.json","reviews/signoff_draft.json")
