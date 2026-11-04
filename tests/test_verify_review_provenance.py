"""Review evidence integrity checks use SIMULATED CSV votes only.

These temp fixtures are NOT actual human review, translation validation or
evidence of scientific correctness. They must never enter publication outputs.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.frozen_pair_inference import FINAL_REL, verify_review_signoff
from scripts.review_holdout import (
    REVIEW_FIELDS, consolidate_reviews, make_review_pack, write_review_csv,
)
from scripts.verify_review_provenance import verify_review_evidence


def prepared_private_review(tmp_path: Path):
    root = tmp_path / "working_copy"
    (root/"data").mkdir(parents=True)
    (root/"docs").mkdir()
    (root/"reviews").mkdir()
    shutil.copy2(ROOT/FINAL_REL, root/FINAL_REL)
    cases = read_jsonl(root/FINAL_REL)
    digest = sha256(root/FINAL_REL)
    paths, entries = [], []
    for code in ("reviewer_A", "reviewer_B"):
        rows = make_review_pack(cases, digest, code)
        for row in rows:
            for field in REVIEW_FIELDS:
                row[field] = "yes"
        filename = root/"reviews"/f"{code}.csv"
        write_review_csv(filename, rows)
        paths.append(filename)
        entries.append({
            "reviewer_code": code, "path": f"reviews/{code}.csv",
            "sha256": sha256(filename),
        })
    reconciliation = consolidate_reviews(paths, cases, digest)
    assert reconciliation["status"] == "READY_FOR_MANUAL_ADJUDICATION_AND_SIGNOFF"
    rec_path = root/"reviews/reconciliation.json"
    rec_path.write_text(json.dumps(reconciliation, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    signoff = {
        "review_complete": True,
        "dataset_file": FINAL_REL,
        "dataset_sha256": digest,
        "reviewers": ["reviewer_A","reviewer_B"],
        "review_evidence": {
            "schema": "policyproof_review_evidence_v1",
            "review_files": entries,
            "reconciliation": {
                "path": "reviews/reconciliation.json",
                "sha256": sha256(rec_path),
            },
            "approvals": [
                {
                    "reviewer_code":"reviewer_A",
                    "approved":True,
                    "independent_review_declared":True,
                    "blinded_to_model_predictions_declared":True,
                    "competent_languages":["en","no"],
                },
                {
                    "reviewer_code":"reviewer_B",
                    "approved":True,
                    "independent_review_declared":True,
                    "blinded_to_model_predictions_declared":True,
                    "competent_languages":["ar"],
                },
            ],
        },
    }
    signoff_path = root/"docs"/"final_benchmark_review_signoff.json"
    signoff_path.write_text(json.dumps(signoff, indent=2), encoding="utf-8")
    return root, cases, digest, signoff, signoff_path


def test_source_linked_synthetic_fixture_verifies_only_consistency(tmp_path):
    root, cases, digest, signoff, signoff_path = prepared_private_review(tmp_path)
    result = verify_review_signoff(signoff_path, digest, root=root)
    assert result["review_source_evidence_validated"] is True
    assert result["review_declared"] is True
    proof = result["review_evidence"]
    assert proof["source_review_files_verified"] == 2
    assert proof["case_count"] == 60
    assert proof["unanimous_yes_slots"] == 180
    assert proof["reviewer_csv_checksums_verified"] is True
    assert proof["independent_human_action_verified"] is False
    assert proof["reviewer_identity_independently_verified"] is False
    assert proof["gold_semantics_independently_verified"] is False
    assert proof["competence_independently_verified"] is False
    assert result["signoff_sha256"] == sha256(signoff_path)


def test_review_signoff_metadata_only_is_never_accepted(tmp_path):
    root, cases, digest, signoff, path = prepared_private_review(tmp_path)
    signoff.pop("review_evidence")
    path.write_text(json.dumps(signoff))
    with pytest.raises(ValueError, match="source evidence|review-source evidence"):
        verify_review_signoff(path, digest, root=root)


def test_reviewer_csv_mutation_after_signoff_fails_sha(tmp_path):
    root, cases, digest, signoff, path = prepared_private_review(tmp_path)
    reviewer = root/"reviews/reviewer_A.csv"
    reviewer.write_text(reviewer.read_text().replace(",yes,yes,yes,", ",no,yes,yes,", 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_review_signoff(path, digest, root=root)


def test_negative_vote_with_fresh_hashes_still_blocks_manual_signoff(tmp_path):
    root, cases, digest, signoff, path = prepared_private_review(tmp_path)
    reviewer = root/"reviews/reviewer_A.csv"
    with reviewer.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    rows[0]["evidence_support_yes_no"]="no"
    write_review_csv(reviewer, rows)
    signoff["review_evidence"]["review_files"][0]["sha256"] = sha256(reviewer)
    path.write_text(json.dumps(signoff))
    with pytest.raises(ValueError, match="Incomplete or negative review judgment"):
        verify_review_signoff(path, digest, root=root)


def test_blank_vote_with_fresh_checksum_still_blocks(tmp_path):
    root, cases, digest, signoff, path = prepared_private_review(tmp_path)
    reviewer = root/"reviews/reviewer_B.csv"
    with reviewer.open(newline="", encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    rows[0]["translation_faithful_yes_no"]=""
    write_review_csv(reviewer, rows)
    signoff["review_evidence"]["review_files"][1]["sha256"] = sha256(reviewer)
    path.write_text(json.dumps(signoff))
    with pytest.raises(ValueError, match="Incomplete or negative review"):
        verify_review_signoff(path, digest, root=root)


def test_reconciliation_forged_even_if_checksum_updated(tmp_path):
    root, cases, digest, signoff, path = prepared_private_review(tmp_path)
    report_file = root/"reviews/reconciliation.json"
    report=json.loads(report_file.read_text())
    report["status"]="READY_FOR_SUBMISSION"
    report_file.write_text(json.dumps(report))
    signoff["review_evidence"]["reconciliation"]["sha256"]=sha256(report_file)
    path.write_text(json.dumps(signoff))
    with pytest.raises(ValueError, match="Reconciliation differs"):
        verify_review_signoff(path, digest, root=root)


@pytest.mark.parametrize("mutation,pattern", [
    (lambda s: s["review_evidence"]["approvals"][0].update({"approved":False}),
     "Explicit reviewer approval"),
    (lambda s: s["review_evidence"]["approvals"][0].update(
        {"blinded_to_model_predictions_declared":False}), "Explicit reviewer approval"),
    (lambda s: s["review_evidence"]["approvals"][0].update(
        {"independent_review_declared":False}), "Explicit reviewer approval"),
    (lambda s: s["review_evidence"]["approvals"][0].update(
        {"competent_languages":["en"]}), "collectively declare"),
    (lambda s: s["review_evidence"]["approvals"][1].update(
        {"competent_languages":["fr"]}), "competently assess"),
    (lambda s: s["review_evidence"]["approvals"][1].update(
        {"reviewer_code":"reviewer_A"}), "unique source reviewer"),
    (lambda s: s.update({"reviewers":["reviewer_B","reviewer_A"]}), "order/codes"),
    (lambda s: s["review_evidence"].update({"schema":"forged-v2"}), "source evidence"),
])
def test_invalid_review_attestations_do_not_open_gate(tmp_path, mutation, pattern):
    root, cases, digest, signoff, path = prepared_private_review(tmp_path)
    mutation(signoff)
    path.write_text(json.dumps(signoff))
    with pytest.raises(ValueError, match=pattern):
        verify_review_signoff(path, digest, root=root)


@pytest.mark.parametrize("path_value", [
    "../reviewer_A.csv",
    "/etc/passwd",
    "reviews/../data/final_holdout_v1.jsonl",
    "reviews//reviewer_A.csv",
    "reviews\\reviewer_A.csv",
])
def test_untrusted_review_file_paths_rejected(tmp_path, path_value):
    root, cases, digest, signoff, path = prepared_private_review(tmp_path)
    signoff["review_evidence"]["review_files"][0]["path"] = path_value
    path.write_text(json.dumps(signoff))
    with pytest.raises(ValueError):
        verify_review_signoff(path, digest, root=root)


def test_symlinked_review_file_is_rejected(tmp_path):
    root, cases, digest, signoff, path = prepared_private_review(tmp_path)
    symlink=root/"reviews/link.csv"
    symlink.symlink_to(root/"reviews/reviewer_A.csv")
    signoff["review_evidence"]["review_files"][0]["path"]="reviews/link.csv"
    path.write_text(json.dumps(signoff))
    with pytest.raises(ValueError, match="Symlinked"):
        verify_review_signoff(path, digest, root=root)


def test_csv_gold_label_mutation_with_updated_sha_still_rejected(tmp_path):
    root, cases, digest, signoff, path = prepared_private_review(tmp_path)
    reviewer=root/"reviews/reviewer_A.csv"
    with reviewer.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    rows[0]["proposed_gold_answer"]="fabricated gold"
    write_review_csv(reviewer,rows)
    signoff["review_evidence"]["review_files"][0]["sha256"]=sha256(reviewer)
    path.write_text(json.dumps(signoff))
    with pytest.raises(ValueError, match="altered source"):
        verify_review_signoff(path,digest,root=root)


def test_dataset_mutation_after_review_rejected(tmp_path):
    root, cases, digest, signoff, path=prepared_private_review(tmp_path)
    dataset=root/FINAL_REL
    dataset.write_text(dataset.read_text()+"\n",encoding="utf-8")
    with pytest.raises(ValueError, match="missing or changed"):
        verify_review_signoff(path,digest,root=root)


def test_cli_defaults_to_cpu_only_blocked_no_evidence_written():
    command=[sys.executable,"-m","scripts.verify_review_provenance"]
    run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr
    assert '"status": "BLOCKED"' in run.stdout
    assert '"human_review_performed_by_this_command": false' in run.stdout
    assert '"model_inference_performed": false' in run.stdout


def test_synced_reconciliation_has_no_inference_output_or_gold_modification(tmp_path):
    root,cases,digest,signoff,path=prepared_private_review(tmp_path)
    report=verify_review_evidence(root,cases,digest,signoff)
    assert report["model_inference_performed"] is False
    assert sha256(root/FINAL_REL)==digest
    assert not list(root.rglob("*.safetensors"))
    assert not list(root.rglob("*predictions.json"))
