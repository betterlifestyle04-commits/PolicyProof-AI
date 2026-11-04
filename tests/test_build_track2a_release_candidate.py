"""Synthetic temporary release-fixture tests — NOT evidence of a native toolkit run.

The real CI Track 2a job separately executes genuine CuratorKIT 0.1.22 and
AuditKIT 1.1.3, then runs the builder on those actual outputs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_STORED

import pytest

from policyproof.core import ROOT, make_gold, make_prompt, prepare_rows
from scripts.build_track2a_release_candidate import (
    EXPECTED_BASELINE_SHA, EXPECTED_CHALLENGE_SHA, EXPECTED_FILES,
    FILES, HASH_FILE, STATUS_FILE, gather, inspect_zip,
    validate_contents, write_bundle,
)
from scripts.generate_track2a_challenge import build_challenge
from scripts.track2a_chain import curator_alpaca_rows


def _write(path: Path, content: bytes | str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)


def _jsonlines(rows, *, sorted_keys=False):
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=sorted_keys) + "\n"
        for row in rows
    )


def original_fixture(tmp_path: Path):
    """Create temporary *fake* toolkit receipts for validating archive mechanics."""
    policies = json.loads((ROOT/"data/policies.json").read_text(encoding="utf-8"))
    _, _, original_rows = prepare_rows(policies)
    original_jsonl = _jsonlines(original_rows)
    assert hashlib.sha256(original_jsonl.encode()).hexdigest() == EXPECTED_BASELINE_SHA

    challenge_rows = build_challenge()
    challenge_jsonl = _jsonlines(challenge_rows, sorted_keys=True)
    assert hashlib.sha256(challenge_jsonl.encode()).hexdigest() == EXPECTED_CHALLENGE_SHA

    input48 = _jsonlines(curator_alpaca_rows(original_rows))
    manifest48 = '{"test_fixture_only":true}'
    output48 = input48
    output72 = _jsonlines([
        {"instruction":"BENCH_CASE_ID:"+r["id"]+"\n"+make_prompt(r),
         "input":"","output":make_gold(r["expected_answer"],r["expected_source_id"])}
        for r in challenge_rows
    ])
    c48sha = hashlib.sha256(output48.encode()).hexdigest()
    c72sha = hashlib.sha256(output72.encode()).hexdigest()

    fixture = {
        "evaluated_48/benchmark_48.jsonl": original_jsonl,
        "evaluated_48/original_saved_tiny_aya_base_model_predictions.json":
            (ROOT/"artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json").read_bytes(),
        "evaluated_48/curator_input.jsonl": input48,
        "evaluated_48/curator_export.jsonl": output48,
        "evaluated_48/curator_manifest.json": manifest48,
        "evaluated_48/curator_checksums.txt": c48sha + "  sft_alpaca.jsonl\n",
        "evaluated_48/curator_audit_chain_report.json": json.dumps({
            "status":"curatorkit_auditkit_chain_complete",
            "case_count":48,"dataset_sha256":EXPECTED_BASELINE_SHA,
            "curatorkit_version":"0.1.22", "auditkit_version":"1.1.3",
            "auditkit_fingerprint":"SYNTHETIC_RECEIPT_NOT_REAL_EVIDENCE",
            "human_labels_independently_reviewed":False,
            "curation":{
                "export_sha256":c48sha,
                "input_sha256":hashlib.sha256(input48.encode()).hexdigest(),
                "manifest_sha256":hashlib.sha256(manifest48.encode()).hexdigest(),
                "accepted":48,
                "label_and_case_id_integrity_verified":True,
            },
            "model":{"fine_tuned":False,"new_inference_performed":False},
            "test_fixture_only":True,
        }),
        "evaluated_48/auditkit_native_receipt.json": json.dumps({
            "status":"complete","sample_count":48,
            "dataset_sha256":EXPECTED_BASELINE_SHA,"auditkit_version":"1.1.3",
            "auditkit_fingerprint":"SYNTHETIC_RECEIPT_NOT_REAL_EVIDENCE",
            "model_inference_executed":False,
            "is_post_training_evaluation":False,
            "model_accuracy_semantically_verified":False,
            "human_review_completed":False,
            "test_fixture_only":True,
        }),
        "challenge_72/challenge_v1_UNSCORED.jsonl": challenge_jsonl,
        "challenge_72/challenge_manifest_UNSCORED.json": json.dumps({
            "status":"curated_unscored",
            "cases":72,"curated_cases":72,
            "dataset_sha256":EXPECTED_CHALLENGE_SHA,
            "curator_export_sha256":c72sha,
            "model_predictions_available":False,
            "independent_human_review":False,
            "evaluation_run":False,"training_data":False,
        }),
        "challenge_72/curator_export_UNSCORED.jsonl": output72,
        "README.md":(ROOT/"templates/huggingface/track2a_dataset_README.md").read_bytes(),
        "docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md":
            (ROOT/"docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md").read_bytes(),
    }
    for name, original_path in FILES.items():
        _write(tmp_path/original_path, fixture[name])
    return tmp_path


def test_read_only_gather_on_fake_temporary_toolkit_receipts(tmp_path):
    root=original_fixture(tmp_path)
    files=gather(root)
    assert set(files)==EXPECTED_FILES
    result=validate_contents(files)
    assert result["status"]=="LOCAL_RESEARCH_HANDOFF_VALIDATED_UNPUBLISHED"
    assert result["file_count"]==len(FILES)+2
    assert result["original_evaluated_cases"]==48
    assert result["new_unscored_cases"]==72
    for flag in ("new_gpu_inference_executed","huggingface_published",
                 "official_lexsi_submission_completed", "license_selected"):
        assert result[flag] is False
    assert not (tmp_path/"release.zip").exists()


def test_bundle_is_repeatable_byte_for_byte_for_identical_inputs(tmp_path):
    root=original_fixture(tmp_path)
    first=tmp_path/"first.zip"
    second=tmp_path/"second.zip"
    a=write_bundle(root,first)
    b=write_bundle(root,second)
    assert a["archive_sha256"]==b["archive_sha256"]
    assert first.read_bytes()==second.read_bytes()
    assert inspect_zip(first)["archive_sha256"]==a["archive_sha256"]
    with ZipFile(first) as z:
        assert set(z.namelist())==EXPECTED_FILES
        assert all(item.compress_type==ZIP_STORED for item in z.infolist())
        assert b"LOCAL_CANDIDATE_UNPUBLISHED" in z.read(STATUS_FILE)
        assert b"owner_approved_publication" in z.read(STATUS_FILE)
        assert b"license:" not in z.read("README.md").split(b"---",2)[1]
    with pytest.raises(FileExistsError):
        write_bundle(root,first)


@pytest.mark.parametrize("filename,replacement,match",[
    ("evaluated_48/curator_audit_chain_report.json",'{}',"chain"),
    ("evaluated_48/auditkit_native_receipt.json",'{}',"AuditKIT"),
    ("challenge_72/challenge_manifest_UNSCORED.json",'{}',"Challenge"),
])
def test_missing_or_fabricated_toolkit_receipts_fail(tmp_path,filename,replacement,match):
    root=original_fixture(tmp_path)
    (root/FILES[filename]).write_text(replacement)
    with pytest.raises(ValueError,match=match):
        validate_contents(gather(root))


def test_tampered_recomputed_original_scores_cannot_enter_handoff(tmp_path):
    root=original_fixture(tmp_path)
    path=root/FILES["evaluated_48/original_saved_tiny_aya_base_model_predictions.json"]
    report=json.loads(path.read_text())
    report["metrics"]["strict_accuracy_pct"]=99.0
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError,match="metric mismatch"):
        validate_contents(gather(root))


def test_inaccurate_72_unscored_flags_are_rejected(tmp_path):
    root=original_fixture(tmp_path)
    path=root/FILES["challenge_72/challenge_v1_UNSCORED.jsonl"]
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["model_evaluated"]=True
    path.write_text(_jsonlines(rows,sorted_keys=True))
    with pytest.raises(ValueError,match="challenge bytes differ"):
        validate_contents(gather(root))


def test_publication_license_or_fake_training_flag_cannot_be_injected(tmp_path):
    root=original_fixture(tmp_path)
    files=gather(root)
    status=json.loads(files[STATUS_FILE])
    status["trained_adapter_included"]=True
    files[STATUS_FILE]=(json.dumps(status)+"\n").encode()
    # Recompute hashes to simulate a stronger adversary.
    files[HASH_FILE]="".join(
        hashlib.sha256(v).hexdigest()+"  "+k+"\n"
        for k,v in sorted(files.items()) if k!=HASH_FILE
    ).encode()
    with pytest.raises(ValueError,match="Release status"):
        validate_contents(files)


def test_zip_file_corruption_and_duplicate_path_are_rejected(tmp_path):
    root=original_fixture(tmp_path)
    archive=tmp_path/"clean.zip"
    write_bundle(root,archive)
    forged=tmp_path/"forged.zip"
    with ZipFile(archive) as src, ZipFile(forged,"w") as out:
        for item in src.infolist():
            data=src.read(item.filename)
            if item.filename=="README.md":
                data=b"tampered without manifest update"
            out.writestr(item.filename,data)
    with pytest.raises(ValueError,match="SHA256"):
        inspect_zip(forged)
    extra=tmp_path/"traversal.zip"
    with ZipFile(archive) as src, ZipFile(extra,"w") as out:
        for item in src.infolist():
            out.writestr(item.filename,src.read(item.filename))
        out.writestr("../wrong.txt",b"oops")
    with pytest.raises(ValueError):
        inspect_zip(extra)


def test_missing_native_evidence_causes_fast_failure_without_output(tmp_path):
    output=tmp_path/"not-created.zip"
    with pytest.raises(ValueError,match="Missing"):
        write_bundle(tmp_path,output)
    assert not output.exists()



def test_historical_pinned_publisher_does_not_accept_new_candidate_without_review(tmp_path):
    from scripts.publish_track2a_dataset import validate_archive
    original_fixture(tmp_path)
    candidate = tmp_path / "new_unpublished.zip"
    write_bundle(tmp_path, candidate)
    with pytest.raises(ValueError, match="ZIP hash differs"):
        validate_archive(candidate)


def test_racing_release_creation_never_removes_another_existing_file(monkeypatch, tmp_path):
    from scripts import build_track2a_release_candidate as module
    original_fixture(tmp_path)
    target = tmp_path / "racing_bundle.zip"
    real_zip = module.ZipFile

    def raced_zip(path, mode):
        if mode == "x":
            Path(path).write_bytes(b"UNRELATED_FILE_CREATED_CONCURRENTLY")
            raise FileExistsError("Simulated concurrent writer")
        return real_zip(path, mode)

    monkeypatch.setattr(module, "ZipFile", raced_zip)
    with pytest.raises(FileExistsError):
        write_bundle(tmp_path, target)
    assert target.read_bytes() == b"UNRELATED_FILE_CREATED_CONCURRENTLY"
