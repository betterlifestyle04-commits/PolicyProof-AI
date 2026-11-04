"""Synthetic-only CPU tests for a future *real* dev48 paired GPU run.

No Tiny Aya weights, CUDA optimizer, real inference or science claims are
produced. Stubbed outputs test only preflight, provenance and write-once flow.
"""
from __future__ import annotations
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from policyproof.core import ROOT, make_gold, prepare_rows, sha256, write_jsonl
from scripts import dev48_gpu_pair as dev
from scripts.paired_run_recovery import write_evidence_once


def fake_root(tmp_path):
    root=tmp_path/"research"
    (root/"data").mkdir(parents=True)
    fixture=json.loads((ROOT/"data/policies.json").read_text(encoding="utf-8"))
    _train,_other,rows=prepare_rows(fixture)
    write_jsonl(root/dev.DEV_REL,rows)
    assert sha256(root/dev.DEV_REL)==dev.DEV_SHA
    return root,rows


def fake_training_zip(tmp_path):
    path=tmp_path/"fake_adapter_NOT_TRAINED.zip"
    # This is metadata-only, NOT a PEFT adapter or proof of model training.
    with ZipFile(path,"w") as z:
        z.writestr("run_info.json",json.dumps({
            "model_revision":"a"*40,
            "model":"CohereLabs/tiny-aya-global",
            "performance_improvement_verified":False,
        }))
    return path,dev.digest(path)


def fake_archive_check(monkeypatch):
    monkeypatch.setattr(dev,"inspect_training_archive",lambda path:{
        "archive_sha256":dev.digest(path),
        "backend":"manual_torch_extended_fallback",
        "optimizer_steps_self_reported":40,
        "expected_tiny_aya_lora_geometry":True,
        "file_hashes_match":True,
        "adapter_sha256":"c"*64,
        "base_revision_pinned":False,
    })


def test_preflight_checks_pinned_development_cases_and_source(tmp_path,monkeypatch):
    root,rows=fake_root(tmp_path)
    z,stamp=fake_training_zip(tmp_path)
    fake_archive_check(monkeypatch)
    result=dev.preflight(root,z,stamp)
    assert result["status"]=="DEV48_GPU_EVALUATION_PREPARED_NOT_EXECUTED"
    assert result["cases"]==48
    assert result["training_zip"]["model_revision_claimed"]=="a"*40
    assert result["historical_saved_base_allowed_as_pair"] is False
    assert result["model_inference_performed"] is False
    assert result["model_improvement_verified"] is False


def test_wrong_hash_stops_before_archive_deserialization(tmp_path,monkeypatch):
    root,rows=fake_root(tmp_path)
    z,stamp=fake_training_zip(tmp_path)
    fake_archive_check(monkeypatch)
    with pytest.raises(ValueError,match="pinned SHA256"):
        dev.preflight(root,z,"0"*64)
    with pytest.raises(ValueError,match="64-hex"):
        dev.preflight(root,z,"forged")


def test_mutated_development_dataset_never_permits_a_gpu_run(tmp_path,monkeypatch):
    root,rows=fake_root(tmp_path)
    z,stamp=fake_training_zip(tmp_path)
    fake_archive_check(monkeypatch)
    (root/dev.DEV_REL).write_bytes((root/dev.DEV_REL).read_bytes()+b"\n")
    with pytest.raises(ValueError,match="missing or changed"):
        dev.preflight(root,z,stamp)


def test_rejects_unverified_or_wrong_backends(tmp_path,monkeypatch):
    root,rows=fake_root(tmp_path)
    z,stamp=fake_training_zip(tmp_path)
    monkeypatch.setattr(dev,"inspect_training_archive",lambda path:{
        "archive_sha256":stamp, "backend":"aligntune_wrapper",
        "optimizer_steps_self_reported":10,
        "expected_tiny_aya_lora_geometry":False,
        "file_hashes_match":True,
    })
    with pytest.raises(ValueError,match="was not verified"):
        dev.preflight(root,z,stamp)


def make_outputs(rows):
    return {
        row["id"]:make_gold(row["expected_answer"],row["expected_source_id"])
        for row in rows
    }


def test_saved_dev_report_recomputes_exact_metrics_and_provenance(tmp_path):
    root,rows=fake_root(tmp_path)
    gpu={"gpu":"SIMULATED_TEST_ONLY_NOT_A_GPU"}
    outputs=make_outputs(rows)
    base=dev.make_dev_report(
        rows,outputs,revision="a"*40,data_sha=dev.DEV_SHA,
        adapter_sha=None,zip_sha="z"*64,gpu=gpu,
    )
    tuned=dev.make_dev_report(
        rows,outputs,revision="a"*40,data_sha=dev.DEV_SHA,
        adapter_sha="c"*64,zip_sha="d"*64,gpu=gpu,
    )
    b=root/"base.json"
    t=root/"tuned.json"
    write_evidence_once(b,base)
    write_evidence_once(t,tuned)
    dev.read_validated_report(
        b,rows,revision="a"*40,data_sha=dev.DEV_SHA,
        adapter_sha=None,zip_sha="d"*64,
    )
    dev.read_validated_report(
        t,rows,revision="a"*40,data_sha=dev.DEV_SHA,
        adapter_sha="c"*64,zip_sha="d"*64,
    )
    assert base["is_finetuned"] is False and tuned["is_finetuned"] is True
    assert base["split_is_blind_final_test"] is False
    assert base["dataset_sha256"]==dev.DEV_SHA
    assert base["metrics"]["samples"]==48


@pytest.mark.parametrize("mutation,pattern",[
    (lambda r:r.update(model_revision="b"*40),"provenance mismatch"),
    (lambda r:r.update(generation_config={"do_sample":True}),"provenance mismatch"),
    (lambda r:r["predictions"][0].update(raw="malicious change"),"recomputed scores"),
    (lambda r:r["predictions"][0].update(id="duplicate"),"case IDs differ"),
    (lambda r:r.update(human_semantic_review_completed=True),"provenance mismatch"),
])
def test_saved_report_rejects_fabricated_pairing_fields(tmp_path,mutation,pattern):
    root,rows=fake_root(tmp_path)
    report=dev.make_dev_report(
        rows,make_outputs(rows),revision="a"*40,data_sha=dev.DEV_SHA,
        adapter_sha="c"*64,zip_sha="d"*64,
        gpu={"gpu":"SIMULATED_TEST_ONLY_NOT_A_GPU"},
    )
    mutation(report)
    path=root/"record.json"
    path.write_text(json.dumps(report),encoding="utf-8")
    with pytest.raises(ValueError,match=pattern):
        dev.read_validated_report(path,rows,revision="a"*40,
            data_sha=dev.DEV_SHA,adapter_sha="c"*64,zip_sha="d"*64)


def test_incomplete_predictions_cannot_make_fake_real_report(tmp_path):
    root,rows=fake_root(tmp_path)
    incomplete=make_outputs(rows)
    del incomplete[rows[0]["id"]]
    with pytest.raises(ValueError,match="Incomplete"):
        dev.make_dev_report(rows,incomplete,revision="a"*40,
            data_sha=dev.DEV_SHA,adapter_sha=None,zip_sha="c"*64,gpu={})


def test_complete_gpu_flow_is_explicit_and_preserves_base_on_second_stage_failure(
    tmp_path,monkeypatch,
):
    root,rows=fake_root(tmp_path)
    z,stamp=fake_training_zip(tmp_path)
    fake_archive_check(monkeypatch)
    outputs=make_outputs(rows)
    called=[]
    def fake_generation(rows,revision,adapter_directory=None):
        called.append(adapter_directory is not None)
        if adapter_directory is not None:
            raise RuntimeError("Synthetic adapted-model loading failure")
        return outputs,{"gpu":"SIMULATED_TEST_NOT_REAL_GPU"}
    monkeypatch.setattr(dev,"_predict_on_gpu",fake_generation)
    monkeypatch.setattr(dev,"safely_stage_zip",lambda archive,dest:None)
    base=root/"saved_base.json";tuned=root/"saved_tuned.json";paired=root/"pair.json"
    with pytest.raises(RuntimeError,match="Synthetic"):
        dev.execute_pair(root,z,stamp,base,tuned,paired)
    assert called==[False,True]
    assert base.is_file()
    assert not tuned.exists() and not paired.exists()
    with pytest.raises(FileExistsError,match="Saved base exists"):
        dev.execute_pair(root,z,stamp,base,tuned,paired)
    # An explicit --resume-base uses fresh original provenance and only
    # computes the missing model side; never the historical unpinned report.
    def ok_generation(rows,revision,adapter_directory=None):
        assert adapter_directory is not None
        return outputs,{"gpu":"SIMULATED_TEST_NOT_REAL_GPU"}
    monkeypatch.setattr(dev,"_predict_on_gpu",ok_generation)
    monkeypatch.setattr(dev,"comparison",lambda base,tuned,rows,digest:{
        "status":"paired_deterministic_audit",
        "sample_count":len(rows),
        "limitations":["Synthetic comparison stub only"],
    })
    result=dev.execute_pair(root,z,stamp,base,tuned,paired,resume_base=True)
    assert result["sample_count"]==48
    assert result["is_final_blind_test"] is False
    assert result["not_comparable_to_historical_unpinned_48_base"] is True
    assert result["is_model_improvement_semantically_verified"] is False
    assert tuned.is_file() and paired.is_file()
    with pytest.raises(FileExistsError,match="Existing tuned or paired"):
        dev.execute_pair(root,z,stamp,base,tuned,paired,resume_base=True)


def write_bundle_fixture(root,rows):
    outputs=make_outputs(rows)
    base=dev.make_dev_report(
        rows,outputs,revision="a"*40,data_sha=dev.DEV_SHA,
        adapter_sha=None,zip_sha="d"*64,
        gpu={"gpu":"SIMULATED_TEST_NOT_REAL_GPU"},
    )
    tuned=dev.make_dev_report(
        rows,outputs,revision="a"*40,data_sha=dev.DEV_SHA,
        adapter_sha="c"*64,zip_sha="d"*64,
        gpu={"gpu":"SIMULATED_TEST_NOT_REAL_GPU"},
    )
    paired={
        "status":"paired_deterministic_audit",
        "sample_count":48,
        "evaluation_checksum":dev.DEV_SHA,
        "split":dev.SPLIT,
        "is_final_blind_test":False,
        "is_model_improvement_semantically_verified":False,
        "base_model_revision_loaded_on_inference":"a"*40,
        "adapter_sha256":"c"*64,
        "training_zip_sha256":"d"*64,
        "not_comparable_to_historical_unpinned_48_base":True,
    }
    paths=[root/"base.json",root/"tuned.json",root/"paired.json"]
    for file,record in zip(paths,(base,tuned,paired)):
        file.write_text(json.dumps(record),encoding="utf-8")
    return paths


def test_bundling_results_is_exclusive_and_checksums_original_files(tmp_path):
    root,rows=fake_root(tmp_path)
    paths=write_bundle_fixture(root,rows)
    dest=tmp_path/"dev48_pair.zip"
    result=dev.create_results_bundle(*paths,dest)
    assert result["files"]==4
    assert result["sha256"]==dev.digest(dest)
    with ZipFile(dest) as z:
        manifest=json.loads(z.read("BUNDLE_MANIFEST.json"))
        assert len(manifest["files_sha256"])==3
    assert manifest["final_blind_evaluation"] is False
    with pytest.raises(FileExistsError):
        dev.create_results_bundle(*paths,dest)


def test_forged_pair_metadata_cannot_be_packaged_as_proven_evaluation(tmp_path):
    root,rows=fake_root(tmp_path)
    paths=write_bundle_fixture(root,rows)
    paired=json.loads(paths[2].read_text())
    paired["is_final_blind_test"]=True
    paths[2].write_text(json.dumps(paired))
    with pytest.raises(ValueError,match="not a complete internally consistent"):
        dev.create_results_bundle(*paths,tmp_path/"do_not_create.zip")
    assert not (tmp_path/"do_not_create.zip").exists()
