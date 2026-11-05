"""CPU-only bundle handoff tests using SIMULATED reports, never real inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pytest

from policyproof.core import ROOT, make_gold, prepare_rows, sha256, write_jsonl
from scripts.dev48_gpu_pair import DEV_REL, DEV_SHA, make_dev_report
from scripts.export_review_from_bundle import (
    ORIGINAL_DEV48_ZIP_SHA256, export_from_bundle, verify_bundle,
)


def make_fake_bundle(tmp_path: Path, *, edits=None, zip_entries=None):
    root = tmp_path / "test-repo"
    (root / "data").mkdir(parents=True)
    policies = json.loads((ROOT / "data/policies.json").read_text(encoding="utf-8"))
    _, _, rows = prepare_rows(policies)
    write_jsonl(root / DEV_REL, rows)
    responses = {r["id"]: make_gold(r["expected_answer"], r["expected_source_id"])
                 for r in rows}
    base = make_dev_report(rows, responses, revision="a" * 40,
                           data_sha=DEV_SHA, adapter_sha=None, zip_sha="c" * 64,
                           gpu={"gpu": "SIMULATED_NO_HARDWARE"})
    tuned = make_dev_report(rows, responses, revision="a" * 40,
                            data_sha=DEV_SHA, adapter_sha="b" * 64, zip_sha="c" * 64,
                            gpu={"gpu": "SIMULATED_NO_HARDWARE"})
    paired = {
        "status": "paired_deterministic_audit",
        "sample_count": 48,
        "evaluation_checksum": DEV_SHA,
        "is_final_blind_test": False,
        "is_model_improvement_semantically_verified": False,
        "base_model_revision_loaded_on_inference": "a" * 40,
        "adapter_sha256": "b" * 64,
        "training_zip_sha256": "c" * 64,
    }
    if edits:
        edits(base, tuned, paired)
    payloads = {
        "dev48_base.json": json.dumps(base, ensure_ascii=False).encode(),
        "dev48_tuned.json": json.dumps(tuned, ensure_ascii=False).encode(),
        "dev48_paired_auditkit.json": json.dumps(paired, ensure_ascii=False).encode(),
    }
    manifest = {
        "schema": "policyproof_dev48_gpu_pair_bundle_v1",
        "split": "already_inspected_synthetic_development_only",
        "final_blind_evaluation": False,
        "model_improvement_semantically_verified": False,
        "files_sha256": {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()},
    }
    payloads["BUNDLE_MANIFEST.json"] = json.dumps(manifest).encode()
    if zip_entries is not None:
        payloads = zip_entries(payloads)
    archive = tmp_path / "simulated_not_real_gpu.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as z:
        for name, content in payloads.items():
            z.writestr(name, content)
    return root, archive, sha256(archive)


def test_verify_and_export_complete_simulated_bundle(tmp_path):
    root, archive, digest = make_fake_bundle(tmp_path)
    payloads, verify = verify_bundle(archive, digest)
    assert len(payloads) == 4
    assert verify["members_verified"] == 4
    assert verify["independent_gpu_rerun_performed"] is False
    assert verify["human_semantic_review_performed"] is False
    result = export_from_bundle(root, archive, digest)
    assert result["cases"] == 48
    assert result["response_rows"] == 96
    assert result["model_labels_in_packet"] is False
    key = json.loads((root / "reviews/dev48_private_mapping.json").read_text())
    assert key["human_review_performed"] is False
    assert key["base_report_sha256"] == hashlib.sha256(payloads["dev48_base.json"]).hexdigest()
    assert key["tuned_report_sha256"] == hashlib.sha256(payloads["dev48_tuned.json"]).hexdigest()
    assert archive.exists()


def test_original_audited_hash_is_pinned_not_placeholder():
    assert ORIGINAL_DEV48_ZIP_SHA256 == (
        "e98b06482985a9c2f5dfa62100dccda7421830de8292bae013250ccf4ca1099c"
    )


def test_wrong_original_hash_fails_before_creating_private_files(tmp_path):
    root, archive, digest = make_fake_bundle(tmp_path)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_from_bundle(root, archive, "0" * 64)
    assert not (root / "reviews").exists()


@pytest.mark.parametrize("tamper, match", [
    (lambda b, t, c: c.update(sample_count=47), "provenance is inconsistent"),
    (lambda b, t, c: c.update(is_final_blind_test=True), "provenance is inconsistent"),
    (lambda b, t, c: t.update(model_revision="b" * 40), "provenance is inconsistent"),
    (lambda b, t, c: t.update(dataset_sha256="f" * 64), "provenance is inconsistent"),
])
def test_rejects_mismatched_pair_report(tmp_path, tamper, match):
    root, archive, digest = make_fake_bundle(tmp_path, edits=tamper)
    with pytest.raises(ValueError, match=match):
        export_from_bundle(root, archive, digest)
    assert not (root / "reviews").exists()


def test_manifest_does_not_authorize_modified_member(tmp_path):
    def modify(data):
        data["dev48_base.json"] += b"  "
        return data
    root, archive, digest = make_fake_bundle(tmp_path, zip_entries=modify)
    with pytest.raises(ValueError, match="Manifest SHA-256 mismatch"):
        verify_bundle(archive, digest)


def test_rejects_extra_member_and_missing_member(tmp_path):
    for index, change in enumerate((
        lambda d: {**d, "evil.txt": b"extra"},
        lambda d: {k: v for k, v in d.items() if k != "dev48_base.json"},
    )):
        root, archive, digest = make_fake_bundle(tmp_path / f"invalid_{index}", zip_entries=change)
        with pytest.raises(ValueError, match="Unexpected, missing"):
            verify_bundle(archive, digest)


def test_rejects_nonfinite_and_duplicate_json_even_if_manifest_resigned(tmp_path):
    def custom(data):
        name = "dev48_paired_auditkit.json"
        data[name] = data[name][:-1] + b', "sample_count": 48}'
        manifest = json.loads(data["BUNDLE_MANIFEST.json"])
        manifest["files_sha256"][name] = hashlib.sha256(data[name]).hexdigest()
        data["BUNDLE_MANIFEST.json"] = json.dumps(manifest).encode()
        return data
    root, archive, digest = make_fake_bundle(tmp_path, zip_entries=custom)
    with pytest.raises(ValueError, match="malformed JSON"):
        verify_bundle(archive, digest)


def test_exclusive_create_never_overwrites_packet(tmp_path):
    root, archive, digest = make_fake_bundle(tmp_path)
    export_from_bundle(root, archive, digest)
    original = (root / "reviews/dev48_blank.csv").read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        export_from_bundle(root, archive, digest)
    assert (root / "reviews/dev48_blank.csv").read_bytes() == original


def test_verify_only_has_no_side_effects(tmp_path):
    root, archive, digest = make_fake_bundle(tmp_path)
    verify_bundle(archive, digest)
    assert not (root / "reviews").exists()
