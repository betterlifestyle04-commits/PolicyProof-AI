"""Offline regression for published native40 readback; NO HF network or write."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_existing_native40_public_model as verify


@pytest.fixture
def public_repo(tmp_path, monkeypatch):
    weights = b"FAKE_TEST_ADAPTER_NOT_REAL_WEIGHTS"
    cfg = b'{"peft_type": "LORA"}'
    monkeypatch.setattr(verify, "ADAPTER_SHA", verify.sha256_bytes(weights))
    monkeypatch.setattr(verify, "CONFIG_SHA", verify.sha256_bytes(cfg))
    card = b"---\nlicense: cc-by-nc-4.0\n---\n" \
        b"NOT a safety-improved model CohereLabs/tiny-aya-global\n"
    metrics = json.dumps({
        "case_count": 48, "base": {
            "schema_valid": 14, "unsupported_correct_abstentions": 4,
        }, "native_aligntune_adapter": {
            "schema_valid": 48, "unsupported_correct_abstentions": 2,
        }, "adapter_safety_improvement_claimed": False,
    }).encode()
    base = {
        "adapter_model.safetensors": weights,
        "adapter_config.json": cfg,
        "README.md": card,
        "EVALUATION_SUMMARY.json": metrics,
    }
    manifest = {
        "original_private_source_zip_sha256": verify.ORIGINAL_PRIVATE_SOURCE,
        "private_raw_model_logs_included": False,
        "source_base_weights_included": False,
        "model_safety_improvement_claimed": False,
        "public_file_sha256": {
            k: verify.sha256_bytes(v) for k, v in base.items()
        },
    }
    base["RELEASE_MANIFEST.json"] = json.dumps(manifest).encode()
    for filename, blob in base.items():
        (tmp_path / filename).write_bytes(blob)
    calls = []
    class API:
        def model_info(self,**kwargs):
            calls.append(("model_info",kwargs))
            return SimpleNamespace(
                private=False, gated=False, sha=verify.PUBLISHED_MODEL_COMMIT,
                siblings=[SimpleNamespace(rfilename=x) for x in
                          sorted(base) + [".gitattributes"]],
            )
        def get_collection(self,slug,**kwargs):
            calls.append(("get_collection",slug,kwargs))
            return SimpleNamespace(
                private=False,
                items=[SimpleNamespace(item_id=verify.MODEL_ID,item_type="model")],
            )
    def download(**kwargs):
        calls.append(("hf_hub_download",kwargs))
        return str(tmp_path / kwargs["filename"])
    return API(), download, calls, tmp_path


def test_public_native40_full_sha_readback_only_uses_anonymous_reads(public_repo):
    api, downloader, calls, _ = public_repo
    receipt = verify.verify_existing_release(api=api,downloader=downloader)
    assert receipt["status"] == (
        "PUBLIC_NATIVE40_MODEL_AND_COLLECTION_ANONYMOUSLY_VERIFIED_NO_UPLOAD"
    )
    assert receipt["public_file_count"] == 5
    assert receipt["all_public_files_sha256_verified"] is True
    assert receipt["public_collection_contains_expected_model"] is True
    assert receipt["publication_attempted_in_this_check"] is False
    assert receipt["official_lexsi_submission_verified"] is False
    assert receipt["license_rights_independently_approved"] is False
    assert [c[0] for c in calls][:2] == ["model_info","get_collection"]
    assert len([c for c in calls if c[0]=="hf_hub_download"]) == 5
    assert calls[0][1] == {"repo_id":verify.MODEL_ID,"token":False}
    for _,kwargs in [c for c in calls if c[0]=="hf_hub_download"]:
        assert kwargs["token"] is False
        assert kwargs["revision"] == verify.PUBLISHED_MODEL_COMMIT
        assert kwargs["repo_type"] == "model"


def test_changed_public_weights_fail_even_if_receipt_honest(public_repo):
    api,downloader,_,root = public_repo
    (root/"adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError,match="SHA256 release manifest"):
        verify.verify_existing_release(api=api,downloader=downloader)


def test_wrong_model_commit_rejected_before_any_file_download(public_repo):
    api,downloader,calls,_=public_repo
    api.model_info=lambda **kwargs: SimpleNamespace(
        private=False,gated=False,sha="0"*40,siblings=[],
    )
    with pytest.raises(RuntimeError,match="not public, ungated, or pinned"):
        verify.verify_existing_release(api=api,downloader=downloader)
    assert not any(c[0]=="hf_hub_download" for c in calls)


def test_private_collection_or_missing_model_rejected(public_repo):
    api,downloader,calls,_=public_repo
    api.get_collection=lambda slug,**kwargs: SimpleNamespace(
        private=True,items=[SimpleNamespace(item_id=verify.MODEL_ID,item_type="model")],
    )
    with pytest.raises(RuntimeError,match="not verifiably public"):
        verify.verify_existing_release(api=api,downloader=downloader)
    api.get_collection=lambda slug,**kwargs: SimpleNamespace(
        private=False,items=[SimpleNamespace(item_id="other/repo",item_type="model")],
    )
    with pytest.raises(RuntimeError,match="no matching"):
        verify.verify_existing_release(api=api,downloader=downloader)
    assert not any(c[0]=="hf_hub_download" for c in calls)


def test_fabricated_safety_improvement_fails_closed(public_repo):
    api,downloader,_,root=public_repo
    report=json.loads((root/"EVALUATION_SUMMARY.json").read_text())
    report["adapter_safety_improvement_claimed"]=True
    (root/"EVALUATION_SUMMARY.json").write_text(json.dumps(report))
    manifest=json.loads((root/"RELEASE_MANIFEST.json").read_text())
    manifest["public_file_sha256"]["EVALUATION_SUMMARY.json"]=verify.sha256_bytes(
        (root/"EVALUATION_SUMMARY.json").read_bytes()
    )
    (root/"RELEASE_MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError,match="Public evaluation"):
        verify.verify_existing_release(api=api,downloader=downloader)


def test_unexpected_private_public_file_fails_closed(public_repo):
    api,downloader,_,root=public_repo
    source_method=api.model_info
    def enhanced(**kwargs):
        result=source_method(**kwargs)
        return SimpleNamespace(
            private=False,gated=False,sha=verify.PUBLISHED_MODEL_COMMIT,
            siblings=result.siblings+[SimpleNamespace(rfilename="raw_predictions.json")],
        )
    api.model_info=enhanced
    with pytest.raises(RuntimeError,match="unexpected/missing"):
        verify.verify_existing_release(api=api,downloader=downloader)


def test_duplicate_json_key_rejected():
    with pytest.raises(ValueError,match="Invalid public JSON"):
        verify.safe_json(b'{"a":1,"a":2}')


def test_manifest_has_correct_real_published_model_and_canonical_collections():
    manifest=json.loads(
        (Path(__file__).resolve().parents[1] /
         "docs/hf_publication_manifest.json").read_text()
    )
    assert manifest["adapter_url"] == verify.MODEL_URL
    assert manifest["model_collection_url"] == verify.COLLECTION_URL
    assert "6af24cc5b3c20ac76f6ceef5" in manifest["dataset_collection_url"]
    assert manifest["submission_confirmation_verified"] is False
    assert manifest["public_items_ungated_verified"] is False
