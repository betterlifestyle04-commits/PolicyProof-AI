"""Offline tests for consent-gated HF adapter-only release.

All model weights are fake bytes with synthetic source SHA pins. HF client
calls are injected; this suite NEVER connects, uploads, or reads owner ZIP.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import zipfile

import pytest

from scripts import owner_native40_hf_release as release


@pytest.fixture
def fake_original(tmp_path, monkeypatch):
    archive = tmp_path / "original-native40.zip"
    data = b"FAKE_LORA_TENSOR_DATA_NOT_A_REAL_MODEL"
    peft = json.dumps({
        "base_model_name_or_path": "CohereLabs/tiny-aya-global",
        "peft_type": "LORA", "r": 8, "lora_alpha": 16,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }).encode()
    receipt = json.dumps({
        "trainer_backend": "aligntune_wrapper",
        "optimizer_steps_completed": 40,
        "aligntune_trainer_completed": True,
        "claims_improved_safety": False,
        "independent_gpu_execution_attested": False,
        "dataset_sha256": release.TRAIN_SHA,
        "examples": 78, "base_revision": release.MODEL_REVISION_REQUESTED,
    }).encode()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("adapter/adapter_model.safetensors", data)
        z.writestr("adapter/adapter_config.json", peft)
        z.writestr("run_info.json", receipt)
    monkeypatch.setattr(release, "ORIGINAL_SHA", release.sha_bytes(archive.read_bytes()))
    monkeypatch.setattr(release, "WEIGHTS_SHA", release.sha_bytes(data))
    monkeypatch.setattr(release, "CONFIG_SHA", release.sha_bytes(peft))
    return archive


def test_preview_is_local_write_once_and_omits_private_sources(fake_original,tmp_path):
    zip_path = tmp_path / "policyproof-HF-PREVIEW-NOT_PUBLISHED.zip"
    result = release.write_offline_preview(fake_original, zip_path)
    assert result["status"] == "OWNER_ONLY_PUBLIC_ADAPTER_PREVIEW_NOT_PUBLISHED"
    assert result["public_model_repo_created"] is False
    assert result["hugging_face_token_used"] is False
    with zipfile.ZipFile(zip_path) as z:
        assert set(z.namelist()) == release.PUBLIC_MEMBERS
        assert "run_info.json" not in z.namelist()
        assert "private_base_model.safetensors" not in z.namelist()
        assert "raw_48_outputs.json" not in z.namelist()
        card = z.read("README.md")
        assert b"NOT a safety-improved model" in card
        stats = json.loads(z.read("EVALUATION_SUMMARY.json"))
        assert stats["base"]["schema_valid"] == 14
        assert stats["native_aligntune_adapter"]["schema_valid"] == 48
        assert stats["native_aligntune_adapter"]["unsupported_nonabstaining_claims"] == 7
        assert stats["adapter_safety_improvement_claimed"] is False
        assert stats["private_prediction_logs_published"] is False
        manifest = json.loads(z.read("RELEASE_MANIFEST.json"))
        assert manifest["source_base_weights_included"] is False
        assert manifest["public_file_sha256"]["adapter_model.safetensors"] == release.WEIGHTS_SHA
    original = zip_path.read_bytes()
    with pytest.raises(FileExistsError):
        release.write_offline_preview(fake_original, zip_path)
    assert zip_path.read_bytes() == original


@pytest.mark.parametrize("name", [
    "wrong.zip", "release.zip", "model_NOT_PUBLISHED.txt",
])
def test_preview_requires_not_published_zip_filename(fake_original,tmp_path,name):
    with pytest.raises(FileExistsError):
        release.write_offline_preview(fake_original, tmp_path / name)


def test_missing_model_card_fails_without_archive(fake_original,tmp_path):
    with pytest.raises(ValueError,match="model-card"):
        release.write_offline_preview(
            fake_original, tmp_path / "release-NOT_PUBLISHED.zip",
            card_path=tmp_path / "missing.md",
        )


def test_source_archive_sha_drift_is_not_publishable(fake_original,tmp_path):
    # Test direct mismatch against real expected original SHA.
    old = release.ORIGINAL_SHA
    release.ORIGINAL_SHA = "0"*64
    try:
        with pytest.raises(ValueError,match="SHA256 mismatch"):
            release.build_release_bytes(fake_original)
    finally:
        release.ORIGINAL_SHA = old


def test_false_improvement_claim_blocks_staging(fake_original,tmp_path):
    with zipfile.ZipFile(fake_original,"r") as z:
        members={name:z.read(name) for name in z.namelist()}
    receipt=json.loads(members["run_info.json"])
    receipt["claims_improved_safety"]=True
    members["run_info.json"]=json.dumps(receipt).encode()
    with zipfile.ZipFile(fake_original,"w") as z:
        for name,blob in members.items():
            z.writestr(name,blob)
    release.ORIGINAL_SHA=release.sha_bytes(fake_original.read_bytes())
    with pytest.raises(ValueError,match="provenance"):
        release.build_release_bytes(fake_original)


@pytest.mark.parametrize("flags", [
    (False, True, True, True),
    (True, False, True, True),
    (True, True, False, True),
    (True, True, True, False),
])
def test_publish_refuses_any_missing_owner_consent_before_hf(fake_original,flags):
    pub,lic,outputs,sha_valid=flags
    with pytest.raises(PermissionError,match="Public adapter release blocked"):
        release.publish_owner_approved(
            fake_original,namespace="shalnark01",
            confirmed_source_sha256=release.ORIGINAL_SHA if sha_valid else "0"*64,
            approve_public=pub, approve_license=lic, approve_outputs=outputs,
            api=object(),anonymous_api=object(),
        )


def test_wrong_authenticated_identity_refuses_creating_repo(fake_original):
    calls=[]
    class API:
        def whoami(self,**kw):
            calls.append("whoami")
            return {"name":"not-owner"}
        def create_repo(self,**kw):
            raise AssertionError("should never create remote repository")
    with pytest.raises(PermissionError,match="identity"):
        release.publish_owner_approved(
            fake_original,namespace="shalnark01",
            confirmed_source_sha256=release.ORIGINAL_SHA,
            approve_public=True,approve_license=True,approve_outputs=True,
            api=API(),anonymous_api=object(),
        )
    assert calls==["whoami"]


def test_owner_approved_publication_has_no_gated_base_files(fake_original,tmp_path):
    calls=[]
    owner_repo = "shalnark01/" + release.MODEL_REPO_NAME
    collection_slug = "shalnark01/policyproof-models-123"
    class API:
        def whoami(self,**kwargs):
            assert kwargs["token"] is True
            calls.append("whoami")
            return {"name":"shalnark01"}
        def create_repo(self,**kwargs):
            assert kwargs == {
                "repo_id":owner_repo,"repo_type":"model",
                "private":False,"exist_ok":False,"token":True,
            }
            calls.append("repo")
        def upload_folder(self,**kwargs):
            path = Path(kwargs["folder_path"])
            assert set(p.name for p in path.iterdir()) == release.PUBLIC_MEMBERS
            assert set(kwargs["allow_patterns"]) == release.PUBLIC_MEMBERS
            assert kwargs["token"] is True
            calls.append("upload")
        def create_collection(self,**kwargs):
            assert kwargs["private"] is False
            assert kwargs["namespace"] == "shalnark01"
            calls.append("collection")
            return SimpleNamespace(slug=collection_slug)
        def add_collection_item(self,**kwargs):
            assert kwargs["item_type"] == "model"
            assert kwargs["item_id"] == owner_repo
            calls.append("add")
    class Anonymous:
        def model_info(self,**kwargs):
            assert kwargs["token"] is False
            calls.append("anon_model")
            return SimpleNamespace(
                private=False,gated=False,
                siblings=[SimpleNamespace(rfilename=n)
                          for n in release.PUBLIC_MEMBERS]
            )
        def get_collection(self,*args,**kwargs):
            calls.append("anon_collection")
            assert kwargs["token"] is False
            return SimpleNamespace(items=[SimpleNamespace(item_id=owner_repo)])
    payload = release.build_release_bytes(fake_original)
    def anon_download(**kwargs):
        assert kwargs["token"] is False
        assert kwargs["repo_id"] == owner_repo
        assert kwargs["repo_type"] == "model"
        filename = kwargs["filename"]
        candidate = tmp_path / filename
        candidate.write_bytes(payload[filename])
        calls.append("anon_file")
        return str(candidate)
    result=release.publish_owner_approved(
        fake_original,namespace="shalnark01",
        confirmed_source_sha256=release.ORIGINAL_SHA,
        approve_public=True,approve_license=True,approve_outputs=True,
        api=API(),anonymous_api=Anonymous(),
        anonymous_downloader=anon_download,
    )
    assert result["status"] == "PUBLIC_ADAPTER_REPO_AND_MODEL_COLLECTION_ANONYMOUSLY_VERIFIED"
    assert result["model_repo_url"]=="https://huggingface.co/"+owner_repo
    assert result["owner_official_Lexsi_submission_done"] is False
    assert calls==[
        "whoami","repo","upload","collection","add","anon_model",
        *["anon_file"]*len(release.PUBLIC_MEMBERS), "anon_collection"
    ]


def test_mismatched_remote_visibility_never_claims_success(fake_original):
    class API:
        def whoami(self,**kwargs):return {"name":"shalnark01"}
        def create_repo(self,**kwargs):pass
        def upload_folder(self,**kwargs):pass
        def create_collection(self,**kwargs):return SimpleNamespace(slug="collection")
        def add_collection_item(self,**kwargs):pass
    class Anonymous:
        def model_info(self,**kwargs):return SimpleNamespace(private=True,gated=False)
    with pytest.raises(RuntimeError,match="not confirmed public"):
        release.publish_owner_approved(
            fake_original,namespace="shalnark01",
            confirmed_source_sha256=release.ORIGINAL_SHA,
            approve_public=True,approve_license=True,approve_outputs=True,
            api=API(),anonymous_api=Anonymous(),
        )
