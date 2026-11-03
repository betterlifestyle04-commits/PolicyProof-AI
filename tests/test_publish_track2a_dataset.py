"""Offline-only publication guard tests. NEVER call the real Hugging Face API."""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from scripts.publish_track2a_dataset import (
    stage, validate_archive, validate_flags, publish
)

CARD="""---
language:
- en
---
# Demo
**Research draft, NOT YET PUBLISHED on Hugging Face.** This ZIP is staged for the dataset owner to review, select a license, and upload after authenticating to their own HF account.

## Licensing
No rights selected.
"""


def fake_package(path:Path, *,tamper=False):
    challenge=[
        {"id":f"CH-{i}","model_evaluated":False,"human_reviewed":False,
         "used_for_training":False} for i in range(72)
    ]
    files={
        "README.md":CARD.encode(),
        "LICENSE_REVIEW_REQUIRED.md":b"REVIEW REQUIRED",
        "PUBLISHING_INSTRUCTIONS.md":b"DO NOT PUBLISH WITHOUT APPROVAL",
        "challenge_72/challenge_v1_UNSCORED.jsonl":(
            "".join(json.dumps(x)+"\n" for x in challenge)).encode(),
        "challenge_72/challenge_manifest_UNSCORED.json":b"{}",
        "evaluated_48/original_saved_tiny_aya_base_model_predictions.json":json.dumps({
            "is_finetuned":False,"predictions":[{"id":str(i),"raw":"{}"} for i in range(48)]
        }).encode(),
        "evaluated_48/curator_audit_chain_report.json":b"{}",
        "evaluated_48/auditkit_native_receipt.json":b"{}",
        "provenance/PACKAGE_STATUS.json":json.dumps({
            "status":"local_hf_upload_staging_ONLY_not_published",
            "evaluated_cases":48,"new_unscored_cases":72,"selected_license":None
        }).encode()
    }
    files["provenance/SHA256SUMS.txt"]=(
        "".join(sha256(value).hexdigest()+"  "+key+"\n"
                for key,value in sorted(files.items()) if key!="provenance/PACKAGE_STATUS.json")
    ).encode()
    if tamper:files["README.md"]=b"CHANGED WITHOUT UPDATING MANIFEST"
    with ZipFile(path,"w") as z:
        for name,data in files.items():z.writestr(name,data)


def test_offline_zip_integrity_and_honest_staging(tmp_path):
    original=tmp_path/"test.zip"
    fake_package(original)
    files=validate_archive(original,expected_sha256=None)
    assert len(files)==10
    staged=tmp_path/"staged"
    names=stage(files,staged,"cc-by-4.0")
    assert "README.md" in names
    assert "LICENSE_REVIEW_REQUIRED.md" not in names
    assert "provenance/PACKAGE_STATUS.json" not in names
    card=(staged/"README.md").read_text()
    assert card.startswith("---\nlicense: cc-by-4.0\n")
    assert "NOT YET PUBLISHED" not in card
    assert "48" in (staged/"provenance/PUBLICATION_SCOPE.json").read_text()
    assert "new_unscored_model_cases" in (staged/"provenance/PUBLICATION_SCOPE.json").read_text()


def test_tampering_fail_closed(tmp_path):
    original=tmp_path/"bad.zip"
    fake_package(original,tamper=True)
    with pytest.raises(ValueError,match="checksum mismatch"):
        validate_archive(original,expected_sha256=None)


def test_network_publish_requires_explicit_user_consent_and_auth(monkeypatch):
    args=SimpleNamespace(repo_id="demo/project",publish=True,confirm_rights=False,
          confirm_publication=True,create_collection=True,collection_slug=None)
    monkeypatch.setenv("HF_TOKEN","fake_not_a_real_token")
    with pytest.raises(ValueError,match="explicit rights"):
        validate_flags(args)
    args.confirm_rights=True
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(ValueError,match="HF_TOKEN"):
        validate_flags(args)
    args.publish=False
    args.create_collection=False
    args.confirm_publication=False
    validate_flags(args)


def test_publication_rejects_existing_receipt_or_bad_license(tmp_path):
    p=tmp_path/"test.zip"
    fake_package(p)
    with pytest.raises(ValueError,match="Invalid/undecided"):
        stage(validate_archive(p,expected_sha256=None),tmp_path/"bad-stage","undefined")


class FakeAPI:
    def __init__(self, public=True):
        self.public=public
        self.repo_files=set()
        self.collection_items=[]
        self.created_collection=False

    def create_repo(self,**kwargs):
        assert kwargs["private"] is False
        return SimpleNamespace()

    def repo_info(self,**kwargs):
        return SimpleNamespace(private=not self.public)

    def upload_folder(self,**kwargs):
        root=Path(kwargs["folder_path"])
        self.repo_files={str(x.relative_to(root)) for x in root.rglob("*") if x.is_file()}
        return SimpleNamespace(oid="test-commit")

    def list_repo_files(self,**kwargs):
        return self.repo_files

    def hf_hub_download(self,**kwargs):
        assert kwargs["token"] is False
        return "README.md"

    def create_collection(self,**kwargs):
        assert kwargs["private"] is False
        self.created_collection=True
        return SimpleNamespace(slug="demo/collection-123")

    def add_collection_item(self,**kwargs):
        self.collection_items.append(SimpleNamespace(
            item_id=kwargs["item_id"],item_type=kwargs["item_type"]))

    def get_collection(self,*args,**kwargs):
        return SimpleNamespace(private=not self.public,items=self.collection_items)


def test_mock_publish_verifies_anonymous_visibility_without_token(tmp_path):
    p=tmp_path/"test.zip"
    fake_package(p)
    staged=tmp_path/"upload"
    names=stage(validate_archive(p,expected_sha256=None),staged,"cc0-1.0")
    authorized=FakeAPI()
    public=FakeAPI()
    # Share only simulated public state, never real access credentials.
    public.repo_files=names
    public.collection_items=authorized.collection_items
    receipt=publish(staged,names,"demo/project","fake",
                   new_collection=True,api=authorized,anonymous_api=public)
    assert receipt["anonymous_repo_and_collection_verified"] is True
    assert receipt["model_collection_created"] is False
    assert receipt["official_lexsi_submission_completed"] is False


def test_no_claim_on_private_repository(tmp_path):
    p=tmp_path/"test.zip"
    fake_package(p)
    staged=tmp_path/"upload"
    names=stage(validate_archive(p,expected_sha256=None),staged,"cc0-1.0")
    with pytest.raises(RuntimeError,match="private"):
        publish(staged,names,"demo/project","fake",
                new_collection=True,api=FakeAPI(public=False),
                anonymous_api=FakeAPI())

def test_owner_publisher_rejects_any_nonmatching_unverified_zip(tmp_path):
    p=tmp_path/"unverified.zip"
    fake_package(p)
    with pytest.raises(ValueError,match="ZIP hash differs"):
        validate_archive(p)
