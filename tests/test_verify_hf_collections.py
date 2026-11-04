"""Anonymous HF publication checks use a fake API: NO network or publishing."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256 as hashlib_sha256
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from policyproof.core import ROOT
from scripts.verify_hf_collections import (
    MAX_AGE_HOURS, REQUIRED_DATASET_FILES,
    _url_id, receipt_is_current, validate_manifest, verify_anonymous_access,
)

MODEL_ID = "owner/policyproof-lexsi-model"
DATASET_ID = "owner/policyproof-lexsi-safety"
MODELS = "owner/lexsi-models-abcdef123"
DATASETS = "owner/lexsi-datasets-abcdef123"


def manifest_file(tmp_path, **updates):
    directory = tmp_path / "docs"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_collection_url": "https://huggingface.co/collections/" + MODELS,
        "dataset_collection_url": "https://huggingface.co/collections/" + DATASETS,
        "dataset_url": "https://huggingface.co/datasets/" + DATASET_ID,
        "external_access_verified": True,  # malicious self-report; ignored
        "public_items_ungated_verified": True,  # also ignored
    }
    payload.update(updates)
    path = directory / "hf_publication_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class AnonymousFakeAPI:
    """Represents external responses, deliberately NOT actual HF resources."""
    def __init__(self, temp: Path):
        self.calls = []
        self.collection_private = False
        self.repo_private = False
        self.repo_gated = False
        self.missing_file = None
        self.omit_dataset_membership = False
        self.bad_public_scope = False
        self.tmp = temp
        files = temp / "api_files"
        files.mkdir(exist_ok=True)
        (files / "README.md").write_text("---\nlicense: cc-by-4.0\n---\n# Synthetic")
        (files / "PUBLICATION_SCOPE.json").write_text(json.dumps({
            "evaluated_base_model_cases":48,
            "new_unscored_model_cases":72,
            "independent_human_review_completed":False,
            "tuned_adapter_included":False,
        }))
        (files / "original_saved_tiny_aya_base_model_predictions.json").write_text(json.dumps({
            "is_finetuned":False,
            "predictions":[{"id":str(i),"raw":"test"} for i in range(48)],
        }))
        (files / "challenge_v1_UNSCORED.jsonl").write_text("".join(
            json.dumps({
                "id":f"C{i}", "model_evaluated":False,
                "human_reviewed":False, "used_for_training":False,
            }) + "\n" for i in range(72)
        ))
        (files / "SHA256SUMS.txt").write_text("".join(
            hashlib_sha256((files / Path(fn).name).read_bytes()).hexdigest()
            + "  " + fn + "\n"
            for fn in REQUIRED_DATASET_FILES if fn != "provenance/SHA256SUMS.txt"
        ))
        self.files = files

    def get_collection(self, slug, *, token):
        assert token is False
        self.calls.append(("get_collection", slug, token))
        if slug == MODELS:
            items = [SimpleNamespace(item_type="model", item_id=MODEL_ID)]
        else:
            items = [SimpleNamespace(
                item_type="dataset", item_id=(
                    "owner/unrelated" if self.omit_dataset_membership else DATASET_ID
                )
            )]
        return SimpleNamespace(private=self.collection_private, items=items)

    def repo_info(self, *, repo_id, repo_type, token):
        assert token is False
        self.calls.append(("repo_info", repo_id, repo_type, token))
        return SimpleNamespace(
            private=self.repo_private, gated=self.repo_gated, sha="a"*40
        )

    def list_repo_files(self, *, repo_id, repo_type, revision, token):
        assert token is False and revision == "a" * 40
        self.calls.append(("list_repo_files", repo_id, token))
        return [x for x in REQUIRED_DATASET_FILES if x != self.missing_file]

    def hf_hub_download(self, *, repo_id, repo_type, revision, filename, token):
        assert token is False and revision == "a" * 40
        self.calls.append(("hf_hub_download", filename, token))
        result = self.files / Path(filename).name
        if self.bad_public_scope and filename.endswith("PUBLICATION_SCOPE.json"):
            result.write_text(json.dumps({
                "evaluated_base_model_cases":48,
                "new_unscored_model_cases":72,
                "independent_human_review_completed":True,
                "tuned_adapter_included":False,
            }))
        return str(result)


def test_default_manifest_not_yet_published():
    # Genuine current project still has no HF collection links.
    with pytest.raises(ValueError):
        validate_manifest(ROOT / "docs/hf_publication_manifest.json")


def test_offline_no_network_by_default():
    done = subprocess.run(
        [sys.executable, "-m", "scripts.verify_hf_collections"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert done.returncode == 0
    assert "WAITING_FOR_REAL_PUBLIC_HF_LINKS" in done.stdout
    assert '"network_used": false' in done.stdout
    assert '"publication_performed": false' in done.stdout


@pytest.mark.parametrize("url", [
    "http://huggingface.co/datasets/owner/data",
    "https://huggingface.co.evil.io/datasets/owner/data",
    "https://user:secret@huggingface.co/datasets/owner/data",
    "https://huggingface.co/datasets/owner/data?token=secret",
    "https://huggingface.co/datasets/owner/data#fragment",
    "https://huggingface.co/models/owner/data",
    "https://huggingface.co/datasets/owner/data/extra",
])
def test_url_validation_rejects_noncanonical_dataset(url):
    with pytest.raises(ValueError):
        _url_id(url, "dataset")


def test_anonymous_fetched_research_files_and_receipt_are_consistent(tmp_path):
    manifest = manifest_file(tmp_path)
    api = AnonymousFakeAPI(tmp_path)
    receipt = verify_anonymous_access(manifest, api=api)
    assert receipt["status"] == "ANONYMOUS_PUBLIC_ACCESS_VERIFIED"
    assert receipt["public_baseline_cases"] == 48
    assert receipt["public_unscored_challenge_cases"] == 72
    assert receipt["all_referenced_repositories_public_ungated"] is True
    assert receipt["model_collection_project_relevance_verified"] is False
    assert receipt["organizer_acceptance_verified"] is False
    assert receipt["official_submission_completed"] is False
    assert all(call[-1] is False for call in api.calls if call[0] != "hf_hub_download")
    assert all(call[-1] is False for call in api.calls if call[0] == "hf_hub_download")
    output = tmp_path / "artifacts" / "hf_anonymous_access_receipt.json"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(receipt))
    assert receipt_is_current(output, manifest)


def test_false_publication_booleans_cannot_create_receipt(tmp_path):
    manifest = manifest_file(tmp_path)
    assert validate_manifest(manifest)["owner_supplied_verification_flags_ignored"]
    assert not receipt_is_current(tmp_path / "absent.json", manifest)


@pytest.mark.parametrize("alter", [
    lambda x: setattr(x, "collection_private", True),
    lambda x: setattr(x, "repo_private", True),
    lambda x: setattr(x, "repo_gated", "auto"),
    lambda x: setattr(x, "missing_file", "README.md"),
    lambda x: setattr(x, "omit_dataset_membership", True),
    lambda x: setattr(x, "bad_public_scope", True),
])
def test_fail_closed_for_private_gated_missing_or_mislabeled_content(tmp_path, alter):
    manifest = manifest_file(tmp_path)
    api = AnonymousFakeAPI(tmp_path)
    alter(api)
    with pytest.raises(ValueError):
        verify_anonymous_access(manifest, api=api)


def test_stale_and_changed_manifest_invalidates_old_receipt(tmp_path):
    manifest = manifest_file(tmp_path)
    receipt = verify_anonymous_access(manifest, api=AnonymousFakeAPI(tmp_path))
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    now = datetime.now(timezone.utc)
    assert receipt_is_current(path, manifest, now=now)
    assert not receipt_is_current(
        path, manifest, now=now + timedelta(hours=MAX_AGE_HOURS + 2)
    )
    assert not receipt_is_current(
        path, manifest, now=now - timedelta(hours=MAX_AGE_HOURS + 2)
    )
    doc = json.loads(manifest.read_text())
    doc["notes"] = "Changed after verification"
    manifest.write_text(json.dumps(doc))
    assert not receipt_is_current(path, manifest)


def test_forged_receipt_flag_or_duplicate_json_fails(tmp_path):
    manifest = manifest_file(tmp_path)
    receipt = verify_anonymous_access(manifest, api=AnonymousFakeAPI(tmp_path))
    path = tmp_path / "receipt.json"
    receipt["all_referenced_repositories_public_ungated"] = False
    path.write_text(json.dumps(receipt))
    assert not receipt_is_current(path, manifest)
    receipt["all_referenced_repositories_public_ungated"] = True
    raw = json.dumps(receipt).replace(
        '"status":', '"status":"FAKE", "status":', 1
    )
    path.write_text(raw)
    assert not receipt_is_current(path, manifest)


def test_untrusted_fake_collection_items_cannot_claim_publication(tmp_path):
    manifest = manifest_file(tmp_path)
    api = AnonymousFakeAPI(tmp_path)
    api.repo_gated = None  # unknown must not silently mean ungated
    with pytest.raises(ValueError, match="gated"):
        verify_anonymous_access(manifest, api=api)
