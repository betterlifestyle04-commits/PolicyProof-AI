"""Offline compatibility check against ACTUAL huggingface_hub Python classes.

This only introspects the installed SDK; no HTTP, token, credentials, uploads
or Hugging Face repository reads are performed. It prevents fake-API-only tests
from masking incompatible method names, parameters and collection-item shape.
"""
from __future__ import annotations

import inspect
import json


def check_contract() -> dict:
    from huggingface_hub import (
        HfApi, CollectionItem, DatasetInfo, __version__,
    )

    api = HfApi(token=False)
    contract = {
        "hf_hub_download": ("repo_id", "repo_type", "filename", "revision", "token"),
        "get_collection": ("collection_slug", "token"),
        "repo_info": ("repo_id", "repo_type", "revision", "token"),
        "list_repo_files": ("repo_id", "repo_type", "revision", "token"),
        "create_repo": ("repo_id", "repo_type", "private"),
        "upload_folder": ("repo_id", "repo_type", "folder_path"),
        "create_collection": ("title", "namespace", "private"),
        "add_collection_item": ("collection_slug", "item_id", "item_type"),
    }
    for name, parameters in contract.items():
        method = getattr(api, name, None)
        if not callable(method):
            raise RuntimeError("Installed Hugging Face Hub HfApi lacks: " + name)
        actual = inspect.signature(method).parameters
        missing = set(parameters) - set(actual)
        if missing:
            raise RuntimeError(
                "Installed HfApi method lacks required arguments: "
                + name + " (" + ", ".join(sorted(missing)) + ")"
            )

    # Use real SDK data classes: fake SimpleNamespace tests cannot catch
    # refactors that rename item_type/item_id or private/gated/sha.
    item = CollectionItem(
        _id="test-object-not-real",
        id="not-a-real-user/synthetic-dataset",
        type="dataset",
        position=0,
    )
    if (item.item_type != "dataset" or
        item.item_id != "not-a-real-user/synthetic-dataset"):
        raise RuntimeError("Unexpected HF collection-item field contract")

    dataset = DatasetInfo(
        id="not-a-real-user/synthetic-dataset",
        private=False, gated=False, sha="a" * 40,
    )
    if (dataset.private is not False or dataset.gated is not False
        or dataset.sha != "a" * 40):
        raise RuntimeError("Unexpected HF dataset visibility/commit field contract")

    return {
        "status": "INSTALLED_HF_SDK_OFFLINE_CONTRACT_VERIFIED",
        "sdk_version": __version__,
        "method_contracts_checked": sorted(contract),
        "collection_item_fields_verified": True,
        "dataset_visibility_and_revision_fields_verified": True,
        "network_used": False,
        "model_inference_performed": False,
        "uploaded_to_huggingface": False,
    }


if __name__ == "__main__":
    print(json.dumps(check_contract(), indent=2))
