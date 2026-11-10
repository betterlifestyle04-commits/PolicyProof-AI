"""Read-only source-pinned verification of the EXISTING public native40 HF release.

This deliberately does not require model ZIP, write token, trainer or GPU. It
never calls create_repo, upload_folder, create_collection, or delete.
The owner's earlier one-shot publish can fail after all publication actions;
an exit code alone must not trigger duplicate uploads. Inspect anonymous
model repo, exact public file hashes and model collection before proceeding.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

MODEL_ID = "shalnark01/policyproof-ai-native40-align-2026"
MODEL_URL = "https://huggingface.co/" + MODEL_ID
COLLECTION_SLUG = (
    "shalnark01/policyproof-ai-lexsi-track-2-model-and-adapter-"
    "research-6af39caacac578d29e1c66c6"
)
COLLECTION_URL = "https://huggingface.co/collections/" + COLLECTION_SLUG
PUBLISHED_MODEL_COMMIT = "3ba0aa22f9505bd6dc5cf4bd3d3c629eb6e8c0d3"
ORIGINAL_PRIVATE_SOURCE = (
    "dcefc5b0babf708aa1b48ac0349ddb3bc4745405c72e9636c5860d7719d8f0e9"
)
ADAPTER_SHA = "ba985c1966050764a5220e87ea581a9bd482ba87b055fee4a1e363643f1cc92e"
CONFIG_SHA = "808fea2999f91c2af09271b275877080ecf89683faec5e7a63f8000ec5d25621"
PUBLIC_FILES = frozenset({
    "adapter_model.safetensors", "adapter_config.json", "README.md",
    "EVALUATION_SUMMARY.json", "RELEASE_MANIFEST.json",
})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_json(data: bytes, *, limit=100_000) -> dict:
    if not 0 < len(data) < limit:
        raise ValueError("Public JSON file empty or oversized")
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("Duplicate key in public JSON")
            result[k] = v
        return result
    try:
        parsed = json.loads(
            data, object_pairs_hook=unique,
            parse_constant=lambda v: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON value")
            ),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid public JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Public JSON root is not an object")
    return parsed


def verify_existing_release(*, api=None, downloader=None) -> dict:
    """Verify exact files anonymously, including adapter weights and collection."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi(token=False)
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download

    info = api.model_info(repo_id=MODEL_ID, token=False)
    if (getattr(info, "private", None) is not False
        or getattr(info, "gated", None) is not False
        or getattr(info, "sha", None) != PUBLISHED_MODEL_COMMIT):
        raise RuntimeError("Published native40 model not public, ungated, or pinned")
    published = [v.rfilename for v in (getattr(info, "siblings", None) or ())]
    if (len(published) != len(set(published))
        or set(published) not in (PUBLIC_FILES, PUBLIC_FILES | {".gitattributes"})):
        raise RuntimeError("Public model repo has unexpected/missing files")
    collection = api.get_collection(COLLECTION_SLUG, token=False)
    if getattr(collection, "private", None) is not False:
        raise RuntimeError("Model collection not verifiably public")
    items = list(getattr(collection, "items", []) or [])
    if not any(
        getattr(item, "item_id", None) == MODEL_ID
        and getattr(item, "item_type", None) == "model"
        for item in items
    ):
        raise RuntimeError("Public model collection has no matching native40 adapter")

    file_bytes = {}
    file_hashes = {}
    for filename in sorted(PUBLIC_FILES):
        path = Path(downloader(
            repo_id=MODEL_ID, filename=filename, repo_type="model",
            revision=PUBLISHED_MODEL_COMMIT, token=False,
        ))
        if not path.is_file() or not 0 < path.stat().st_size <= 11_000_000:
            raise RuntimeError("Public model file absent/oversized: " + filename)
        blob = path.read_bytes()
        file_bytes[filename] = blob
        file_hashes[filename] = sha256_bytes(blob)

    manifest = safe_json(file_bytes["RELEASE_MANIFEST.json"])
    declared = manifest.get("public_file_sha256")
    if (not isinstance(declared, dict)
        or set(declared) != PUBLIC_FILES - {"RELEASE_MANIFEST.json"}
        or any(
            not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
            for v in declared.values()
        )
        or any(file_hashes[name] != declared[name] for name in declared)):
        raise RuntimeError("Public model files differ from SHA256 release manifest")
    if (manifest.get("original_private_source_zip_sha256") != ORIGINAL_PRIVATE_SOURCE
        or manifest.get("source_base_weights_included") is not False
        or manifest.get("private_raw_model_logs_included") is not False
        or manifest.get("model_safety_improvement_claimed") is not False
        or file_hashes["adapter_model.safetensors"] != ADAPTER_SHA
        or file_hashes["adapter_config.json"] != CONFIG_SHA):
        raise RuntimeError("Original native adapter hash or safety scope not verified")
    metrics = safe_json(file_bytes["EVALUATION_SUMMARY.json"])
    if (metrics.get("case_count") != 48
        or metrics.get("base", {}).get("schema_valid") != 14
        or metrics.get("native_aligntune_adapter", {}).get("schema_valid") != 48
        or metrics.get("base", {}).get("unsupported_correct_abstentions") != 4
        or metrics.get("native_aligntune_adapter", {}).get(
            "unsupported_correct_abstentions"
        ) != 2
        or metrics.get("adapter_safety_improvement_claimed") is not False):
        raise RuntimeError("Public evaluation data don't match documented native trial")
    readme = file_bytes["README.md"].decode("utf-8")
    if (not readme.startswith("---\n")
        or "license: cc-by-nc-4.0" not in readme
        or "NOT a safety-improved model" not in readme
        or "CohereLabs/tiny-aya-global" not in readme):
        raise RuntimeError("Public adapter README missing license or safety warning")
    return {
        "status": "PUBLIC_NATIVE40_MODEL_AND_COLLECTION_ANONYMOUSLY_VERIFIED_NO_UPLOAD",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "model_repo_url": MODEL_URL,
        "model_collection_url": COLLECTION_URL,
        "model_revision": PUBLISHED_MODEL_COMMIT,
        "original_owner_zip_sha256": ORIGINAL_PRIVATE_SOURCE,
        "public_file_count": len(PUBLIC_FILES),
        "public_file_sha256": file_hashes,
        "public_collection_contains_expected_model": True,
        "public_ungated_model_verified": True,
        "all_public_files_sha256_verified": True,
        "raw_private_predictions_published": False,
        "gated_upstream_base_weights_published": False,
        "publication_attempted_in_this_check": False,
        "authentication_token_used": False,
        "license_rights_independently_approved": False,
        "model_semantic_safety_improvement_verified": False,
        "official_lexsi_submission_verified": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        help="Optional write-once local receipt; never overwrites evidence",
    )
    args = parser.parse_args()
    record = verify_existing_release()
    if args.output is not None:
        destination = args.output
        if destination.is_symlink() or destination.exists():
            parser.error("Receipt path already exists, choose a new file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as file:
            json.dump(record, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write("\n")
    print(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
