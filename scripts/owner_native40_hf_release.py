"""Owner-controlled, source-pinned, adapter-ONLY Hugging Face release.

Default mode PREPARES an offline ZIP -- no network / credentials / publication.
Remote publication can occur ONLY with explicit owner license & public-release
approvals, exact original archive SHA and authenticated HF account. Never
package gated Tiny Aya base weights or private original inference logs.

Owner instructions: docs/LEXSI_HF_NATIVE40_OWNER_PUBLICATION.md
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_SHA = "dcefc5b0babf708aa1b48ac0349ddb3bc4745405c72e9636c5860d7719d8f0e9"
WEIGHTS_SHA = "ba985c1966050764a5220e87ea581a9bd482ba87b055fee4a1e363643f1cc92e"
CONFIG_SHA = "808fea2999f91c2af09271b275877080ecf89683faec5e7a63f8000ec5d25621"
TRAIN_SHA = "4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c"
MODEL_REVISION_REQUESTED = "af89d219b53ed9b13b8a4645f9c8028973510324"
SOURCE_MEMBERS = {
    "adapter/adapter_model.safetensors",
    "adapter/adapter_config.json",
    "run_info.json",
}
PUBLIC_MEMBERS = {
    "adapter_model.safetensors", "adapter_config.json", "README.md",
    "EVALUATION_SUMMARY.json", "RELEASE_MANIFEST.json",
}
MODEL_REPO_NAME = "policyproof-ai-native40-align-2026"
MODEL_COLLECTION_TITLE = "PolicyProof AI — Lexsi Track 2 Model and Adapter Research"
MAX_ZIP_SIZE = 15_000_000
MAX_CARD_SIZE = 100_000


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _synthetic_metrics() -> dict:
    # Aggregate-only values from the archived owner dev48 paired report.
    # Never write the individual raw generations into a public model repo.
    return {
        "schema": "policyproof_public_aggregate_native40_dev48_v1",
        "source_private_native40_paired_bundle_sha256":
            "e0319c83fbb87ef8570ca0c42caf68bea4a4fef1f8ae320bbe11e9a7c1761c68",
        "dataset_type": "previously_inspected_synthetic_development_only",
        "case_count": 48, "supported_case_count": 39, "unsupported_case_count": 9,
        "base": {"schema_valid": 14, "strict_matches": 4,
                 "supported_answer_exact": 0, "raw_expected_source": 31,
                 "unsupported_correct_abstentions": 4,
                 "unsupported_nonabstaining_claims": 5},
        "native_aligntune_adapter": {
            "schema_valid": 48, "strict_matches": 2,
            "supported_answer_exact": 0, "raw_expected_source": 39,
            "unsupported_correct_abstentions": 2,
            "unsupported_nonabstaining_claims": 7,
        },
        "independent_semantic_correctness_verified": False,
        "train_time_base_revision_independently_verified": False,
        "GPU_execution_independently_attested": False,
        "adapter_safety_improvement_claimed": False,
        "private_prediction_logs_published": False,
        "notes": [
            "Matching a citation ID is not evidence of passage entailment",
            "Formatting improved but unsupported-question abstention regressed",
            "Small, correlated development set; no blind final evaluation",
        ],
    }


def _verify_source(archive_path: Path) -> tuple[bytes, bytes, dict]:
    archive_path = Path(archive_path)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("Original private adapter ZIP must be an ordinary file")
    if not 0 < archive_path.stat().st_size <= MAX_ZIP_SIZE:
        raise ValueError("Original ZIP missing or too large")
    original_digest = sha_bytes(archive_path.read_bytes())
    if original_digest != ORIGINAL_SHA:
        raise ValueError("Owner original ZIP SHA256 mismatch — refusing release")
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != SOURCE_MEMBERS or len(archive.infolist()) != 3:
            raise ValueError("Original adapter ZIP has missing/extra/duplicate members")
        if archive.testzip() is not None:
            raise ValueError("Original adapter ZIP CRC integrity failure")
        for member in archive.infolist():
            if stat.S_IFMT(member.external_attr >> 16) == stat.S_IFLNK:
                raise ValueError("ZIP symlink member is forbidden")
        weights = archive.read("adapter/adapter_model.safetensors")
        config = archive.read("adapter/adapter_config.json")
        receipt_raw = archive.read("run_info.json")
    if sha_bytes(weights) != WEIGHTS_SHA or sha_bytes(config) != CONFIG_SHA:
        raise ValueError("Saved adapter tensor/config hashes do not match frozen original")
    if len(receipt_raw) > 100_000:
        raise ValueError("Original native run receipt unexpectedly large")
    try:
        receipt = json.loads(receipt_raw)
        peft = json.loads(config)
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Original native receipt/config is invalid JSON") from exc
    if (not isinstance(receipt, dict) or
        receipt.get("trainer_backend") != "aligntune_wrapper" or
        receipt.get("optimizer_steps_completed") != 40 or
        receipt.get("aligntune_trainer_completed") is not True or
        receipt.get("claims_improved_safety") is not False or
        receipt.get("independent_gpu_execution_attested") is not False or
        receipt.get("dataset_sha256") != TRAIN_SHA or
        receipt.get("examples") != 78 or
        receipt.get("base_revision") != MODEL_REVISION_REQUESTED):
        raise ValueError("Original training provenance is missing/misrepresented")
    if (not isinstance(peft, dict) or
        peft.get("base_model_name_or_path") != "CohereLabs/tiny-aya-global" or
        peft.get("peft_type") != "LORA" or
        peft.get("r") != 8 or peft.get("lora_alpha") != 16 or
        set(peft.get("target_modules", [])) !=
        {"q_proj", "k_proj", "v_proj", "o_proj"}):
        raise ValueError("PEFT config is not the exact native Tiny Aya adapter")
    return weights, config, receipt


def build_release_bytes(
    archive_path: Path, *, card_path: Path | None = None,
) -> dict[str, bytes]:
    """Create exactly five public-safe files, no raw owner generations."""
    weights, config, _ = _verify_source(archive_path)
    card_path = Path(card_path or ROOT / "docs/HF_NATIVE40_ADAPTER_MODEL_CARD.md")
    if card_path.is_symlink() or not card_path.is_file():
        raise ValueError("Missing checked model-card source")
    card = card_path.read_bytes()
    if not 100 < len(card) <= MAX_CARD_SIZE or not card.startswith(b"---\n"):
        raise ValueError("Invalid HF model-card content")
    if b"NOT a safety-improved model" not in card:
        raise ValueError("Model card lacks required safety regression warning")
    metrics = (json.dumps(_synthetic_metrics(), sort_keys=True,
                          indent=2, ensure_ascii=False) + "\n").encode()
    release = {
        "adapter_model.safetensors": weights,
        "adapter_config.json": config,
        "README.md": card,
        "EVALUATION_SUMMARY.json": metrics,
    }
    manifest = {
        "schema": "policyproof_native40_public_adapter_release_v1",
        "original_private_source_zip_sha256": ORIGINAL_SHA,
        "original_training_dataset_sha256": TRAIN_SHA,
        "requested_train_time_base_revision": MODEL_REVISION_REQUESTED,
        "loaded_train_time_base_revision_independently_observed": False,
        "private_raw_model_logs_included": False,
        "source_base_weights_included": False,
        "semantic_correctness_independently_adjudicated": False,
        "publication_performed_by_preview": False,
        "model_safety_improvement_claimed": False,
        "public_file_sha256": {k: sha_bytes(v) for k, v in sorted(release.items())},
    }
    release["RELEASE_MANIFEST.json"] = (
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)+"\n"
    ).encode()
    if set(release) != PUBLIC_MEMBERS:
        raise ValueError("Unexpected release member attempted")
    return release


def write_offline_preview(
    archive_path: Path, output_zip: Path, *, card_path: Path | None = None,
) -> dict:
    """Only local write-once preview; it NEVER imports HF API or network."""
    release = build_release_bytes(archive_path, card_path=card_path)
    output_zip = Path(output_zip)
    if (output_zip.is_symlink() or output_zip.exists() or
        output_zip.suffix != ".zip" or
        "NOT_PUBLISHED" not in output_zip.name or
        not output_zip.parent.is_dir() or
        output_zip.parent.is_symlink()):
        raise FileExistsError("Choose new existing-dir ...NOT_PUBLISHED.zip preview path")
    with tempfile.NamedTemporaryFile(
        prefix=".native40-release-", suffix=".partial", dir=output_zip.parent,
        delete=False,
    ) as tmp:
        partial = Path(tmp.name)
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, blob in sorted(release.items()):
                info = zipfile.ZipInfo(name, (2026, 11, 10, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (0o644 << 16)
                z.writestr(info, blob)
        if output_zip.exists():
            raise FileExistsError("Refusing overwrite of prior preview")
        # Atomic hard-link creation prevents a simultaneous preview overwrite.
        import os
        os.link(partial, output_zip)
    finally:
        partial.unlink(missing_ok=True)
    return {
        "status": "OWNER_ONLY_PUBLIC_ADAPTER_PREVIEW_NOT_PUBLISHED",
        "preview_zip": str(output_zip),
        "preview_sha256": sha_bytes(output_zip.read_bytes()),
        "original_source_zip_sha256": ORIGINAL_SHA,
        "file_count": len(release),
        "public_model_repo_created": False,
        "collection_created": False,
        "hugging_face_token_used": False,
        "owner_rights_approved_by_this_tool": False,
    }


def publish_owner_approved(
    archive_path: Path, *, namespace: str,
    confirmed_source_sha256: str, approve_public: bool,
    approve_license: bool, approve_outputs: bool,
    api=None, anonymous_api=None, anonymous_downloader=None,
    card_path: Path | None = None,
) -> dict:
    """Explicit owner-only path; no publish without independent legal review.

    Cannot guarantee a complete transaction across HF model/collection APIs.
    If an upload fails after repo creation, it leaves a partial owner repo
    requiring manual owner inspection; it never claims successful publication.
    """
    if not (
        approve_public and approve_license and approve_outputs
        and confirmed_source_sha256 == ORIGINAL_SHA
    ):
        raise PermissionError(
            "Public adapter release blocked: review CC BY-NC + Cohere AUP, "
            "verify ownership of weights and aggregate data, authorize a "
            "PUBLIC ungated adapter-only repo and supply exact ZIP SHA."
        )
    if not isinstance(namespace, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,38}", namespace
    ):
        raise ValueError("Explicit legal Hugging Face account namespace required")
    release = build_release_bytes(archive_path, card_path=card_path)
    if api is None or anonymous_api is None:
        from huggingface_hub import HfApi
        api = api or HfApi()
        anonymous_api = anonymous_api or HfApi(token=False)

    identity = api.whoami(token=True)
    if identity.get("name", "").lower() != namespace.lower():
        raise PermissionError(
            "Authenticated Hugging Face identity does not match selected owner namespace"
        )
    repo_id = f"{namespace}/{MODEL_REPO_NAME}"
    # Owner-controlled, intentionally not exist_ok: never overwrite a
    # potentially different pre-existing public model repository.
    with tempfile.TemporaryDirectory(prefix="policyproof-hf-owner-") as td:
        folder = Path(td)
        for name, blob in release.items():
            (folder / name).write_bytes(blob)
        api.create_repo(
            repo_id=repo_id, repo_type="model", private=False, exist_ok=False,
            token=True,
        )
        api.upload_folder(
            repo_id=repo_id, repo_type="model", folder_path=str(folder),
            allow_patterns=sorted(PUBLIC_MEMBERS),
            token=True,
            commit_message="Publish original source-hashed PolicyProof research adapter",
        )
    collection = api.create_collection(
        title=MODEL_COLLECTION_TITLE, namespace=namespace,
        description="Track 2 research: native AlignTune adapter and its measured limitations",
        private=False, exists_ok=True, token=True,
    )
    api.add_collection_item(
        collection_slug=collection.slug, item_id=repo_id, item_type="model",
        exists_ok=True, token=True,
    )
    # Independent ANONYMOUS readback: mere upload success is not enough.
    info = anonymous_api.model_info(repo_id=repo_id, token=False)
    if getattr(info, "private", None) is not False or getattr(info, "gated", False):
        raise RuntimeError("Model repository not confirmed public and ungated")
    names = {v.rfilename for v in (getattr(info, "siblings", None) or [])}
    if (not PUBLIC_MEMBERS.issubset(names)
        or names - PUBLIC_MEMBERS - {".gitattributes"}):
        raise RuntimeError("Anonymous model repo file list differs from release manifest")
    if anonymous_downloader is None:
        from huggingface_hub import hf_hub_download
        anonymous_downloader = hf_hub_download
    for filename, original in sorted(release.items()):
        file_path = Path(anonymous_downloader(
            repo_id=repo_id, filename=filename, repo_type="model", token=False,
        ))
        if not file_path.is_file() or sha_bytes(file_path.read_bytes()) != sha_bytes(original):
            raise RuntimeError("Anonymous model repo file SHA256 mismatch: " + filename)
    collection_public = anonymous_api.get_collection(collection.slug, token=False)
    items = getattr(collection_public, "items", None) or []
    if not any(getattr(item, "item_id", None) == repo_id for item in items):
        raise RuntimeError("Public model collection missing uploaded adapter repo")
    return {
        "status": "PUBLIC_ADAPTER_REPO_AND_MODEL_COLLECTION_ANONYMOUSLY_VERIFIED",
        "model_repo_url": f"https://huggingface.co/{repo_id}",
        "model_collection_url": f"https://huggingface.co/collections/{collection.slug}",
        "original_source_sha256": ORIGINAL_SHA,
        "raw_private_predictions_uploaded": False,
        "upstream_base_weights_uploaded": False,
        "human_semantic_safety_claim": False,
        "model_became_safer_claim": False,
        "owner_official_Lexsi_submission_done": False,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["preview", "publish"])
    p.add_argument("--archive", required=True)
    p.add_argument("--preview-zip", default="artifacts/policyproof-native40-HF-PREVIEW-NOT_PUBLISHED.zip")
    p.add_argument("--namespace", default="shalnark01")
    p.add_argument("--confirmed-source-sha256")
    p.add_argument("--approve-public", action="store_true")
    p.add_argument("--approve-license", action="store_true")
    p.add_argument("--approve-outputs", action="store_true")
    args = p.parse_args()
    if args.action == "preview":
        result = write_offline_preview(Path(args.archive), Path(args.preview_zip))
    else:
        result = publish_owner_approved(
            Path(args.archive), namespace=args.namespace,
            confirmed_source_sha256=args.confirmed_source_sha256 or "",
            approve_public=args.approve_public,
            approve_license=args.approve_license,
            approve_outputs=args.approve_outputs,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
