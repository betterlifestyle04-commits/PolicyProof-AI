"""Verify a PEFT adapter *artifact* on CPU before trusting training metadata.

This checks safetensors structure, on-disk checksums and saved provenance.
It CANNOT independently prove a GPU ran, optimizer steps happened or the
model learned. In particular, a manual PyTorch fallback is not an executed
AlignTune trainer and cannot satisfy the Track 1 trainer requirement.

Run:
  python -m scripts.verify_training_archive --archive /path/to/smoke.zip
  python -m scripts.verify_training_archive --directory artifacts/align_qlora_smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

MODEL = "CohereLabs/tiny-aya-global"
CURATOR_TRAIN_SHA256 = "4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c"
SHA = re.compile(r"^[0-9a-f]{64}$")
REVISION = re.compile(r"^[0-9a-f]{40}$")
SINGLE_FILE_LIMIT = 100_000_000
ARCHIVE_LIMIT = 220_000_000
SAFE_TENSOR_HEADER_LIMIT = 1_000_000
EXPECTED_TRAINABLE_TENSORS = 288
EXPECTED_TRAINABLE_PARAMETERS = 3_833_856
DTYPE_SIZES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _parse_json(payload: bytes, description: str) -> dict:
    try:
        parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(description + ": invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(description + ": expected a JSON object")
    return parsed


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def validate_safetensors(path: Path) -> dict:
    """Validate format, tensor byte ranges and LoRA A/B presence; no torch used."""
    if path.stat().st_size > SINGLE_FILE_LIMIT or path.stat().st_size < 32:
        raise ValueError("Invalid adapter safetensors file size")
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length < 2 or length > SAFE_TENSOR_HEADER_LIMIT:
            raise ValueError("Invalid safetensors header length")
        header = _parse_json(handle.read(length), "safetensors header")
    data_size = path.stat().st_size - 8 - length
    if data_size <= 0:
        raise ValueError("No tensor data saved")
    spans = []
    tensor_names = []
    parameter_count = 0
    for name, meta in header.items():
        if name == "__metadata__":
            if not isinstance(meta, dict) or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in meta.items()
            ):
                raise ValueError("Invalid safetensors metadata")
            continue
        if not isinstance(name, str) or not name or not isinstance(meta, dict):
            raise ValueError("Invalid tensor entry")
        shape, offsets, dtype = (
            meta.get("shape"), meta.get("data_offsets"), meta.get("dtype")
        )
        if dtype not in DTYPE_SIZES:
            raise ValueError("Unknown tensor dtype")
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 8
            or any(type(dim) is not int or not 1 <= dim <= SINGLE_FILE_LIMIT
                   for dim in shape)):
            raise ValueError("Invalid tensor shape")
        if (not isinstance(offsets, list) or len(offsets) != 2
            or any(type(o) is not int or o < 0 for o in offsets)
            or not offsets[0] < offsets[1] <= data_size):
            raise ValueError("Invalid tensor offsets")
        count = math.prod(shape)
        parameter_count += count
        if count > SINGLE_FILE_LIMIT or count * DTYPE_SIZES[dtype] != offsets[1] - offsets[0]:
            raise ValueError("Tensor dimensions do not match stored bytes")
        spans.append(tuple(offsets))
        tensor_names.append(name)
    if not tensor_names or not any("lora_A" in n for n in tensor_names) or not any(
        "lora_B" in n for n in tensor_names
    ):
        raise ValueError("Adapter lacks LoRA A/B tensor names")
    last = 0
    for start, end in sorted(spans):
        if start != last:
            raise ValueError("Overlapping or missing tensor bytes")
        last = end
    if last != data_size:
        raise ValueError("Unclaimed trailing tensor bytes")
    return {
        "tensor_count": len(tensor_names), "parameter_count": parameter_count,
        "data_bytes": data_size, "sha256": digest(path),
    }


def inspect_training_directory(directory: Path) -> dict:
    """Read-only receipt inspection. Never equate self-reports to real training."""
    directory = Path(directory)
    info_path = directory / "run_info.json"
    config_path = directory / "adapter" / "adapter_config.json"
    weights_path = directory / "adapter" / "adapter_model.safetensors"
    if not all(p.is_file() and not p.is_symlink()
               for p in (info_path, config_path, weights_path)):
        raise ValueError("run_info, PEFT configuration and safe adapter weights required")
    if info_path.stat().st_size > 1_000_000 or config_path.stat().st_size > 1_000_000:
        raise ValueError("Oversized training metadata")
    info = _parse_json(info_path.read_bytes(), "training receipt")
    config = _parse_json(config_path.read_bytes(), "PEFT adapter config")
    if info.get("status") != "completed" or info.get("model") != MODEL:
        raise ValueError("No completed Tiny Aya training receipt")
    steps = info.get("optimizer_steps", info.get("optimizer_steps_completed"))
    if type(steps) is not int or steps < 10 or steps > 1_000_000:
        raise ValueError("At least ten declared optimizer steps required")
    if info.get("examples") != 78 or info.get("dataset_sha256") != CURATOR_TRAIN_SHA256:
        raise ValueError("Training data do not match audited CuratorKIT 78-row set")
    if config.get("peft_type") != "LORA" or (
        type(config.get("r")) is not int or config["r"] <= 0
    ):
        raise ValueError("Invalid LoRA adapter configuration")
    tensor = validate_safetensors(weights_path)
    declared = info.get("adapter_sha256")
    if (not isinstance(declared, dict)
        or declared.get(weights_path.name) != tensor["sha256"]):
        raise ValueError("Adapter weights hash mismatch or missing")
    config_hash = digest(config_path)
    if info.get("adapter_config_sha256") != config_hash:
        raise ValueError("PEFT config hash mismatch or missing")
    commit = info.get("code_commit", info.get("code_revision"))
    if not isinstance(commit, str) or not REVISION.fullmatch(commit):
        raise ValueError("Full 40-character code commit required")
    backend = info.get("trainer_backend")
    if backend is None and info.get("kind") == "10-step QLoRA integration smoke test":
        backend = "aligntune_wrapper"
    if backend not in {
        "aligntune_wrapper", "manual_torch_smoke_fallback",
        "manual_torch_extended_fallback",
    }:
        raise ValueError("Unexpected/unreported training backend")
    if backend.startswith("manual_torch_") and info.get(
        "aligntune_trainer_completed"
    ) is not False:
        raise ValueError("Manual fallback must explicitly disclaim AlignTune training")
    if backend == "manual_torch_extended_fallback":
        plan = info.get("training_plan")
        if (steps != 40 or not isinstance(plan, dict)
            or plan.get("steps") != 40 or plan.get("seed") != 42
            or plan.get("validation_on_training_source_only") is not True
            or plan.get("final_benchmark_or_challenge_used_for_training") is not False):
            raise ValueError("Extended PyTorch run lacks a valid training-only schedule")
        from scripts.t4_torch_fallback import training_plan
        expected = training_plan(steps=40)
        if (plan.get("train_indices") != expected["train_indices"]
            or plan.get("validation_indices") != expected["validation_indices"]):
            raise ValueError("Extended run training/validation split does not match frozen schedule")
        validation = info.get("training_only_validation_losses")
        if (not isinstance(validation, list) or len(validation) != 4
            or [x.get("step") for x in validation if isinstance(x, dict)]
               != [10, 20, 30, 40]
            or any(
                not isinstance(item, dict)
                or type(item.get("training_only_validation_loss")) not in (float, int)
                or not math.isfinite(item["training_only_validation_loss"])
                or item["training_only_validation_loss"] < 0
                for item in validation
            )):
            raise ValueError("Extended training-only validation losses incomplete or invalid")
    revision = info.get("model_revision", info.get("base_revision"))
    pinned_base = isinstance(revision, str) and REVISION.fullmatch(revision) is not None
    if backend == "manual_torch_extended_fallback" or (
        backend == "aligntune_wrapper" and steps == 40
    ):
        if backend == "aligntune_wrapper" and (
            info.get("kind") != "40-step native AlignTune QLoRA trial"
            or info.get("requested_optimizer_steps") != 40
            or info.get("training_engine") != "AlignTune TRL wrapper.train()"
            or info.get("final_benchmark_or_challenge_used_for_training") is not False
            or info.get("independent_gpu_execution_attested") is not False
        ):
            raise ValueError("Native 40-step trainer receipt lacks an approved provenance shape")
        evidence = info.get("loaded_revision_evidence")
        observed = evidence.get("loaded_model_revision_observed") if isinstance(evidence, dict) else None
        confirmed = evidence.get("loaded_model_revision_verified") if isinstance(evidence, dict) else None
        # A remote Hub model_info SHA alone cannot authenticate the weights
        # actually loaded in Colab. No observed _commit_hash => unverified.
        if observed is not None and (
            not isinstance(observed, str) or not REVISION.fullmatch(observed)
            or observed != revision
        ):
            raise ValueError("Loaded extended-model revision conflicts with declared base")
        if confirmed is True and observed != revision:
            raise ValueError("Claimed verified model revision lacks matching observed SHA")
        pinned_base = bool(pinned_base and confirmed is True and observed == revision)
    aligntune_path = backend == "aligntune_wrapper"
    expected_geometry = (
        tensor["tensor_count"] == EXPECTED_TRAINABLE_TENSORS
        and tensor["parameter_count"] == EXPECTED_TRAINABLE_PARAMETERS
    )
    return {
        "status": "ARTIFACT_INTEGRITY_CHECKED",
        "file_hashes_match": True,
        "safetensors_structure_valid": True,
        "optimizer_steps_self_reported": steps,
        "original_tiny_aya_model": MODEL,
        "curated_train_sha256": CURATOR_TRAIN_SHA256,
        "adapter_sha256": tensor["sha256"],
        "adapter_config_sha256": config_hash,
        "tensor_count": tensor["tensor_count"],
        "parameter_count": tensor["parameter_count"],
        "expected_tiny_aya_lora_geometry": expected_geometry,
        "backend": backend,
        "base_revision_pinned": pinned_base,
        "track1_trainer_path_declared": aligntune_path,
        "native_aligntune_40step_receipt_checked": bool(
            aligntune_path and steps == 40
        ),
        "base_revision_loaded_evidence_verified": bool(
            aligntune_path and steps == 40 and pinned_base
        ),
        "track1_submission_evidence_candidate": bool(
            pinned_base and aligntune_path and expected_geometry
        ),
        "training_independently_verified": False,
        "model_improvement_verified": False,
        "warning": "Hash + format validation confirms artifact consistency only. "
                   "Optimizer steps and backend are still self-reported; inspect "
                   "GPU/AlignTune logs, exact source commit and independent evaluation.",
    }


def inspect_training_archive(archive: Path) -> dict:
    """Bounded ZIP inspection in a temporary directory; never extract blindly."""
    archive = Path(archive)
    if not archive.is_file() or archive.stat().st_size > ARCHIVE_LIMIT:
        raise ValueError("Missing or oversized training ZIP")
    with ZipFile(archive) as zipped, tempfile.TemporaryDirectory() as temp:
        members = [m for m in zipped.infolist() if not m.is_dir()]
        if len(members) < 3 or len(members) > 60:
            raise ValueError("Unexpected training archive member count")
        seen = set()
        if sum(m.file_size for m in members) > ARCHIVE_LIMIT:
            raise ValueError("Excessive decompressed training ZIP size")
        for member in members:
            name = member.filename
            path = PurePosixPath(name)
            if (not path.parts or name in seen or name.startswith("/") or "\\" in name
                or ".." in path.parts
                or (path.parts[0] == "adapter" and len(path.parts) != 2)
                or (path.parts[0] in {"run_info.json", "run_info.failed_encoding_typeerror.json"}
                    and len(path.parts) != 1)
                or path.parts[0] not in {
                    "adapter", "run_info.json", "run_info.failed_encoding_typeerror.json"
                }
                or member.file_size > SINGLE_FILE_LIMIT
                or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe or unsupported training ZIP member")
            seen.add(name)
            target = Path(temp).joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("xb") as output:
                while block := source.read(1024 * 1024):
                    output.write(block)
        result = inspect_training_directory(Path(temp))
    return {
        **result,
        "archive_sha256": digest(archive),
        "archive_member_count": len(members),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only PEFT artifact verification")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive")
    group.add_argument("--directory")
    parser.add_argument("--strict-track1", action="store_true",
                        help="Require pinned base revision and AlignTune trainer receipt")
    args = parser.parse_args()
    result = (inspect_training_archive(Path(args.archive)) if args.archive
              else inspect_training_directory(Path(args.directory)))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.strict_track1 and not result["track1_submission_evidence_candidate"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
