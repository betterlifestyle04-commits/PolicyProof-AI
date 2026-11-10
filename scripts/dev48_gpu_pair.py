"""Real base-versus-adapter GPU inference on the INSPECTED 48-case development set.

This is development DIAGNOSTICS ONLY: previous saved Tiny Aya base responses
have an unknown model revision and MUST NOT be reused as a paired baseline.
Generate a FRESH base and tuned pass at the same exact requested HF revision.

The script defaults to CPU-only, read-only preflight; --execute explicitly
authorizes an actual T4/P100 GPU model download and two sequential passes.
Outputs are exclusive-create, never overwrite old data, adapter, or research.
No final 60-case benchmark, 72-case unscored challenge, or training source is
read for inference. No human semantic improvement is asserted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

from policyproof.core import ROOT, aggregate, read_jsonl, score, sha256
from scripts.auditkit_compare import comparison
from scripts.frozen_pair_inference import _predict_on_gpu, GENERATION, MODEL_ID
from scripts.paired_run_recovery import write_evidence_once
from scripts.verify_training_archive import (
    digest, inspect_training_archive, inspect_training_directory,
)

DEV_REL = "data/unseen_policy_test.jsonl"
DEV_SHA = "d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c"
REVISION = re.compile(r"^[0-9a-f]{40}$")
SHA = re.compile(r"^[0-9a-f]{64}$")
SPLIT = "unseen-policy"
EXPECTED_CATEGORIES = {
    "unseen_supported": 36,
    "unseen_unanswerable": 9,
    "context_injection": 3,
}
MAX_PUBLIC_RESULT_BYTES = 4_000_000


def check_dev_rows(data: Path) -> tuple[list[dict], str]:
    if not data.is_file() or data.is_symlink() or sha256(data) != DEV_SHA:
        raise ValueError(
            "48-case development data missing or changed; run "
            "python -m scripts.prepare then recheck."
        )
    rows = read_jsonl(data)
    if (len(rows) != 48 or len({r.get("id") for r in rows}) != 48
        or Counter(r.get("language") for r in rows) !=
           {"en":16, "no":16, "ar":16}
        or Counter(r.get("category") for r in rows) != EXPECTED_CATEGORIES
        or any(r.get("split") != SPLIT for r in rows)):
        raise ValueError("Development candidate inventory differs from original inspected 48 cases")
    return rows, DEV_SHA


def check_training_zip(
    archive: Path, trusted_zip_sha: str, *,
    expected_backend: str = "manual_torch_extended_fallback",
) -> dict:
    if expected_backend not in {"manual_torch_extended_fallback", "aligntune_wrapper"}:
        raise ValueError("Unsupported expected training backend")
    archive = Path(archive)
    if not isinstance(trusted_zip_sha, str) or not SHA.fullmatch(trusted_zip_sha):
        raise ValueError("Require exact user-verified 64-hex original archive SHA256")
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("Original trained checkpoint ZIP missing or symlinked")
    if digest(archive) != trusted_zip_sha:
        raise ValueError("Training ZIP does not match the owner's pinned SHA256")
    report = inspect_training_archive(archive)
    if (report["archive_sha256"] != trusted_zip_sha
        or report.get("backend") != expected_backend
        or report.get("optimizer_steps_self_reported") != 40
        or report.get("expected_tiny_aya_lora_geometry") is not True
        or report.get("file_hashes_match") is not True):
        raise ValueError("Original 40-step PEFT archive was not verified for requested backend")
    # Reads only a known, fixed small metadata member. The full ZIP was
    # structurally checked and its size bounded by the archive validator.
    with zipfile.ZipFile(archive) as z:
        receipt = json.loads(z.read("run_info.json"))
    revision = receipt.get("model_revision", receipt.get("base_revision"))
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise ValueError("Adapter lacks a full claimed HF base revision")
    if expected_backend == "aligntune_wrapper":
        if (report.get("native_aligntune_40step_receipt_checked") is not True
            or receipt.get("trainer_backend") != "aligntune_wrapper"
            or receipt.get("kind") != "40-step native AlignTune QLoRA trial"
            or receipt.get("aligntune_trainer_completed") is not True
            or receipt.get("independent_gpu_execution_attested") is not False
            or receipt.get("claims_improved_safety") is not False
            or receipt.get("loaded_revision_independently_verified") is not False):
            raise ValueError(
                "Native AlignTune source lacks verified receipt shape or truthful disclaimers"
            )
    elif receipt.get("performance_improvement_verified") is not False:
        raise ValueError("Unwarranted improvement claim in adapter receipt")
    return {
        "archive_sha256": trusted_zip_sha,
        "model_revision_claimed": revision,
        "adapter_sha256": report["adapter_sha256"],
        "optimizer_steps_self_reported": 40,
        "backend": report["backend"],
        "loaded_model_revision_verified_from_training": report["base_revision_pinned"],
        "training_independently_verified": False,
    }


def preflight(root: Path, archive: Path, trusted_zip_sha: str, *,
              expected_backend: str = "manual_torch_extended_fallback") -> dict:
    rows, data_sha = check_dev_rows(Path(root) / DEV_REL)
    adapter = check_training_zip(
        archive, trusted_zip_sha, expected_backend=expected_backend
    )
    return {
        "status": "DEV48_GPU_EVALUATION_PREPARED_NOT_EXECUTED",
        "cases": len(rows),
        "dataset_sha256": data_sha,
        "split": SPLIT,
        "training_zip": adapter,
        "generation": dict(GENERATION),
        "fresh_base_inference_required": True,
        "historical_saved_base_allowed_as_pair": False,
        "independent_human_semantic_evaluation": False,
        "model_inference_performed": False,
        "model_improvement_verified": False,
    }


def safely_stage_zip(archive: Path, target: Path) -> None:
    """Extract previously validated archive via bounded, safe paths only.

    Every staged adapter hash is verified again before the model is touched.
    """
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.infolist() if not m.is_dir()]
        if len(members) > 60 or sum(m.file_size for m in members) > 220_000_000:
            raise ValueError("Unsafe checkpoint member count or decompressed bytes")
        seen = set()
        for m in members:
            path = PurePosixPath(m.filename)
            name = m.filename
            if (not path.parts or name in seen or name.startswith("/")
                or chr(92) in name or ".." in path.parts
                or any(part in ("", ".") for part in name.split("/"))
                or (m.external_attr >> 16) & 0o170000 == 0o120000
                or m.file_size > 100_000_000
                or (path.parts[0] == "adapter" and len(path.parts) != 2)
                or path.parts[0] not in {
                    "adapter", "run_info.json",
                    "run_info.failed_encoding_typeerror.json",
                }):
                raise ValueError("Unsafe checkpoint archive member")
            seen.add(name)
            dest = target.joinpath(*path.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(m) as src, dest.open("xb") as output:
                while chunk := src.read(1024 * 1024):
                    output.write(chunk)
    inspect_training_directory(target)


def make_dev_report(
    rows: list[dict], predictions: dict[str, str], *,
    revision: str, data_sha: str, adapter_sha: str | None,
    zip_sha: str, gpu: dict,
) -> dict:
    if (len(rows) != 48 or len(predictions) != 48
        or set(predictions) != {r["id"] for r in rows}
        or any(not isinstance(s, str) or len(s) > 10000 for s in predictions.values())):
        raise ValueError("Incomplete or invalid 48-case real predictions")
    scores = [score(r, predictions[r["id"]]) for r in rows]
    return {
        "project": "PolicyProof AI / Lexsi 2026",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "transformers",
        "is_real_model_result": True,
        "is_finetuned": adapter_sha is not None,
        "model": MODEL_ID,
        "model_revision": revision,
        "adapter_sha256": adapter_sha,
        "adapter_source_zip_sha256": zip_sha if adapter_sha else None,
        "split": SPLIT,
        "dataset_sha256": data_sha,
        "generation_config": dict(GENERATION),
        "actual_gpu_inference_metadata": gpu,
        "human_semantic_review_completed": False,
        "split_is_blind_final_test": False,
        "training_performed_in_this_process": False,
        "warning": (
            "Already-inspected synthetic development set only. Exact-string "
            "scoring and source-ID overlap are not semantic correctness or "
            "generalized safety. No training occurred during inference."
        ),
        "metrics": aggregate(scores),
        "scores": scores,
        "predictions": [{"id": r["id"], "raw": predictions[r["id"]]} for r in rows],
    }


def read_validated_report(
    path: Path, rows: list[dict], *,
    revision: str, data_sha: str, adapter_sha: str | None,
    zip_sha: str,
) -> dict:
    path = Path(path)
    if (not path.is_file() or path.is_symlink()
        or not 0 < path.stat().st_size <= MAX_PUBLIC_RESULT_BYTES):
        raise ValueError("Saved dev48 prediction report is missing or unsafe")
    def unique(pairs):
        out = {}
        for key, val in pairs:
            if key in out:
                raise ValueError("Duplicate field in saved dev report: " + key)
            out[key] = val
        return out
    data = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique,
        parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)),
    )
    expected = {
        "engine":"transformers", "is_real_model_result":True,
        "is_finetuned":adapter_sha is not None, "model":MODEL_ID,
        "model_revision":revision,"adapter_sha256":adapter_sha,
        "adapter_source_zip_sha256":zip_sha if adapter_sha else None,
        "split":SPLIT,"dataset_sha256":data_sha,
        "generation_config":GENERATION,
        "human_semantic_review_completed":False,"split_is_blind_final_test":False,
    }
    if not isinstance(data, dict):
        raise ValueError("Saved report is not a JSON object")
    for key,value in expected.items():
        if type(value) is bool and type(data.get(key)) is not bool:
            raise ValueError("Ambiguous boolean in dev report: "+key)
        if data.get(key) != value:
            raise ValueError("Saved report provenance mismatch: "+key)
    gpu = data.get("actual_gpu_inference_metadata")
    if not isinstance(gpu, dict) or not isinstance(gpu.get("gpu"), str) or not gpu["gpu"]:
        raise ValueError("Missing reported real GPU inference metadata")
    records = data.get("predictions")
    if not isinstance(records, list) or len(records) != 48:
        raise ValueError("Incomplete raw model predictions")
    outputs = {}
    for item in records:
        if (not isinstance(item,dict) or not isinstance(item.get("id"),str)
            or not isinstance(item.get("raw"),str) or len(item["raw"]) > 10000
            or item["id"] in outputs):
            raise ValueError("Invalid or duplicate saved raw prediction")
        outputs[item["id"]] = item["raw"]
    if set(outputs) != {r["id"] for r in rows}:
        raise ValueError("Saved model case IDs differ from dataset")
    scores = [score(r,outputs[r["id"]]) for r in rows]
    if data.get("scores") != scores or data.get("metrics") != aggregate(scores):
        raise ValueError("Saved output metrics do not match recomputed scores")
    return data


def audit_existing(
    root: Path, archive: Path, trusted_zip_sha: str,
    base_file: Path, tuned_file: Path, paired_file: Path, *,
    expected_backend: str = "manual_torch_extended_fallback",
) -> dict:
    evidence = preflight(
        root, archive, trusted_zip_sha, expected_backend=expected_backend
    )
    rows, data_sha = check_dev_rows(Path(root)/DEV_REL)
    revision = evidence["training_zip"]["model_revision_claimed"]
    adapter_sha = evidence["training_zip"]["adapter_sha256"]
    base = read_validated_report(
        base_file, rows, revision=revision, data_sha=data_sha,
        adapter_sha=None, zip_sha=trusted_zip_sha,
    )
    tuned = read_validated_report(
        tuned_file, rows, revision=revision, data_sha=data_sha,
        adapter_sha=adapter_sha, zip_sha=trusted_zip_sha,
    )
    if Path(paired_file).exists() or Path(paired_file).is_symlink():
        raise FileExistsError("Paired dev48 evidence exists; no overwrite")
    result = comparison(base,tuned,rows,data_sha)
    result.update({
        "split":SPLIT,
        "is_final_blind_test":False,
        "is_model_improvement_semantically_verified":False,
        "base_model_revision_loaded_on_inference":revision,
        "training_zip_sha256":trusted_zip_sha,
        "training_backend_declared":evidence["training_zip"]["backend"],
        "training_loaded_revision_independently_verified":
            evidence["training_zip"]["loaded_model_revision_verified_from_training"],
        "generation_config":dict(GENERATION),
        "gpu_inference_executed_in_this_audit":False,
        "not_comparable_to_historical_unpinned_48_base":True,
        "limitations":result["limitations"]+[
            "The inspected development set is not a blind final benchmark",
            "The model revision actually loaded during training was not independently pinned",
            "This is exact/structural scoring, not evidence-grounded human semantic review",
        ],
    })
    write_evidence_once(paired_file,result)
    return result


def execute_pair(
    root: Path, archive: Path, trusted_zip_sha: str,
    base_file: Path, tuned_file: Path, paired_file: Path,
    *, resume_base: bool = False,
    expected_backend: str = "manual_torch_extended_fallback",
) -> dict:
    evidence = preflight(
        root, archive, trusted_zip_sha, expected_backend=expected_backend
    )
    base_file,tuned_file,paired_file = map(Path,(base_file,tuned_file,paired_file))
    if len({p.resolve() for p in (base_file,tuned_file,paired_file)}) != 3:
        raise ValueError("Three independent output paths are mandatory")
    if paired_file.exists() or tuned_file.exists():
        raise FileExistsError("Existing tuned or paired predictions; refuse overwrite")
    if resume_base:
        if not base_file.is_file():
            raise FileNotFoundError("No saved complete original base pass to resume")
    elif base_file.exists():
        raise FileExistsError("Saved base exists; use explicit --resume-base")
    protected = {
        (Path(root)/DEV_REL).resolve(),
        archive.resolve(),
    }
    if any(p.resolve() in protected for p in (base_file,tuned_file,paired_file)):
        raise ValueError("Output would overwrite source evidence")
    rows,data_sha = check_dev_rows(Path(root)/DEV_REL)
    revision = evidence["training_zip"]["model_revision_claimed"]
    adapter_sha = evidence["training_zip"]["adapter_sha256"]
    if resume_base:
        base = read_validated_report(
            base_file,rows,revision=revision,data_sha=data_sha,
            adapter_sha=None,zip_sha=trusted_zip_sha,
        )
    else:
        outputs,gpu = _predict_on_gpu(rows,revision)
        base = make_dev_report(
            rows,outputs,revision=revision,data_sha=data_sha,adapter_sha=None,
            zip_sha=trusted_zip_sha,gpu=gpu,
        )
        if sha256(Path(root)/DEV_REL) != data_sha:
            raise ValueError("Development cases changed during base inference")
        write_evidence_once(base_file,base)
    # Verify ZIP and staged adapter AGAIN after the expensive base generation;
    # a substituted or changed archive cannot silently become tuned evidence.
    if digest(archive) != trusted_zip_sha:
        raise ValueError("Adapter ZIP changed during base inference")
    with tempfile.TemporaryDirectory(prefix="policyproof_dev48_adapter_") as temp:
        staged = Path(temp)
        safely_stage_zip(archive,staged)
        outputs,gpu = _predict_on_gpu(
            rows,revision,adapter_directory=staged/"adapter",
        )
    tuned = make_dev_report(
        rows,outputs,revision=revision,data_sha=data_sha,
        adapter_sha=adapter_sha,zip_sha=trusted_zip_sha,gpu=gpu,
    )
    if sha256(Path(root)/DEV_REL) != data_sha:
        raise ValueError("Development cases changed during tuned inference")
    write_evidence_once(tuned_file,tuned)
    return audit_existing(
        root, archive, trusted_zip_sha, base_file, tuned_file, paired_file,
        expected_backend=expected_backend,
    )


def create_results_bundle(
    base_file: Path,tuned_file: Path,paired_file: Path,bundle: Path,
) -> dict:
    inputs = {"dev48_base.json":Path(base_file),
              "dev48_tuned.json":Path(tuned_file),
              "dev48_paired_auditkit.json":Path(paired_file)}
    if any(not p.is_file() or p.is_symlink() for p in inputs.values()):
        raise ValueError("All three original result files required")
    bundle = Path(bundle)
    if bundle.exists() or bundle.is_symlink():
        raise FileExistsError("Results bundle already exists; never overwrite")
    if len({p.resolve() for p in inputs.values()}) != 3:
        raise ValueError("Distinct original base, tuned and comparison files required")
    payloads = {name:path.read_bytes() for name,path in inputs.items()}
    if any(not 0 < len(blob) <= MAX_PUBLIC_RESULT_BYTES for blob in payloads.values()):
        raise ValueError("Oversized or empty prediction report")
    try:
        base = json.loads(payloads["dev48_base.json"])
        tuned = json.loads(payloads["dev48_tuned.json"])
        paired = json.loads(payloads["dev48_paired_auditkit.json"])
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ValueError("Cannot bundle malformed evaluation records") from exc
    if (not all(isinstance(obj, dict) for obj in (base,tuned,paired))
        or base.get("is_real_model_result") is not True
        or base.get("is_finetuned") is not False
        or tuned.get("is_real_model_result") is not True
        or tuned.get("is_finetuned") is not True
        or base.get("split") != SPLIT or tuned.get("split") != SPLIT
        or base.get("dataset_sha256") != DEV_SHA
        or tuned.get("dataset_sha256") != DEV_SHA
        or not isinstance(base.get("model_revision"), str)
        or not REVISION.fullmatch(base["model_revision"])
        or tuned.get("model_revision") != base["model_revision"]
        or tuned.get("generation_config") != base.get("generation_config")
        or base.get("generation_config") != GENERATION
        or len(base.get("predictions", [])) != 48
        or len(tuned.get("predictions", [])) != 48
        or paired.get("status") != "paired_deterministic_audit"
        or paired.get("sample_count") != 48
        or paired.get("evaluation_checksum") != DEV_SHA
        or paired.get("split") != SPLIT
        or paired.get("is_final_blind_test") is not False
        or paired.get("is_model_improvement_semantically_verified") is not False
        or paired.get("base_model_revision_loaded_on_inference") != base["model_revision"]
        or paired.get("adapter_sha256") != tuned.get("adapter_sha256")
        or paired.get("training_zip_sha256") != tuned.get("adapter_source_zip_sha256")
        or paired.get("not_comparable_to_historical_unpinned_48_base") is not True):
        raise ValueError("Results bundle is not a complete internally consistent dev48 paired evaluation")
    manifest = {
        "schema":"policyproof_dev48_gpu_pair_bundle_v1",
        "split":"already_inspected_synthetic_development_only",
        "final_blind_evaluation":False,
        "model_improvement_semantically_verified":False,
        "files_sha256": {
            name:__import__("hashlib").sha256(blob).hexdigest()
            for name,blob in sorted(payloads.items())
        },
    }
    payloads["BUNDLE_MANIFEST.json"] = (
        json.dumps(manifest,indent=2,ensure_ascii=False)+"\n"
    ).encode("utf-8")
    bundle.parent.mkdir(parents=True,exist_ok=True)
    with bundle.open("xb") as output:
        with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as z:
            for name,blob in sorted(payloads.items()):
                z.writestr(name,blob)
    return {"status":"DEV48_PAIREDBUNDLE_CREATED",
            "archive":str(bundle),"sha256":digest(bundle),"files":len(payloads)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--expected-archive-sha256",required=True)
    p.add_argument("--adapter-backend", default="manual_torch_extended_fallback",
                   choices=["manual_torch_extended_fallback", "aligntune_wrapper"],
                   help="Explicit native-vs-fallback provenance; never auto-detect as a successful trial")
    p.add_argument("--execute",action="store_true")
    p.add_argument("--confirm-gpu-inference",action="store_true")
    p.add_argument("--resume-base",action="store_true")
    p.add_argument("--audit-existing",action="store_true")
    p.add_argument("--make-bundle",action="store_true")
    p.add_argument("--base",type=Path,default=Path("artifacts/dev48_paired_base.json"))
    p.add_argument("--tuned",type=Path,default=Path("artifacts/dev48_paired_tuned.json"))
    p.add_argument("--paired",type=Path,default=Path("artifacts/dev48_paired_auditkit.json"))
    p.add_argument("--bundle",type=Path,default=Path("policyproof_dev48_paired_gpu_results.zip"))
    args=p.parse_args()
    if (sum(bool(x) for x in (args.execute,args.audit_existing,args.make_bundle))>1):
        p.error("--execute, --audit-existing and --make-bundle are mutually exclusive")
    if args.resume_base and not args.execute:
        p.error("--resume-base requires explicit --execute and --confirm-gpu-inference")
    if args.confirm_gpu_inference and not args.execute:
        p.error("--confirm-gpu-inference must accompany --execute")
    if args.execute and not args.confirm_gpu_inference:
        p.error("Actual model generation requires --execute --confirm-gpu-inference")
    if args.make_bundle:
        summary=create_results_bundle(args.base,args.tuned,args.paired,args.bundle)
    elif args.audit_existing:
        summary=audit_existing(args.root,args.archive,args.expected_archive_sha256,
                               args.base,args.tuned,args.paired,
                               expected_backend=args.adapter_backend)
    elif args.execute:
        summary=execute_pair(args.root,args.archive,args.expected_archive_sha256,
                             args.base,args.tuned,args.paired,resume_base=args.resume_base,
                             expected_backend=args.adapter_backend)
    else:
        summary=preflight(args.root,args.archive,args.expected_archive_sha256,
                          expected_backend=args.adapter_backend)
    print(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()
