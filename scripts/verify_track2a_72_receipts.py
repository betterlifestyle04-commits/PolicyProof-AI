"""Read-only, source-recomputed verification of a saved 72-case CPU AuditKIT run.

The existence of EVIDENCE_COMPLETE.json is insufficient. Its three hashes,
source inputs, 72-case scoring, language/family slices, and native AuditKIT
replay must ALL match. This never loads a language model, authenticates
declared GPU execution, independently reviews semantics, or changes files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.generate_track2a_challenge import validate_challenge
from scripts.score_track2a_72 import (
    KNOWN_CHALLENGE_SHA256, MAX_CHALLENGE_DATASET_BYTES,
    MAX_SOURCE_REPORT_BYTES, SAFE_OUTPUT_DIR,
    evaluate_real_run, summarize, validate_model_report,
)
from scripts.verify_evidence import read_evidence_json

_EXPECTED_FILES = {
    "measured_72_report.json",
    "auditkit_72_receipt.json",
    "EVIDENCE_COMPLETE.json",
}
_MAX_SCORE_BYTES = 2_000_000
_MAX_MARKER_BYTES = 50_000


def _source_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _saved_folder(value: str | Path, root: Path) -> Path:
    """Only previously generated direct artifacts/<name> directories qualify."""
    literal = str(value)
    if (not literal or len(literal) > 1024
        or any(ord(c) < 32 or ord(c) == 127 for c in literal)):
        raise ValueError("Saved evaluation path must be direct artifacts/<name>")
    given = Path(literal)
    if given.is_absolute():
        if given.parent != root / "artifacts":
            raise ValueError("Saved evaluation path must be direct artifacts/<name>")
        name = given.name
    else:
        parts = literal.split("/")
        if ("\\" in literal or len(parts) != 2 or parts[0] != "artifacts"):
            raise ValueError("Saved evaluation path must be direct artifacts/<name>")
        name = parts[1]
    if not SAFE_OUTPUT_DIR.fullmatch(name):
        raise ValueError("Unsafe saved evaluation directory name")
    folder = root / "artifacts" / name
    if (folder.parent.is_symlink() or folder.is_symlink()
        or not folder.is_dir()):
        raise ValueError("Saved evaluation directory absent, unsafe or symlinked")
    files = list(folder.iterdir())
    if {p.name for p in files} != _EXPECTED_FILES or len(files) != 3:
        raise ValueError("Incomplete or unexpected saved 72-case evidence inventory")
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("Saved evidence contains unsafe files or symbolic links")
    return folder


def verify_completed_72(
    folder: str | Path,
    dataset: str | Path,
    model_report: str | Path,
    *, root: Path = ROOT,
) -> dict:
    """Recompute 72 synthetic-case scores and genuine AuditKIT from source bytes.

    Strictly a *local* consistency check; source metadata can be self-authored,
    so this cannot assert real GPU provenance, reviewer action, or quality.
    """
    root = Path(root).resolve()
    saved = _saved_folder(folder, root)
    data = _source_path(dataset, root)
    source = _source_path(model_report, root)
    for path, limit in ((data, MAX_CHALLENGE_DATASET_BYTES),
                        (source, MAX_SOURCE_REPORT_BYTES)):
        if (path.is_symlink() or not path.is_file()
            or not 0 < path.stat().st_size <= limit):
            raise ValueError("Source report or dataset absent, symlinked or oversized")
    if data.resolve() == source.resolve():
        raise ValueError("Model outputs cannot be the same file as the dataset")
    hashes = {
        "frozen_dataset_sha256": sha256(data),
        "original_model_report_sha256": sha256(source),
    }
    if hashes["frozen_dataset_sha256"] != KNOWN_CHALLENGE_SHA256:
        raise ValueError("Frozen 72-case challenge checksum differs from approved source")
    files = {
        "measured_report_sha256": saved / "measured_72_report.json",
        "auditkit_receipt_sha256": saved / "auditkit_72_receipt.json",
    }
    recorded_hashes = {key: sha256(path) for key, path in files.items()}
    marker = read_evidence_json(saved / "EVIDENCE_COMPLETE.json", limit=_MAX_MARKER_BYTES)
    expected_marker = {
        "status": "SAVED_72_CPU_AUDIT_COMPLETED",
        "model_inference_executed_in_this_process": False,
        "human_semantic_review_completed": False,
        **hashes,
        **recorded_hashes,
    }
    if marker != expected_marker:
        raise ValueError("Completed marker not bound to exact original evidence and saved reports")
    scores = read_evidence_json(files["measured_report_sha256"], limit=_MAX_SCORE_BYTES)
    receipt = read_evidence_json(files["auditkit_receipt_sha256"], limit=_MAX_SCORE_BYTES)
    rows = read_jsonl(data)
    validate_challenge(rows)
    source_report = read_evidence_json(source, limit=MAX_SOURCE_REPORT_BYTES)
    predicted = validate_model_report(
        source_report, rows, hashes["frozen_dataset_sha256"]
    )
    expected_scores = summarize(rows, predicted, hashes["frozen_dataset_sha256"], source_report)
    expected_scores.update({
        **hashes,
    })
    if scores != expected_scores:
        raise ValueError("Saved 72-case totals/language/attack metrics do not reproduce from raw model responses")

    # AuditKIT is a CPU-only replay on SAVED outputs. It does not execute any
    # model or contact an external judge. We recompute its native receipt too.
    replay_scores, expected_receipt = evaluate_real_run(
        source_report, rows, hashes["frozen_dataset_sha256"]
    )
    if replay_scores != summarize(
        rows, predicted, hashes["frozen_dataset_sha256"], source_report
    ):
        raise ValueError("AuditKIT replay changed deterministic scoring")
    expected_receipt.update(hashes)
    if receipt != expected_receipt:
        raise ValueError("Saved AuditKIT receipt differs from current actual toolkit replay")

    # Rehash all source/evidence files after replay: never accept mixed versions.
    if any(sha256(path) != digest for path, digest in (
        (data, hashes["frozen_dataset_sha256"]),
        (source, hashes["original_model_report_sha256"]),
        (files["measured_report_sha256"], recorded_hashes["measured_report_sha256"]),
        (files["auditkit_receipt_sha256"], recorded_hashes["auditkit_receipt_sha256"]),
    )):
        raise ValueError("Saved evidence changed while it was being verified")
    if read_evidence_json(saved / "EVIDENCE_COMPLETE.json", limit=_MAX_MARKER_BYTES) != expected_marker:
        raise ValueError("Completion receipt changed while it was being verified")
    return {
        "status": "VERIFIED_LOCAL_72_CASE_SOURCE_REPLAY",
        "cases": 72,
        "dataset_sha256": hashes["frozen_dataset_sha256"],
        "original_model_report_sha256": hashes["original_model_report_sha256"],
        "measured_report_sha256": recorded_hashes["measured_report_sha256"],
        "auditkit_receipt_sha256": recorded_hashes["auditkit_receipt_sha256"],
        "auditkit_version": expected_receipt["auditkit_version"],
        "languages": {
            lang: values["n"] for lang, values in scores["by_language"].items()
        },
        "original_model_inference_performed_here": False,
        "training_performed": False,
        "original_gpu_provenance_independently_authenticated": False,
        "human_semantic_quality_verified": False,
        "official_submission_verified": False,
        "warning": (
            "This is local hash, score and genuine AuditKIT replay consistency "
            "only, NOT independently verified original GPU execution, "
            "human answer correctness, model safety or official submission"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation-dir", default="artifacts/actual_72_case_evaluation")
    p.add_argument(
        "--dataset", default="artifacts/track2a_challenge_unscored/challenge_v1.jsonl"
    )
    p.add_argument(
        "--model-report", default="artifacts/track2a_72_unchanged_base_predictions.json"
    )
    p.add_argument("--strict", action="store_true",
                   help="Return nonzero exit code on incomplete/altered evidence")
    args = p.parse_args()
    try:
        result = verify_completed_72(
            args.evaluation_dir, args.dataset, args.model_report
        )
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        result = {
            "status": "BLOCKED",
            "reason": str(exc)[:240],
            "gpu_inference_verified": False,
            "semantic_accuracy_verified": False,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.strict:
            raise SystemExit(2)
        return
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
