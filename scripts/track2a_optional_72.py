"""Read-only bridge between an unscored 72-case challenge and actual saved results.

Always distinguish (a) NO saved model evaluation, (b) malformed/incomplete
optional evidence and (c) a genuine source-recomputed local AuditKIT replay.
A completed marker by itself never qualifies. This CPU-only bridge does NOT
authenticate GPU execution, human answer correctness, rights or submission.
"""
from __future__ import annotations

from pathlib import Path

from policyproof.core import ROOT
from scripts.verify_track2a_72_receipts import verify_completed_72

EVALUATION_DIR = "artifacts/actual_72_case_evaluation"
MODEL_REPORT = "artifacts/track2a_72_unchanged_base_predictions.json"
FROZEN_DATA = "artifacts/track2a_challenge_unscored/challenge_v1.jsonl"


def optional_72_evidence(root: Path = ROOT) -> dict:
    """Inspect only exact original paths, never create or modify evidence."""
    root = Path(root).resolve()
    folder = root / EVALUATION_DIR
    report = root / MODEL_REPORT
    # No file and no evaluation folder: an intentionally NOT SCORED result,
    # never report 'failed GPU inference' or claim an evaluation happened.
    present = (
        folder.exists() or folder.is_symlink()
        or report.exists() or report.is_symlink()
    )
    if not present:
        return {
            "status": "NO_SAVED_72_MODEL_EVALUATION",
            "evidence_present": False,
            "verified_source_replay": False,
            "cases_source_replayed": 0,
            "source_replay": None,
            "original_gpu_execution_independently_authenticated": False,
            "human_semantic_review_completed": False,
            "warning": (
                "New frozen 72-case challenge is curated but UNSCORED; "
                "no original model report or complete AuditKIT evidence is present."
            ),
        }
    try:
        result = verify_completed_72(
            EVALUATION_DIR, FROZEN_DATA, MODEL_REPORT, root=root
        )
        if (result["status"] != "VERIFIED_LOCAL_72_CASE_SOURCE_REPLAY"
            or result["cases"] != 72
            or result["original_gpu_provenance_independently_authenticated"] is not False
            or result["human_semantic_quality_verified"] is not False):
            raise ValueError("Saved local replay does not meet approved 72-case verifier contract")
    except (OSError, ValueError, TypeError, KeyError, RecursionError, ImportError) as exc:
        return {
            "status": "INVALID_OR_INCOMPLETE_SAVED_72_EVIDENCE",
            "evidence_present": True,
            "verified_source_replay": False,
            "cases_source_replayed": 0,
            "source_replay": None,
            "original_gpu_execution_independently_authenticated": False,
            "human_semantic_review_completed": False,
            "warning": (
                "Saved model evidence exists but failed frozen-source/score/"
                "original AuditKIT replay verification: "
                + type(exc).__name__ + ": " + str(exc)[:220]
            ),
        }
    return {
        "status": "SAVED_72_CPU_SOURCE_REPLAY_VERIFIED_NOT_GPU_ATTESTED",
        "evidence_present": True,
        "verified_source_replay": True,
        "cases_source_replayed": 72,
        "source_replay": result,
        "original_gpu_execution_independently_authenticated": False,
        "human_semantic_review_completed": False,
        "warning": (
            "All 72 preserved raw responses, exact dataset, measured scores "
            "and AuditKIT receipt reproduce locally. Model runner metadata is "
            "self-recorded, NOT independently verified original GPU execution; "
            "synthetic gold answers still need human semantic judgment. "
            "This does not establish a trained model or official submission."
        ),
    }
