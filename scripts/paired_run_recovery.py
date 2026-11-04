"""Fail-closed recovery of the COMPLETED base pass in paired GPU evaluation.

No model execution occurs here. A saved base response file is not proof that
GPU inference actually happened; this module validates consistency and
declared provenance before allowing an explicit operator-controlled resume.

A base file is written immediately after completing its 60 cases, preserving
that expensive stage if the tuned-model load or AuditKIT step later fails.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path

from policyproof.core import aggregate, score


MAX_BASE_REPORT_BYTES = 2_000_000


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Ambiguous duplicate field in saved base report: " + key)
        result[key] = value
    return result


def write_evidence_once(path: Path, content: dict) -> None:
    """Exclusive create; clean up OUR partial file if serialization fails.

    This never truncates or replaces a completed report. Evidence is readable
    only when the complete JSON has been fsync'd successfully.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Only our exclusive-new file may be removed; never touch prior evidence.
        path.unlink(missing_ok=True)
        raise


def validate_saved_base(
    path: Path, rows: list[dict], *,
    dataset_sha256: str, revision: str,
    generation_config: dict, model_id: str,
) -> dict:
    """Validate complete raw outputs AND recomputed deterministic scores.

    Self-declared GPU provenance cannot be independently authenticated.
    Reusing a base file ALWAYS requires a separate explicit resume flag.
    """
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BASE_REPORT_BYTES:
        raise ValueError("Missing, symlinked or oversized saved base report")
    try:
        report = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_fields
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Saved base report is invalid or ambiguous JSON") from exc
    if not isinstance(report, dict):
        raise ValueError("Saved base must be a JSON object")
    expected = {
        "engine": "transformers",
        "is_real_model_result": True,
        "is_finetuned": False,
        "model": model_id,
        "model_revision": revision,
        "adapter_sha256": None,
        "split": "final_candidate_holdout_v1",
        "dataset_sha256": dataset_sha256,
        "generation_config": generation_config,
    }
    for key, value in expected.items():
        actual = report.get(key)
        if (key in ("is_real_model_result", "is_finetuned")
            and (type(actual) is not bool or actual is not value)):
            raise ValueError("Saved base has incompatible " + key)
        if actual != value:
            raise ValueError("Saved base provenance mismatch: " + key)
    provenance = report.get("actual_gpu_inference_metadata")
    if (not isinstance(provenance, dict)
        or not isinstance(provenance.get("gpu"), str) or not provenance["gpu"]
        or not isinstance(provenance.get("torch_version"), str)
        or not provenance["torch_version"]):
        raise ValueError("Saved base lacks declared GPU provenance")
    created = report.get("created_utc")
    if not isinstance(created, str):
        raise ValueError("Saved base has no inference timestamp")
    try:
        parsed_time = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if parsed_time.tzinfo is None:
            raise ValueError("Timestamp missing offset")
    except ValueError as exc:
        raise ValueError("Saved base has invalid UTC timestamp") from exc
    predictions = report.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise ValueError("Saved base has incomplete prediction coverage")
    outputs = {}
    for record in predictions:
        if (not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not isinstance(record.get("raw"), str)
            or len(record["raw"]) > 10000):
            raise ValueError("Saved base has malformed prediction data")
        if record["id"] in outputs:
            raise ValueError("Saved base contains duplicated prediction IDs")
        outputs[record["id"]] = record["raw"]
    expected_ids = {row["id"] for row in rows}
    if len(expected_ids) != len(rows) or set(outputs) != expected_ids:
        raise ValueError("Saved base case IDs differ from frozen benchmark")
    recomputed = [score(row, outputs[row["id"]]) for row in rows]
    if report.get("scores") != recomputed or report.get("metrics") != aggregate(recomputed):
        raise ValueError("Saved base scores or metrics differ from recomputed values")
    if report.get("human_semantic_review_completed") is not False:
        raise ValueError("Saved base makes unsupported human semantic claim")
    return report
