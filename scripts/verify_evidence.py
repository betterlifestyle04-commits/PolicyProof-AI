"""Read-only, CPU-only evidence validation for PolicyProof AI.

Verifies real saved Tiny Aya model predictions against the EXACT dataset bytes
and recomputes deterministic scoring; optionally verifies the original
CuratorKIT archive. It does not access the network or run the model.

Example (in repo root, using user-provided files):
 python -m scripts.verify_evidence \
   --baseline /path/to/tiny_aya_base_unseen_policy-rescored-v2.json \
   --dataset data/unseen_policy_test.jsonl \
   --curator-zip /path/to/policyproof_curatorkit_train_v1.zip \
   --out artifacts/evidence-verified.json
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import math

from policyproof.core import aggregate, read_jsonl, score, sha256
from scripts.verify_curator_archive import check_zip

EXPECTED_CURATOR_TRAIN_SHA256 = (
    "4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c"
)
MAX_BASELINE_REPORT_BYTES = 8_000_000
MAX_BASELINE_DATASET_BYTES = 3_000_000


def _reject_duplicate_evidence_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON evidence field: " + key)
        result[key] = value
    return result


def read_evidence_json(path: Path, *, limit: int = MAX_BASELINE_REPORT_BYTES) -> dict:
    """Bounded, ambiguity-free source reader; evidence is never modified."""
    path = Path(path)
    if (path.is_symlink() or not path.is_file()
        or not 0 < path.stat().st_size <= limit):
        raise ValueError("Evidence JSON missing, symlinked, empty or oversized")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_evidence_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Nonfinite JSON evidence: " + token)
            ),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError("Malformed or ambiguous JSON evidence") from exc
    if not isinstance(value, dict):
        raise ValueError("Evidence JSON must be an object")
    return value


def verify_baseline(baseline: Path, dataset: Path) -> dict:
    data = read_evidence_json(baseline)
    dataset = Path(dataset)
    if (dataset.is_symlink() or not dataset.is_file()
        or not 0 < dataset.stat().st_size <= MAX_BASELINE_DATASET_BYTES):
        raise ValueError("Baseline dataset missing, symlinked, empty or oversized")
    rows = read_jsonl(dataset)
    actual_hash = sha256(dataset)
    if data.get("engine") != "transformers" or data.get("is_real_model_result") is not True:
        raise ValueError("Report is not labeled as actual Transformers model inference")
    if data.get("is_finetuned") is not False:
        raise ValueError("This report must describe the untouched base model")
    if data.get("model") != "CohereLabs/tiny-aya-global":
        raise ValueError("Unexpected model name")
    if data.get("dataset_sha256") != actual_hash:
        raise ValueError("Baseline dataset SHA256 does not match provided dataset bytes")
    if data.get("split") != "unseen-policy":
        raise ValueError("Expected the original unseen-policy development split")
    if len(rows) != 48 or len(set(r["id"] for r in rows)) != len(rows):
        raise ValueError("Expected 48 unique development evaluation rows")
    predictions = data.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise ValueError("Baseline predictions not complete")
    prediction_map = {}
    for p in predictions:
        if (not isinstance(p, dict) or not isinstance(p.get("id"), str)
            or not isinstance(p.get("raw"), str)):
            raise ValueError("Malformed model prediction")
        if p["id"] in prediction_map:
            raise ValueError("Duplicate model prediction ID")
        prediction_map[p["id"]] = p["raw"]
    if set(prediction_map) != {r["id"] for r in rows}:
        raise ValueError("Baseline prediction/sample IDs do not align")
    scores = [score(row, prediction_map[row["id"]]) for row in rows]
    actual_metrics = aggregate(scores)
    stored = data.get("metrics")
    if not isinstance(stored, dict):
        raise ValueError("No claimed metrics for baseline")
    # Compare exactly the scorer's published fields, not historical reports
    # with optional diagnostics added in later revisions.
    for key, value in actual_metrics.items():
        if isinstance(value, (float, int, str)) or value is None:
            original = stored.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if (type(original) not in (int, float)
                    or not math.isfinite(original)
                    or not math.isclose(original, float(value), abs_tol=0.011)):
                    raise ValueError(f"Metric discrepancy: {key}, stored={original}, recomputed={value}")
            elif original != value:
                raise ValueError(f"Metric discrepancy: {key}")
    if "scores" in data:
        if (not isinstance(data["scores"], list)
            or any(not isinstance(x, dict) or not isinstance(x.get("id"), str)
                   for x in data["scores"])):
            raise ValueError("Malformed per-row baseline scores")
        original_scores = {x["id"]: x for x in data["scores"]}
        if len(original_scores) != len(data["scores"]) or set(original_scores) != set(prediction_map):
            raise ValueError("Original scorer rows have missing or duplicate IDs")
        for row in scores:
            old = original_scores[row["id"]]
            for field in ("correct", "format_valid", "abstention_correct", "unsupported_claim"):
                if old.get(field) != row[field]:
                    raise ValueError(f"Stored per-row score differs for {row['id']}: {field}")
    return {
        "status": "verified_against_exact_dataset_bytes",
        "dataset_sha256": actual_hash,
        "sample_count": len(rows),
        "prediction_count": len(prediction_map),
        "languages": dict(sorted(Counter(r["language"] for r in rows).items())),
        "recomputed_metrics": actual_metrics,
        "real_inference_claim_supported_by_report_metadata_only": True,
        "original_inference_provenance_independently_verified": False,
        "model_training_performed": False,
        "human_semantic_review_completed": False,
    }


def verify_bundle(*, baseline=None, dataset=None, curator_zip=None) -> dict:
    if not curator_zip and not baseline:
        raise ValueError("Provide a CuratorKIT ZIP, baseline and dataset, or both")
    result = {"status": "cpu_artifact_validation", "no_model_execution": True}
    if curator_zip:
        z = check_zip(curator_zip)
        from zipfile import ZipFile
        import hashlib
        with ZipFile(curator_zip) as archive:
            hash_train = hashlib.sha256(archive.read("sft_alpaca.jsonl")).hexdigest()
        if hash_train != EXPECTED_CURATOR_TRAIN_SHA256:
            raise ValueError("CuratorKIT training export changed from audited dataset")
        result["curatorkit"] = {"status": z["status"], "rows": z["row_count"],
                                "languages": z["languages"], "train_sha256": hash_train}
    if baseline or dataset:
        if not baseline or not dataset:
            raise ValueError("Baseline and dataset must be provided together")
        result["baseline"] = verify_baseline(Path(baseline), Path(dataset))
    return result


def main():
    p = argparse.ArgumentParser(description="Verify synthetic research evidence without GPU")
    p.add_argument("--baseline")
    p.add_argument("--dataset")
    p.add_argument("--curator-zip")
    p.add_argument("--out")
    args = p.parse_args()
    result = verify_bundle(baseline=args.baseline, dataset=args.dataset,
                           curator_zip=args.curator_zip)
    payload = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        target = Path(args.out)
        if args.baseline and target.resolve() == Path(args.baseline).resolve():
            raise ValueError("Refusing to overwrite the baseline report")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
