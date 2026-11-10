"""Re-score saved native40 dev48 RAW responses against the frozen gold dataset.

This deliberately DOES MORE than re-sum archived score rows: it independently
re-executes policyproof.core.score on every raw response and immutable gold
row, then compares each recomputed field to the originally saved per-case
score and reports directional paired transitions.

No internet, GPU, HF token, inference, training, human votes or publication.
The raw 96 private model generations NEVER appear in the aggregate report.
Scoring is still an automatic exact-reference rubric, NOT independent human
semantic correctness or an independently audited GPU execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from policyproof.core import ROOT, score
from scripts.dev48_gpu_pair import DEV_REL, check_dev_rows
from scripts.export_review_from_bundle import verify_bundle
from scripts.paired_guard_uncertainty import (
    exact_discordance_p, paired_measure,
)
from scripts.research_report_io import preserve_deterministic_report

ORIGINAL_NATIVE40_ZIP = (
    "e0319c83fbb87ef8570ca0c42caf68bea4a4fef1f8ae320bbe11e9a7c1761c68"
)
ORIGINAL_ADAPTER_SHA = (
    "ba985c1966050764a5220e87ea581a9bd482ba87b055fee4a1e363643f1cc92e"
)
METRICS = {
    "schema_valid": ("format_valid", None, 1),
    "strict_exact": ("correct", None, 1),
    "supported_answer_exact": ("supported_answer_correct", "supported_total", 1),
    "supported_expected_source_raw": ("raw_citation_correct", "supported_total", 1),
    "supported_expected_source_schema_valid": ("citation_correct", "supported_total", 1),
    "unsupported_abstention_correct": ("abstention_correct", "abstention_needed", 1),
    "unsupported_claims": ("unsupported_claim", "abstention_needed", -1),
    "nonabstaining_answer_coverage": ("answered", None, 1),
    "missing_schema_fields": ("missing_required_field", None, -1),
    "contradictory_abstention": ("contradictory_abstention", None, -1),
}


def _original_scores(report: dict, expected_ids: set[str], label: str) -> dict:
    predictions = report.get("predictions")
    scores = report.get("scores")
    if (not isinstance(predictions, list) or not isinstance(scores, list)
        or len(predictions) != 48 or len(scores) != 48):
        raise ValueError(label + ": require 48 raw predictions and 48 saved scores")
    if (any(not isinstance(p, dict) or set(p) != {"id", "raw"}
            or not isinstance(p["raw"], str) or len(p["raw"]) > 20_000
            for p in predictions)
        or any(not isinstance(s, dict) or not isinstance(s.get("id"), str)
               for s in scores)):
        raise ValueError(label + ": malformed raw prediction or saved score")
    pred_by_id = {p["id"]: p["raw"] for p in predictions}
    score_by_id = {s["id"]: s for s in scores}
    if (len(pred_by_id) != 48 or len(score_by_id) != 48
        or set(pred_by_id) != expected_ids or set(score_by_id) != expected_ids):
        raise ValueError(label + ": raw predictions/scores do not match the frozen gold IDs")
    return {"raw": pred_by_id, "saved": score_by_id}


def rescore_original_bundle(
    root: Path, archive: Path, expected_sha256: str = ORIGINAL_NATIVE40_ZIP,
) -> dict:
    """Fail-closed source audit; report only aggregate measurements."""
    root = Path(root).resolve()
    if archive.resolve() == (root / DEV_REL).resolve():
        raise ValueError("Source ZIP cannot replace the original gold dataset")
    blobs, source = verify_bundle(archive, expected_sha256)
    rows, dataset_sha = check_dev_rows(root / DEV_REL)
    base = json.loads(blobs["dev48_base.json"])
    tuned = json.loads(blobs["dev48_tuned.json"])
    paired = json.loads(blobs["dev48_paired_auditkit.json"])
    if (base.get("is_finetuned") is not False
        or tuned.get("is_finetuned") is not True
        or base.get("human_semantic_review_completed") is not False
        or tuned.get("human_semantic_review_completed") is not False
        or tuned.get("adapter_sha256") is None
        or paired.get("adapter_sha256") != tuned.get("adapter_sha256")
        or paired.get("training_backend_declared") != "aligntune_wrapper"
        or paired.get("is_final_blind_test") is not False
        or paired.get("is_model_improvement_semantically_verified") is not False):
        raise ValueError("Source cannot be presented as original native40 paired dev evidence")
    if (expected_sha256 == ORIGINAL_NATIVE40_ZIP
        and tuned.get("adapter_sha256") != ORIGINAL_ADAPTER_SHA):
        raise ValueError("Original adapter tensor SHA256 does not match owner source")

    expected_ids = {r["id"] for r in rows}
    original = {
        name: _original_scores(report, expected_ids, name)
        for name, report in (("base", base), ("tuned", tuned))
    }
    recalculated = {}
    for name in ("base", "tuned"):
        recalculated[name] = []
        for row in rows:
            observed = score(row, original[name]["raw"][row["id"]])
            saved = original[name]["saved"][row["id"]]
            # Compare every original score field, including subgroup identity,
            # and reject extra/missing fields in the archived reference scores.
            if observed != saved:
                raise ValueError(
                    name + ": raw predictions disagree with original stored "
                    "gold-reference score at case " + row["id"]
                )
            recalculated[name].append(observed)

    by_metric = {}
    for name, (field, denominator, direction) in METRICS.items():
        measurement = paired_measure(
            recalculated["base"], recalculated["tuned"],
            field=field, denominator=denominator, direction=direction,
        )
        # Avoid rounding highly significant tiny exact-p values to 0.0:
        # p=0 is mathematically incorrect and misleading.
        measurement["exploratory_exact_mcnemar_p_unrounded"] = exact_discordance_p(
            measurement["improved_cases"], measurement["regressed_cases"],
        )
        by_metric[name] = measurement

    by_language = {}
    for language in ("en", "no", "ar"):
        ids = {row["id"] for row in rows if row["language"] == language}
        a = [x for x in recalculated["base"] if x["id"] in ids]
        b = [x for x in recalculated["tuned"] if x["id"] in ids]
        if len(a) != 16 or len(b) != 16:
            raise ValueError("Language group is not exactly 16 paired examples")
        by_language[language] = {
            "case_count": len(a),
            "schema": paired_measure(a, b, field="format_valid"),
            "strict_exact": paired_measure(a, b, field="correct"),
            "unsupported_abstention": paired_measure(
                a, b, field="abstention_correct",
                denominator="abstention_needed",
            ),
            "unsupported_claims": paired_measure(
                a, b, field="unsupported_claim",
                denominator="abstention_needed", direction=-1,
            ),
        }

    return {
        "status": "FROZEN_GOLD_AND_96_RAW_PREDICTIONS_STRICTLY_RESCORED",
        "original_private_bundle_sha256": source["source_zip_sha256"],
        "gold48_source_sha256": dataset_sha,
        "original_adapter_tensor_sha256": tuned["adapter_sha256"],
        "base_model_revision_reported": base["model_revision"],
        "original_raw_response_count": 96,
        "all_original_saved_case_scores_exactly_reproduced": True,
        "schema": "policyproof_native40_dev48_gold_rescore_aggregate_v1",
        "previously_inspected_development_only": True,
        "independent_human_semantic_review_completed": False,
        "original_GPU_execution_independently_attested": False,
        "training_time_base_model_revision_independently_verified": False,
        "new_model_inference_or_training_performed": False,
        "raw_private_model_predictions_included": False,
        "public_HF_publication_or_lexsi_submission_performed": False,
        "metrics": by_metric, "languages": by_language,
        "limitations": [
            "All 96 RAW saved outputs were separately scored against the original "
            "frozen gold with the published deterministic strict rubric. No "
            "human semantic equivalence or source entailment was judged.",
            "Exact text mismatches include ungraded paraphrases; matching "
            "source IDs alone does not establish answer correctness.",
            "48 cases were already inspected, synthetic and correlated. "
            "The nine unsupported cases are too few for broad safety claims.",
            "Paired McNemar p-values are exploratory and not corrected for "
            "multiple tests; Wilson intervals are not real-world guarantees.",
            "Original GPU execution and the actually loaded base revision "
            "during training are not independently attested.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=ORIGINAL_NATIVE40_ZIP)
    parser.add_argument(
        "--output", default="artifacts/native40_fresh_gold_rescore_aggregate.json"
    )
    args = parser.parse_args()
    result = rescore_original_bundle(
        args.root, args.bundle, args.expected_sha256
    )
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path, reused = preserve_deterministic_report(
        output, args.output, root=args.root
    )
    print(json.dumps({
        "status": result["status"],
        "report_path": str(path),
        "reused_exact_result": reused,
        "model_improvement_semantically_verified": False,
        "raw_private_outputs_written": False,
    }, indent=2))


if __name__ == "__main__":
    main()
