"""Conservative multilingual *currency-consistency triage* for saved model text.

A source ID and the right numeric amount do NOT prove factual grounding when
a response misstates the monetary unit (e.g. an original NOK fee described
in Arabic as SWEDISH kroner). This checks only the specifically cited trusted
policy block; it never looks at expected answers, labels or the gold source.

Flags are AUDIT CANDIDATES, not independent semantic judgments, corrections,
weight-space improvements or proof of overall correctness. Generic "kr",
"krone" and "كرونة" are deliberately ignored because they are ambiguous.
No GPU, model call, token, HF upload or human vote is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from policyproof.core import ROOT
from scripts.dev48_gpu_pair import DEV_REL, check_dev_rows
from scripts.export_review_from_bundle import verify_bundle
from scripts.inference_guard import safe_output, trusted_policy_blocks
from scripts.research_report_io import preserve_deterministic_report

ORIGINAL_NATIVE40_ZIP_SHA256 = (
    "e0319c83fbb87ef8570ca0c42caf68bea4a4fef1f8ae320bbe11e9a7c1761c68"
)
NATIVE_ADAPTER_SHA256 = (
    "ba985c1966050764a5220e87ea581a9bd482ba87b055fee4a1e363643f1cc92e"
)

# Highly explicit currency-name matches in English, Norwegian, Arabic.
# Do not guess the bank currency from generic kroner/kr/krona/كرونة.
RULES = {
    "NOK": (
        r"(?<!\w)NOK(?!\w)|\bNorwegian\s+kron(?:e|er)\b|"
        r"\bnorske?\s+kroner?\b|كرون(?:ة|ات)\s+نرويجي(?:ة|ه)?"
    ),
    "SEK": (
        r"(?<!\w)SEK(?!\w)|\bSwedish\s+kron(?:a|or|as)\b|"
        r"\bsvenske?\s+kroner?\b|كرون(?:ة|ات)\s+سويدي(?:ة|ه)?"
    ),
    "DKK": (
        r"(?<!\w)DKK(?!\w)|\bDanish\s+kron(?:e|er)\b|"
        r"\bdanske?\s+kroner?\b|كرون(?:ة|ات)\s+دنماركي(?:ة|ه)?"
    ),
    "USD": (
        r"(?<!\w)USD(?!\w)|\b(?:US|American)\s+dollars?\b|"
        r"\bamerikanske?\s+dollar\b|دولار(?:ات)?\s+أمريكي(?:ة|ه)?"
    ),
    "EUR": (
        r"(?<!\w)EUR(?!\w)|\beuros?\b|"
        r"€|\beuro\b|يورو"
    ),
}
_COMPILED = {
    code: re.compile(pattern, re.IGNORECASE | re.UNICODE)
    for code, pattern in RULES.items()
}


def explicit_currencies(value: str) -> list[str]:
    """Only detect clearly specified currency, never infer from generic 'kr'."""
    if not isinstance(value, str) or len(value) > 20_000:
        return []
    return sorted(code for code, regex in _COMPILED.items() if regex.search(value))


def inspect_currency_evidence(raw: str, *, context: str) -> dict:
    """Review stated currencies against cited source, without an oracle.

    Invalid output shapes remain abstention/format concerns of the existing
    guard; don't reclassify them as correctly or incorrectly grounded.
    """
    decision = safe_output(raw, context=context)
    result = {
        "status": "no_valid_supported_answer_to_check",
        "guard_action": decision.action,
        "cited_source_id": None,
        "answer_currencies": [],
        "cited_source_currencies": [],
        "unmatched_answer_currencies": [],
        "needs_currency_review": False,
        "factual_accuracy_verified": False,
        "source_entailment_verified": False,
    }
    if decision.action != "valid_supported_shape":
        if decision.action == "ambiguous_policy_source":
            result["status"] = "ambiguous_cited_source_requires_review"
            result["needs_currency_review"] = True
        return result
    parsed = json.loads(decision.output)
    source_id = parsed["source_id"]
    answer_currencies = explicit_currencies(parsed["answer"])
    result["cited_source_id"] = source_id
    result["answer_currencies"] = answer_currencies
    blocks = trusted_policy_blocks(context)
    if source_id not in blocks:
        result["status"] = "ambiguous_cited_source_requires_review"
        result["needs_currency_review"] = True
        return result
    cited = explicit_currencies(blocks[source_id])
    unmatched = sorted(set(answer_currencies) - set(cited))
    result["cited_source_currencies"] = cited
    result["unmatched_answer_currencies"] = unmatched
    if not answer_currencies:
        result["status"] = "no_explicit_currency_in_answer"
    elif unmatched and cited:
        result["status"] = "explicit_currency_conflict_requires_review"
        result["needs_currency_review"] = True
    elif unmatched:
        result["status"] = "currency_asserted_without_explicit_cited_source_unit"
        result["needs_currency_review"] = True
    else:
        result["status"] = "currency_mentions_match_but_semantics_unverified"
    return result


def replay_native40_currency_risk(
    root: Path, owner_zip: Path,
    expected_sha256: str = ORIGINAL_NATIVE40_ZIP_SHA256,
) -> dict:
    """Exact original saved 48×2 paired replay; aggregate-only public output."""
    root = Path(root).resolve()
    blobs, provenance = verify_bundle(owner_zip, expected_sha256)
    cases, data_sha = check_dev_rows(root / DEV_REL)
    ids = [row["id"] for row in cases]
    if len(ids) != 48 or len(set(ids)) != 48:
        raise ValueError("Frozen development sample must be 48 unique cases")
    reports = {
        name: json.loads(blobs["dev48_" + name + ".json"])
        for name in ("base", "tuned")
    }
    if (reports["base"].get("is_finetuned") is not False
        or reports["tuned"].get("is_finetuned") is not True
        or reports["tuned"].get("adapter_sha256") != NATIVE_ADAPTER_SHA256
        or reports["base"].get("dataset_sha256") != data_sha
        or reports["tuned"].get("dataset_sha256") != data_sha):
        raise ValueError("Original paired base/native adapter provenance mismatch")
    by_arm = {}
    for name, report in reports.items():
        predictions = report.get("predictions")
        if (not isinstance(predictions, list) or len(predictions) != 48
            or any(not isinstance(x, dict)
                   or set(x) != {"id", "raw"}
                   or not isinstance(x["raw"], str) or len(x["raw"]) > 20_000
                   for x in predictions)):
            raise ValueError("Invalid original per-case model predictions")
        observed = {p["id"]: p["raw"] for p in predictions}
        if len(observed) != 48 or set(observed) != set(ids):
            raise ValueError("Original model outputs are not the exact frozen cases")
        counts = Counter()
        review_candidates = []
        lang_counts = {
            lang: {"cases": 0, "currency_conflict_flags": 0}
            for lang in ("en", "no", "ar")
        }
        for case in cases:
            diag = inspect_currency_evidence(
                observed[case["id"]], context=case["context"]
            )
            counts[diag["status"]] += 1
            lang_counts[case["language"]]["cases"] += 1
            if diag["status"] == "explicit_currency_conflict_requires_review":
                lang_counts[case["language"]]["currency_conflict_flags"] += 1
            if diag["needs_currency_review"]:
                # IDs and unit codes only. Raw private model text, questions and
                # citations are never serialized into this public result.
                review_candidates.append({
                    "case_id": case["id"],
                    "language": case["language"],
                    "status": diag["status"],
                    "source_currency_codes": diag["cited_source_currencies"],
                    "response_currency_codes": diag["answer_currencies"],
                    "unmatched_currency_codes": diag["unmatched_answer_currencies"],
                })
        by_arm[name] = {
            "original_saved_responses": 48,
            "status_counts": dict(sorted(counts.items())),
            "explicit_currency_conflict_flags": counts[
                "explicit_currency_conflict_requires_review"
            ],
            "all_currency_review_candidates": len(review_candidates),
            "by_language": lang_counts,
            "flagged_synthetic_case_ids_only": review_candidates,
        }
    return {
        "schema": "policyproof_native40_saved_dev48_currency_triage_v1",
        "status": "ORIGINAL_SAVED_PAIRED_DEVELOPMENT_CURRENCY_REVIEW_FLAGS_ONLY",
        "source_original_private_zip_sha256": provenance["source_zip_sha256"],
        "dataset_sha256": data_sha,
        "adapter_tensor_sha256": NATIVE_ADAPTER_SHA256,
        "cases": 48, "saved_original_model_responses": 96,
        "gold_expected_answers_or_correct_source_ids_used": False,
        "currency_flags_are_semantic_verdicts": False,
        "human_reviewer_votes_collected": False,
        "independent_semantic_accuracy_verified": False,
        "new_gpu_inference_training_or_model_weights_changed": False,
        "private_raw_model_responses_published": False,
        "arms": by_arm,
        "limitations": [
            "These are source-local lexical review flags only, NOT fact-checking "
            "or evidence entailment. No improvement in model safety is claimed.",
            "Ambiguous generic units (kr/kroner/كرونة) and unspecified currencies "
            "are deliberately not interpreted as explicit currencies.",
            "Quoted disclaimers/negations or foreign exchange comparisons can "
            "create false positives; equivalent currency abbreviations can "
            "create false negatives. Have the owner review flagged cases.",
            "Invalid schema or contradictory abstention is handled by the "
            "separate guard; this currency check never certifies such outputs.",
            "Original saved GPU provenance is owner-reported, not externally attested.",
            "The 48-case synthetic development set is previously inspected, "
            "not blind; flags are not population-level rates or human judgment.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256", default=ORIGINAL_NATIVE40_ZIP_SHA256
    )
    parser.add_argument(
        "--output", default="artifacts/native40_dev48_currency_risk_aggregate.json"
    )
    args = parser.parse_args()
    record = replay_native40_currency_risk(
        args.root, args.bundle, args.expected_sha256
    )
    result = json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    output, reused = preserve_deterministic_report(
        result, args.output, root=args.root,
    )
    print(json.dumps({
        "status": record["status"],
        "original_case_count": record["cases"],
        "base_currency_conflict_flags":
            record["arms"]["base"]["explicit_currency_conflict_flags"],
        "adapter_currency_conflict_flags":
            record["arms"]["tuned"]["explicit_currency_conflict_flags"],
        "report_path": str(output),
        "identical_existing_report_reused": reused,
        "new_model_inference_or_training": False,
        "independent_human_review": False,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
