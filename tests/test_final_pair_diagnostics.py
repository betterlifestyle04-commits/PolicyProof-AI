"""Synthetic-only statistical tests; NO training, GPU work or model inference."""

from copy import deepcopy
import math

import pytest

from scripts.final_pair_diagnostics import (
    analyze_scored_pairs, diagnostics_from_raw_reports,
)


def example_pairs():
    # 3 supported and 3 unsupported fictional cases across en/no/ar.
    rows = [
        ("en", "supported", True, 0, 1, 0, 0, 0, 0, 0, 0),
        ("en", "unknown", False, 0, 1, 0, 0, 0, 1, 1, 0),
        ("no", "supported", True, 1, 0, 1, 0, 1, 0, 0, 0),
        ("no", "unknown", False, 0, 1, 0, 0, 0, 0, 1, 1),
        ("ar", "supported", True, 0, 0, 0, 1, 0, 0, 0, 0),
        ("ar", "unknown", False, 1, 1, 0, 0, 1, 1, 0, 0),
    ]
    output = {}
    for index, (lang, cat, supported, b_correct, t_correct,
                 b_cite, t_cite, b_abstain, t_abstain,
                 b_claim, t_claim) in enumerate(rows, 1):
        case_id = f"SYNTHETIC-{index}"
        def scored(correct, citation, abstain, claim):
            return {
                "id": case_id, "lang": lang, "category": cat,
                "correct": correct, "format_valid": 1,
                "supported_total": int(supported),
                "supported_answer_correct": correct if supported else 0,
                "raw_citation_correct": citation,
                "abstention_needed": int(not supported),
                "abstention_correct": abstain if not supported else 0,
                "unsupported_claim": claim if not supported else 0,
            }
        output[case_id] = {
            "lang": lang, "category": cat,
            "baseline": scored(b_correct, b_cite, b_abstain, b_claim),
            "candidate": scored(t_correct, t_cite, t_abstain, t_claim),
        }
    return output


def test_paired_report_uses_only_correct_subset_denominators_and_direction():
    result = analyze_scored_pairs(example_pairs())
    assert result["sample_count"] == 6
    assert result["status"] == "DESCRIPTIVE_ONLY_NOT_MODEL_QUALITY_PROOF"
    assert result["independent_human_semantic_accuracy_verified"] is False
    assert result["statistical_superiority_established"] is False
    all_cases = result["by_metric"]["strict_exact"]
    assert all_cases["denominator"] == 6
    assert all_cases["improved_cases"] == 3
    assert all_cases["regressed_cases"] == 1
    assert all_cases["discordant_cases"] == 4
    assert all_cases["unchanged_cases"] == 2
    supported = result["by_metric"]["supported_raw_citation_id"]
    assert supported["denominator"] == 3
    assert supported["baseline_count"] == 1
    assert supported["candidate_count"] == 1
    abstention = result["by_metric"]["unsupported_correct_abstention"]
    assert abstention["denominator"] == 3
    assert abstention["baseline_count"] == 1
    assert abstention["candidate_count"] == 2
    claim = result["by_metric"]["unsupported_claim"]
    assert claim["beneficial_direction"] == "lower"
    assert claim["denominator"] == 3
    assert claim["improved_cases"] == 1
    assert claim["regressed_cases"] == 0
    assert claim["beneficial_delta_percentage_points"] == pytest.approx(33.33)


def test_language_slices_retain_matched_case_counts():
    report = analyze_scored_pairs(example_pairs())
    assert set(report["by_language"]) == {"en", "no", "ar"}
    assert set(report["by_category"]) == {"supported", "unknown"}
    assert {k: v["cases"] for k, v in report["by_language"].items()} == {
        "en": 2, "no": 2, "ar": 2,
    }
    assert report["by_language"]["en"]["metrics"]["supported_answer_exact"]["denominator"] == 1
    assert report["by_language"]["en"]["metrics"]["unsupported_correct_abstention"]["denominator"] == 1
    assert report["by_category"]["supported"]["metrics"]["unsupported_claim"]["denominator"] == 0
    assert report["by_category"]["supported"]["metrics"]["unsupported_claim"]["exploratory_exact_mcnemar_two_sided_p"] is None


def test_no_discordances_is_not_reported_as_equivalence():
    rows = example_pairs()
    for item in rows.values():
        item["candidate"] = dict(item["baseline"])
    result = analyze_scored_pairs(rows)
    strict = result["by_metric"]["strict_exact"]
    assert strict["discordant_cases"] == 0
    assert strict["exploratory_exact_mcnemar_two_sided_p"] == 1.0
    assert strict["conditional_improvement_share_wilson_95_pct"] is None
    assert result["statistical_superiority_established"] is False
    assert result["multiple_testing_adjusted"] is False


def test_one_sided_gain_is_still_two_sided_exploratory_exact_test():
    rows = {}
    for i in range(6):
        sample = example_pairs()["SYNTHETIC-1"]
        sample = deepcopy(sample)
        cid = f"FIXTURE-{i}"
        sample["baseline"]["id"] = cid
        sample["candidate"]["id"] = cid
        sample["baseline"]["correct"] = 0
        sample["candidate"]["correct"] = 1
        rows[cid] = sample
    result = analyze_scored_pairs(rows)
    metric = result["by_metric"]["strict_exact"]
    assert metric["improved_cases"] == 6
    assert metric["regressed_cases"] == 0
    assert math.isclose(metric["exploratory_exact_mcnemar_two_sided_p"], 0.03125)
    lo, hi = metric["conditional_improvement_share_wilson_95_pct"]
    assert 0 < lo < hi <= 100
    assert result["statistical_superiority_established"] is False


@pytest.mark.parametrize("mutation", [
    lambda x: x["baseline"].update(correct="1"),
    lambda x: x["candidate"].update(correct=True),
    lambda x: x["candidate"].update(correct=2),
    lambda x: x["candidate"].update(lang="wrong"),
    lambda x: x["candidate"].update(abstention_needed=1 - x["candidate"]["abstention_needed"]),
    lambda x: x["baseline"].update(id="duplicate-id"),
])
def test_malformed_or_swapped_scores_fail_closed(mutation):
    rows = example_pairs()
    mutation(rows["SYNTHETIC-1"])
    with pytest.raises(ValueError):
        analyze_scored_pairs(rows)


def test_real_data_missing_raw_prediction_or_duplicate_id_refused_without_gpu():
    # This exercise deliberately avoids producing any plausible real outputs.
    rows = [
        {"id": "CASE-A", "language": "en", "category": "supported"},
        {"id": "CASE-B", "language": "no", "category": "unsupported"},
    ]
    original = {"predictions": [
        {"id": "CASE-A", "raw": "{}"},
        {"id": "CASE-A", "raw": "{}"},
    ]}
    other = {"predictions": [
        {"id": "CASE-A", "raw": "{}"},
        {"id": "CASE-B", "raw": "{}"},
    ]}
    with pytest.raises(ValueError, match="Duplicated or malformed"):
        diagnostics_from_raw_reports(rows, original, other)
    original["predictions"][1]["id"] = "CASE-B"
    other["predictions"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        diagnostics_from_raw_reports(rows, original, other)
