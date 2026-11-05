"""Source-reproducible, descriptive paired diagnostics for a future final 60-case run.

No inference, training, model loading or new data is performed here. Statistics
are computed from precisely matched saved *raw* outputs and deterministic
scorer results. Exact discordance tests and conditional Wilson intervals are
exploratory, NOT evidence of human-validated model superiority, and are not
adjusted for multiple comparisons. Very small language/category slices have
low power; zero discordances are not proof of equivalence.
"""
from __future__ import annotations

from policyproof.core import score
from scripts.paired_guard_uncertainty import exact_discordance_p, wilson_interval


# Metric names deliberately specify their denominators: unsupported cases
# must not dilute supported-citation and abstention-specific rates.
METRICS = (
    ("strict_exact", "correct", None, 1),
    ("schema_valid", "format_valid", None, 1),
    ("supported_answer_exact", "supported_answer_correct", "supported_total", 1),
    ("supported_raw_citation_id", "raw_citation_correct", "supported_total", 1),
    ("unsupported_correct_abstention", "abstention_correct", "abstention_needed", 1),
    ("unsupported_claim", "unsupported_claim", "abstention_needed", -1),
)


def _binary(record: dict, key: str) -> int:
    value = record.get(key)
    if type(value) is not int or value not in (0, 1):
        raise ValueError("Paired statistical input must be binary: " + key)
    return value


def _metric_cases(records: list[dict], field: str, denominator: str | None,
                  direction: int) -> dict:
    paired = []
    for record in records:
        before = record["baseline"]
        after = record["candidate"]
        if denominator is not None:
            b_denom, a_denom = _binary(before, denominator), _binary(after, denominator)
            if b_denom != a_denom:
                raise ValueError("Paired score denominator changed between same frozen case")
            if b_denom == 0:
                continue
        b, a = _binary(before, field), _binary(after, field)
        paired.append((b, a))
    n = len(paired)
    b_count = sum(b for b, _ in paired)
    a_count = sum(a for _, a in paired)
    improved = sum((a - b) * direction == 1 for b, a in paired)
    regressed = sum((a - b) * direction == -1 for b, a in paired)
    discordant = improved + regressed
    return {
        "denominator": n,
        "baseline_count": b_count,
        "candidate_count": a_count,
        "baseline_rate_pct": round(100 * b_count / n, 2) if n else None,
        "candidate_rate_pct": round(100 * a_count / n, 2) if n else None,
        "beneficial_delta_percentage_points": round(
            direction * 100 * (a_count - b_count) / n, 2
        ) if n else None,
        "beneficial_direction": "higher" if direction == 1 else "lower",
        "improved_cases": improved,
        "regressed_cases": regressed,
        "unchanged_cases": n - discordant,
        "discordant_cases": discordant,
        "exploratory_exact_mcnemar_two_sided_p": round(
            exact_discordance_p(improved, regressed), 8
        ) if n else None,
        "conditional_improvement_share_wilson_95_pct": wilson_interval(
            improved, discordant
        ) if discordant else None,
    }


def _group(records: list[dict]) -> dict:
    return {
        label: _metric_cases(records, field, denom, direction)
        for label, field, denom, direction in METRICS
    }


def analyze_scored_pairs(cases: dict[str, dict]) -> dict:
    """Compute reproducible diagnostics from an already scored, ID-matched pair.

    Each dictionary entry is one distinct frozen case with two score dicts.
    The deterministic structure intentionally excludes model identity fields;
    raw source validation belongs to the calling evidence gate.
    """
    if not isinstance(cases, dict) or not cases:
        raise ValueError("Paired diagnostic data must contain distinct case IDs")
    records = []
    for case_id in sorted(cases):
        record = cases[case_id]
        if (not isinstance(case_id, str) or not case_id
            or not isinstance(record, dict)
            or not isinstance(record.get("baseline"), dict)
            or not isinstance(record.get("candidate"), dict)
            or not isinstance(record.get("lang"), str) or not record["lang"]
            or not isinstance(record.get("category"), str) or not record["category"]):
            raise ValueError("Malformed paired case records")
        if (record["baseline"].get("id") != case_id
            or record["candidate"].get("id") != case_id
            or record["baseline"].get("lang") != record["lang"]
            or record["candidate"].get("lang") != record["lang"]
            or record["baseline"].get("category") != record["category"]
            or record["candidate"].get("category") != record["category"]):
            raise ValueError("Case ID or subgroup changed between paired scores")
        records.append(record)
    languages = sorted({r["lang"] for r in records})
    categories = sorted({r["category"] for r in records})
    return {
        "schema": "policyproof_final_case_matched_exploratory_diagnostics_v1",
        "status": "DESCRIPTIVE_ONLY_NOT_MODEL_QUALITY_PROOF",
        "sample_count": len(records),
        "by_metric": _group(records),
        "by_language": {
            lang: {"cases": sum(r["lang"] == lang for r in records),
                   "metrics": _group([r for r in records if r["lang"] == lang])}
            for lang in languages
        },
        "by_category": {
            cat: {"cases": sum(r["category"] == cat for r in records),
                  "metrics": _group([r for r in records if r["category"] == cat])}
            for cat in categories
        },
        "multiple_testing_adjusted": False,
        "independent_human_semantic_accuracy_verified": False,
        "statistical_superiority_established": False,
        "limitations": [
            "Deterministic exact-string and citation-ID scoring does not establish semantic correctness",
            "Exploratory two-sided McNemar tests use only discordant matched cases; p=1 when none disagree is not proof of equivalence",
            "Wilson intervals describe beneficial fractions among CHANGED cases, NOT confidence limits on overall accuracy difference",
            "Language and category slices may be tiny; no multiplicity correction, independence check, or safety certification",
            "Public synthetic candidate is not a hidden, independently sampled real-world benchmark",
            "No model inference, GPU attestation, training, or human review is performed by these diagnostics",
        ],
    }


def diagnostics_from_raw_reports(rows: list[dict], base: dict, tuned: dict) -> dict:
    """Reproduce full paired diagnostics from 2x original saved model responses."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Cannot score an empty frozen benchmark")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Repeated IDs in frozen benchmark")
    mappings = []
    for report in (base, tuned):
        predictions = report.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(rows):
            raise ValueError("Incomplete paired raw prediction coverage")
        mapping = {}
        for p in predictions:
            if (not isinstance(p, dict) or not isinstance(p.get("id"), str)
                or not isinstance(p.get("raw"), str) or p["id"] in mapping):
                raise ValueError("Duplicated or malformed raw paired response")
            mapping[p["id"]] = p["raw"]
        if set(mapping) != {r["id"] for r in rows}:
            raise ValueError("Raw paired case coverage differs from frozen dataset")
        mappings.append(mapping)
    baseline, candidate = mappings
    return analyze_scored_pairs({
        row["id"]: {
            "lang": row["language"],
            "category": row["category"],
            "baseline": score(row, baseline[row["id"]]),
            "candidate": score(row, candidate[row["id"]]),
        }
        for row in rows
    })
