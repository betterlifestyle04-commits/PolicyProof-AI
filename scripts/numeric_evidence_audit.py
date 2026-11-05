"""Conservative, source-local numeric *risk flagger* for fictional policies.

Runs on previously saved model responses only. It never uses a gold answer,
expected policy ID, human vote, model inference or an LLM. It does NOT prove
the truth of numbers that happen to appear in a cited policy: the same number
can be attached to an entirely different product, term or unit.

Default output is a human-review signal, not a forced model abstention.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.auditkit_replay import check_saved_outputs
from scripts.inference_guard import safe_output, trusted_policy_blocks

NUMERIC = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[ ,.\u00a0\u202f\u066c]\d{3})+|\d+(?:[.,\u066b]\d+)?)(?::\d{2})?(?!\w)",
    flags=re.UNICODE,
)
ENGLISH_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14",
    "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def _ascii_digits(value: str) -> str:
    out = []
    for c in value:
        if c.isdecimal():
            try:
                out.append(str(unicodedata.decimal(c)))
            except ValueError:
                out.append(c)
        else:
            out.append(c)
    return "".join(out)


def _normalize_numeric(token: str) -> str:
    """Normalize numeral scripts, grouped thousands and basic decimal formats."""
    token = _ascii_digits(token).replace("\u00a0", " ").replace("\u202f", " ")
    token = token.replace("\u066c", ",").replace("\u066b", ".")
    if ":" in token:
        hour, minute = token.split(":", 1)
        return f"{int(hour)}:{minute}"
    compact = token.replace(" ", "")
    # Thousands notation: 5,000 or 5.000; repeated grouping likewise.
    if re.fullmatch(r"\d{1,3}(?:[,.]\d{3})+", compact):
        return str(int(compact.replace(",", "").replace(".", "")))
    if re.fullmatch(r"\d+(?:[.,]\d+)?", compact):
        if "," in compact:
            compact = compact.replace(",", ".")
        if "." in compact:
            whole, frac = compact.split(".")
            return f"{int(whole)}.{frac.rstrip('0') or '0'}"
        return str(int(compact))
    return compact


def number_literals(text: str, *, include_english_words: bool = False) -> set[str]:
    if not isinstance(text, str):
        return set()
    result = {_normalize_numeric(m.group(0)) for m in NUMERIC.finditer(text)}
    if include_english_words:
        for word in re.findall(r"(?u)\b[a-z]+\b", text.casefold()):
            if word in ENGLISH_NUMBERS:
                result.add(ENGLISH_NUMBERS[word])
    return result


def inspect_numeric_evidence(raw: str, *, context: str) -> dict:
    """Audit literal answer numbers against the cited source, not all sources.

    An unmatched number is a REVIEW FLAG, not proof of hallucination. A match is
    NOT proof of grounding, semantic equivalence, correct units or completeness.
    """
    decision = safe_output(raw, context=context)
    if decision.action != "valid_supported_shape":
        ambiguous = decision.action == "ambiguous_policy_source"
        return {
            "status": ("source_boundary_ambiguous" if ambiguous
                       else "no_supported_answer_to_check"),
            "guard_action": decision.action,
            "source_id": None, "answer_literals": [],
            "source_literals": [], "unmatched_literals": [],
            "human_review_required": ambiguous,
            "factual_support_verified": False,
        }
    parsed = json.loads(decision.output)
    source_id = parsed["source_id"]
    blocks = trusted_policy_blocks(context)
    if source_id not in blocks:
        return {
            "status": "source_boundary_ambiguous",
            "guard_action": decision.action,
            "source_id": source_id,
            "answer_literals": sorted(number_literals(parsed["answer"])),
            "source_literals": [], "unmatched_literals": [],
            "human_review_required": True,
            "factual_support_verified": False,
        }
    answer_nums = number_literals(parsed["answer"])
    evidence_nums = number_literals(blocks[source_id], include_english_words=True)
    missing = answer_nums - evidence_nums
    status = ("unmatched_literal_requires_review" if missing
              else "literal_present_but_semantics_unverified" if answer_nums
              else "no_numeric_literal_in_answer")
    return {
        "status": status,
        "guard_action": decision.action,
        "source_id": source_id,
        "answer_literals": sorted(answer_nums),
        "source_literals": sorted(evidence_nums),
        "unmatched_literals": sorted(missing),
        "human_review_required": bool(missing),
        "factual_support_verified": False,
    }


def audit_saved_predictions(report: dict, cases: list[dict], data_hash: str) -> dict:
    """Compute reviewer triage from the saved model, not from reference labels."""
    mapping = check_saved_outputs(report, cases, data_hash)
    if len(cases) != 48 or len({r["id"] for r in cases}) != 48:
        raise ValueError("Expected 48 unique synthetic development cases")
    if report.get("split") != "unseen-policy":
        raise ValueError("Not the inspected development experiment")
    details = []
    counter: Counter[str] = Counter()
    for case in cases:
        check = inspect_numeric_evidence(mapping[case["id"]], context=case["context"])
        counter[check["status"]] += 1
        if check["human_review_required"]:
            details.append({
                "id": case["id"], "language": case["language"],
                "category": case["category"], "source_id": check["source_id"],
                "status": check["status"],
                "unmatched_literals": check["unmatched_literals"],
            })
    return {
        "status": "saved_dev48_numeric_risk_diagnostics_only",
        "dataset_sha256": data_hash,
        "samples": len(cases),
        "model_reported": report.get("model"),
        "model_revision_reported": report.get("model_revision"),
        "model_tuned_reported": report.get("is_finetuned"),
        "model_inference_executed": False,
        "training_performed": False,
        "human_semantic_review_completed": False,
        "factual_accuracy_verified": False,
        "reference_answers_or_expected_source_ids_used": False,
        "numeric_review_flags": len(details),
        "statuses": dict(sorted(counter.items())),
        "flagged_cases": details,
        "limitations": [
            "An in-source number may still refer to a different charge, condition or time unit",
            "This does not detect unsupported nonnumeric claims or spelled-out numbers in responses",
            "A numeric paraphrase or conversion can be falsely flagged",
            "An attacker could control text before the synthetic untrusted-note boundary; real sources must be authenticated",
            "The 48 synthetic development cases were inspected; do not call this a blind final benchmark",
            "Results are replayed from saved predictions without independent CUDA execution",
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=ROOT / "artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json")
    p.add_argument("--dataset", type=Path, default=ROOT / "data/unseen_policy_test.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/dev48_numeric_risk_audit.json")
    args = p.parse_args()
    if args.output.resolve() in {args.report.resolve(), args.dataset.resolve()}:
        raise ValueError("Do not overwrite original source evidence")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = audit_saved_predictions(report, read_jsonl(args.dataset), sha256(args.dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "samples": result["samples"],
        "numeric_review_flags": result["numeric_review_flags"],
        "statuses": result["statuses"], "model_inference_executed": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
