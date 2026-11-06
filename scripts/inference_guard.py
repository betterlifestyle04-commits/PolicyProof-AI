"""A conservative, model-agnostic output safety gate for PolicyProof AI.

Transforms only saved model outputs and trusted source-ID syntax. It never
consults reference answers, evaluation labels, or the correct source ID.
It does not verify the factual/semantic truth of an answer and does not
improve model weights. Untrusted source instructions are not followed.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import re

from policyproof.core import ABSTAIN, read_jsonl, sha256, score, aggregate
from scripts.auditkit_replay import check_saved_outputs
from scripts.verify_evidence import read_evidence_json
from scripts.research_report_io import preserve_deterministic_report


@dataclass(frozen=True)
class GuardDecision:
    output: str
    action: str


def canonical_abstention() -> str:
    return json.dumps(
        {"answer": ABSTAIN, "source_id": None, "abstain": True},
        ensure_ascii=False, separators=(",", ":")
    )


MAX_CONTEXT_CHARS = 20_000
MAX_OUTPUT_CHARS = 10_000
_UNTRUSTED_MARKER = re.compile(r"\[\s*UNTRUSTED[\s_-]+NOTE\s*\]", re.IGNORECASE)


class DuplicateJSONKeyError(ValueError):
    """Ambiguous duplicate JSON members must not be silently overwritten."""


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str):
    raise ValueError(f"Non-standard JSON constant: {value}")


_POLICY_HEADER = re.compile(r"(?m)^\[(POL-[A-Za-z0-9-]+)\][ \t]+")


def _policy_inventory(context: str) -> tuple[dict[str, str], set[str]]:
    """One trusted policy block per ID; duplicates/empty blocks are ambiguous.

    Input remains fictional benchmark text; only authenticated source stores
    can establish provenance in a real application. Do not accept policy
    headers following the first [UNTRUSTED-NOTE] boundary.
    """
    if not isinstance(context, str) or len(context) > MAX_CONTEXT_CHARS:
        return {}, set()
    trusted = _UNTRUSTED_MARKER.split(context, maxsplit=1)[0]
    headers = list(_POLICY_HEADER.finditer(trusted))
    blocks: dict[str, str] = {}
    seen: set[str] = set()
    ambiguous: set[str] = set()
    for i, match in enumerate(headers):
        source_id = match.group(1)
        end = headers[i + 1].start() if i + 1 < len(headers) else len(trusted)
        body = trusted[match.end():end].strip()
        if source_id in seen or not body:
            ambiguous.add(source_id)
        else:
            blocks[source_id] = body
        seen.add(source_id)
    return {sid: text for sid, text in blocks.items() if sid not in ambiguous}, ambiguous


def trusted_policy_blocks(context: str) -> dict[str, str]:
    """Canonical guard/audit source parser; empty or duplicate IDs excluded."""
    return _policy_inventory(context)[0]


def policy_source_ids(context: str) -> set[str]:
    """Source IDs accepted only when a nonempty, uniquely identified block exists.

    A syntactic fixture check; not independent authentication of source text.
    """
    return set(trusted_policy_blocks(context))


def safe_output(raw: str, *, context: str) -> GuardDecision:
    """Fail closed on invalid schema/abstention/source; never use gold answers."""
    # Apply the same bounds as the public UI at the library/API boundary.
    # This avoids excessively large or deeply nested untrusted inputs and
    # makes direct callers fail closed as well.
    if (isinstance(raw, str) and len(raw) > MAX_OUTPUT_CHARS) or (
        isinstance(context, str) and len(context) > MAX_CONTEXT_CHARS
    ):
        return GuardDecision(canonical_abstention(), "input_too_large")
    try:
        parsed = json.loads(
            raw, object_pairs_hook=_unique_json_object,
            parse_constant=_reject_nonfinite,
        )
    except DuplicateJSONKeyError:
        return GuardDecision(canonical_abstention(), "duplicate_json_keys")
    except (TypeError, ValueError, RecursionError):
        return GuardDecision(canonical_abstention(), "invalid_json")
    if not isinstance(parsed, dict) or set(parsed) != {"answer", "source_id", "abstain"}:
        return GuardDecision(canonical_abstention(), "invalid_schema")
    answer, source, abstain = parsed["answer"], parsed["source_id"], parsed["abstain"]
    if not isinstance(answer, str) or not isinstance(abstain, bool) or (
        source is not None and not isinstance(source, str)
    ):
        return GuardDecision(canonical_abstention(), "invalid_types")

    if abstain is True:
        if answer == ABSTAIN and source is None:
            return GuardDecision(canonical_abstention(), "valid_abstention")
        return GuardDecision(canonical_abstention(), "contradictory_abstention")

    # The asserted source must be a policy ID in the supplied context.
    # A citation to [UNTRUSTED-NOTE] fails, even if the model used one.
    if not answer.strip() or answer == ABSTAIN or source is None:
        return GuardDecision(canonical_abstention(), "unsupported_claim_shape")
    if not re.fullmatch(r"POL-[A-Za-z0-9-]+", source):
        return GuardDecision(canonical_abstention(), "non_policy_source")
    blocks, ambiguous = _policy_inventory(context)
    if source in ambiguous:
        return GuardDecision(canonical_abstention(), "ambiguous_policy_source")
    if source not in blocks:
        return GuardDecision(canonical_abstention(), "unrecognized_source")

    # Preserve verbatim answer: no claims about semantic evidence support.
    output = json.dumps(
        {"answer": answer, "source_id": source, "abstain": False},
        ensure_ascii=False, separators=(",", ":")
    )
    return GuardDecision(output, "valid_supported_shape")


def evaluate_saved_guard(baseline: dict, rows: list[dict], dataset_sha: str) -> dict:
    """Genuine paired deterministic evaluation on the SAME saved model outputs.

    "After" is guarded output from the same base model, NOT a fine-tuned model.
    This benchmark was inspected during development; not a blind holdout.
    """
    if baseline.get("is_finetuned") is not False:
        raise ValueError("A baseline must use the original untuned model")
    before = check_saved_outputs(baseline, rows, dataset_sha)
    actions: Counter[str] = Counter()
    transformed: list[dict] = []
    original_scores, guarded_scores = [], []
    for row in rows:
        raw = before[row["id"]]
        # Only context, not expected_answer / expected_source_id / labels, is
        # available to safe_output, preventing an oracle-based "improvement".
        decision = safe_output(raw, context=row["context"])
        actions[decision.action] += 1
        transformed.append({"id": row["id"], "raw": decision.output, "action": decision.action})
        original_scores.append(score(row, raw))
        guarded_scores.append(score(row, decision.output))
    return {
        "status": "completed",
        "experiment_type": "inference_time_output_guard",
        "training_performed": False,
        "model_weights_updated": False,
        "new_model_inference_performed": False,
        "model_report_label": baseline.get("model"),
        "original_model_revision": baseline.get("model_revision"),
        "data_sha256": dataset_sha,
        "samples": len(rows),
        "before": aggregate(original_scores),
        "after": aggregate(guarded_scores),
        "answer_coverage": {
            "original_answered_cases": sum(x["answered"] for x in original_scores),
            "guarded_answered_cases": sum(x["answered"] for x in guarded_scores),
            "original_answered_supported_cases": sum(
                x["answered"] for x in original_scores if x["supported_total"]
            ),
            "guarded_answered_supported_cases": sum(
                x["answered"] for x in guarded_scores if x["supported_total"]
            ),
            "total_supported_cases": sum(x["supported_total"] for x in original_scores),
        },
        "decision_counts": dict(sorted(actions.items())),
        "guarded_predictions": transformed,
        "guard_limitations": [
            "Guard validates output schema, abstention consistency and unique nonempty source-ID blocks; not semantic support",
            "It cannot prove that any remaining answer is factually supported by the cited text",
            "Fail-closed abstentions may reduce answer coverage; the trade-off must be shown",
            "Development dataset has been inspected; this is NOT a blind final benchmark",
            "Original model-revision SHA was not recorded in the user's saved baseline",
            "This is not a fine-tuned model or proof of model safety improvement",
        ],
    }


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate zero-GPU inference-time guard")
    p.add_argument("--baseline", required=True)
    p.add_argument("--dataset", default="data/unseen_policy_test.jsonl")
    p.add_argument("--output", default="artifacts/guarded_development_evaluation.json")
    args = p.parse_args()
    baseline_file, data_file, output = (
        Path(args.baseline), Path(args.dataset), Path(args.output)
    )
    if output.resolve() in (baseline_file.resolve(), data_file.resolve()):
        raise ValueError("Refusing to overwrite original research evidence")
    result = evaluate_saved_guard(
        read_evidence_json(baseline_file),
        read_jsonl(data_file), sha256(data_file),
    )
    preserve_deterministic_report(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        output,
    )
    # Concise legitimate paired development diagnostics; no quality assertions.
    print(json.dumps({
        "status": result["status"], "type": result["experiment_type"],
        "cases": result["samples"], "decisions": result["decision_counts"],
        "before": result["before"], "after": result["after"],
        "no_new_model_inference": not result["new_model_inference_performed"],
    }, ensure_ascii=False, indent=2))
    print("Saved guarded development evaluation:", output)


if __name__ == "__main__":
    main()
