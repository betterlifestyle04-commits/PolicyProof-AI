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


@dataclass(frozen=True)
class GuardDecision:
    output: str
    action: str


def canonical_abstention() -> str:
    return json.dumps(
        {"answer": ABSTAIN, "source_id": None, "abstain": True},
        ensure_ascii=False, separators=(",", ":")
    )


def policy_source_ids(context: str) -> set[str]:
    """Identifiers at the start of trusted-looking policy lines only.

    This is a syntactic source filter, NOT a semantic or cryptographic
    authenticity verifier. The caller must supply trusted policy context;
    attacker-controlled text must not be appended as a trusted policy.
    """
    return set(re.findall(r"(?m)^\[(POL-[A-Za-z0-9-]+)\]\s+", context))


def safe_output(raw: str, *, context: str) -> GuardDecision:
    """Fail closed on invalid schema/abstention/source; never use gold answers."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
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
    if source not in policy_source_ids(context):
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
            "Guard only validates output schema, source-ID syntax, and abstention consistency",
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
        json.loads(baseline_file.read_text(encoding="utf-8")),
        read_jsonl(data_file), sha256(data_file),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
