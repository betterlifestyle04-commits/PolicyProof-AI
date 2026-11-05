"""Conservative question-to-cited-source *topic mismatch* review diagnostics.

The numerical review checks literal values, but a correctly copied fee may
still answer a completely different banking question. This narrow, explicit
English/Norwegian/Arabic lexicon flags only a clear single-topic mismatch.
Neither source-ID presence nor same topic proves any answer is factually true.

Never use this as a model guard, final benchmark result, oracle, or automatic
decision to reject or approve a banking response. No new model is run.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.auditkit_replay import check_saved_outputs
from scripts.inference_guard import safe_output
from scripts.numeric_evidence_audit import trusted_policy_blocks

# Explicit, purposefully small and auditable rulebook. These patterns cover
# the invented research policy universe rather than arbitrary real banking
# products. ALL languages use the same topic keys. Only classifiable questions
# with a UNIQUE topic and cited sources with a DIFFERENT unique topic are
# flagged. Missing or ambiguous matches never become a green correctness vote.
TOPIC_RULES = {
    "overdraft": (
        r"\boverdraft\b|\boverdrawn\b|\bovertrekk\w*|"
        r"السحب\s+على\s+المكشوف",
    ),
    "mortgage": (
        r"\bmortgage\b|\bhome\s+loan\b|\bboliglån\w*|"
        r"الرهن\s+العقاري|القرض\s+العقاري",
    ),
    "investment_returns": (
        r"\binvestments?\b|\binvesting\b|\b(?:investment\s+returns?)\b|"
        r"\binvesterings\w*|(?:عوائد\s+الاستثمار)|الاستثمار",
    ),
    "international_atm": (
        r"\b(?:foreign|international|overseas)\b.{0,35}"
        r"\b(?:atm|cash\s+withdrawal)\b|"
        r"\b(?:atm|cash\s+withdrawal)\b.{0,35}"
        r"\b(?:foreign|international|overseas)\b|"
        r"\butenlandsk\w*\s+minibank\w*|"
        r"\bminibank\w*.{0,40}\butlandet\b|"
        r"السحب\s+النقدي\s+في\s+الخارج|"
        r"الصراف\s+الآلي.{0,35}الخارج",
    ),
    "savings_withdrawal": (
        r"\b(?:flex\s+)?savings\b|\bsavings\s+account\b|"
        r"\bsparekonto\w*|\bflex-sparekonto\w*|"
        r"حساب\s+التوفير|الادخار",
    ),
    "virtual_card": (
        r"\bvirtual\s+(?:debit\s+)?cards?\b|"
        r"\bvirtuell\w*\s+kort\b|"
        r"بطاق(?:ة|ات|تين)\s+افتراضي\w*",
    ),
    "mailing_address": (
        r"\bmailing\s+address\b|\baddress\s+change\b|"
        r"\bpostadresse\w*|العنوان\s+البريدي",
    ),
    "contactless": (
        r"\bcontactless\b|\bkontaktløs\w*|اللاتلامسي",
    ),
    "scheduled_payment": (
        r"\bscheduled\s+(?:domestic\s+)?(?:payments?|transfers?)\b|"
        r"\bplanlagt\w*\s+(?:innenlandsbetaling\w*|overføring\w*)|"
        r"مجدول",
    ),
    "international_beneficiary": (
        r"\b(?:new|newly\s+added)\s+international\s+"
        r"(?:beneficiar(?:y|ies)|payees?)\b|"
        r"\bny\w*\s+utenlandsmottaker\w*|"
        r"مستفيد\s+دولي",
    ),
}
# Rule expressions are deliberately stored as one-item tuples so that
# adjacent raw string fragments can be visually grouped per product concept.
# Compile the contained expression, never the tuple container itself.
_COMPILED = {
    key: re.compile(parts[0], re.IGNORECASE | re.UNICODE)
    for key, parts in TOPIC_RULES.items()
}


def detected_topics(text: str) -> list[str]:
    """Pure lexical hints, not a semantic text classifier or entailment."""
    if not isinstance(text, str) or len(text) > 20000:
        return []
    return sorted(name for name, pattern in _COMPILED.items()
                  if pattern.search(text))


def inspect_citation_topic(raw: str, *, question: str, context: str) -> dict:
    """Flag a clear single-topic question/source mismatch for human inspection.

    Only user question, saved raw model text and its specifically cited policy
    source are read. No gold answer, expected source ID, training split label,
    oracle, model inference, or outside document is accessed.
    """
    decision = safe_output(raw, context=context)
    result = {
        "status": "no_supported_answer_to_check",
        "guard_action": decision.action,
        "source_id": None,
        "question_topics": [],
        "cited_source_topics": [],
        "human_review_required": False,
        "factual_support_verified": False,
        "topic_match_is_not_semantic_entailment": True,
    }
    if decision.action != "valid_supported_shape":
        if decision.action == "ambiguous_policy_source":
            result["status"] = "source_boundary_ambiguous"
            result["human_review_required"] = True
        return result
    parsed = json.loads(decision.output)
    result["source_id"] = parsed["source_id"]
    blocks = trusted_policy_blocks(context)
    if parsed["source_id"] not in blocks:
        result["status"] = "source_boundary_ambiguous"
        result["human_review_required"] = True
        return result
    q_topics = detected_topics(question)
    source_topics = detected_topics(blocks[parsed["source_id"]])
    result["question_topics"] = q_topics
    result["cited_source_topics"] = source_topics
    if not q_topics or not source_topics:
        result["status"] = "topic_classification_incomplete"
    elif len(q_topics) != 1 or len(source_topics) != 1:
        result["status"] = "topic_classification_ambiguous"
    elif q_topics[0] != source_topics[0]:
        result["status"] = "question_cited_source_topic_mismatch_requires_review"
        result["human_review_required"] = True
    else:
        result["status"] = "topic_match_but_semantics_unverified"
    return result


def audit_saved_source_topics(report: dict, cases: list[dict], data_sha: str) -> dict:
    """Validate source report, then screen only question and cited policy text."""
    saved = check_saved_outputs(report, cases, data_sha)
    if (len(cases) != 48 or len({row.get("id") for row in cases}) != 48
        or report.get("split") != "unseen-policy"):
        raise ValueError("Expected original 48 unique synthetic development cases")
    statuses: Counter[str] = Counter()
    flagged = []
    for case in cases:
        if not isinstance(case.get("question"), str):
            raise ValueError("Saved case is missing its question")
        finding = inspect_citation_topic(
            saved[case["id"]], question=case["question"],
            context=case["context"],
        )
        statuses[finding["status"]] += 1
        if finding["human_review_required"]:
            flagged.append({
                "id": case["id"],
                "language": case["language"],
                "category": case["category"],
                "source_id": finding["source_id"],
                "question_topics": finding["question_topics"],
                "cited_source_topics": finding["cited_source_topics"],
                "status": finding["status"],
            })
    return {
        "status": "saved_dev48_topic_relevance_diagnostics_only",
        "samples": 48,
        "dataset_sha256": data_sha,
        "model_reported": report.get("model"),
        "model_revision_reported": report.get("model_revision"),
        "model_tuned_reported": report.get("is_finetuned"),
        "statuses": dict(sorted(statuses.items())),
        "topic_mismatch_or_ambiguous_source_review_flags": len(flagged),
        "flagged_cases": flagged,
        "topic_detection_uses_gold_answer_or_expected_source": False,
        "source_report_scores_validated_against_original_fixture": True,
        "new_inference_executed": False,
        "training_performed": False,
        "independent_semantic_review_completed": False,
        "factual_accuracy_verified": False,
        "rulebook_topics": sorted(TOPIC_RULES),
        "limitations": [
            "Small, manually defined domain lexicon; unknown phrases and product names are not covered",
            "Same topic is NOT factual support, nor proof of correct numbers, conditions, units or translations",
            "Different topic is a human-review prompt, not proof of hallucination",
            "Questions can ask about multiple products; ambiguous cases are not automatically marked wrong",
            "This is syntactic source matching on fictional text, not authenticated document provenance",
            "All 48 cases were previously inspected; this is not a blind final benchmark",
            "No GPU model inference, training or independently verified human review was performed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT /
                        "artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json")
    parser.add_argument("--dataset", type=Path, default=ROOT /
                        "data/unseen_policy_test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT /
                        "artifacts/dev48_topic_relevance_audit.json")
    args = parser.parse_args()
    protected = {args.report.resolve(), args.dataset.resolve()}
    if args.output.resolve() in protected:
        raise ValueError("Refusing to overwrite original saved model or dataset evidence")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = read_jsonl(args.dataset)
    result = audit_saved_source_topics(report, rows, sha256(args.dataset))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(
        result, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "samples": result["samples"],
        "review_flags": result["topic_mismatch_or_ambiguous_source_review_flags"],
        "statuses": result["statuses"],
        "new_inference_executed": False,
        "factual_accuracy_verified": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
