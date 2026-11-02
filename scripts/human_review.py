"""Export a blind semantic-review CSV from real baseline inference output.

The template intentionally omits expected answers. Two independent reviewers
should rate factual correctness and evidence grounding. This script never
automatically claims a model is correct.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from policyproof.core import read_jsonl

HEADERS = [
    "id", "language", "category", "question", "provided_evidence",
    "raw_model_response", "reviewer_id", "semantic_correct",
    "evidence_supported", "safety_concern", "reviewer_notes",
]


def export_blind_review(report: dict, cases: list[dict], *, reviewer_id: str = "") -> list[dict]:
    if report.get("engine") != "transformers" or report.get("is_real_model_result") is not True:
        raise ValueError("Human review requires actual Transformers inference, not mocks or imported files")
    predictions = report.get("predictions", [])
    if len(predictions) != len(cases) or len({x.get("id") for x in predictions}) != len(predictions):
        raise ValueError("Prediction IDs are incomplete or duplicated")
    by_id = {x["id"]: x.get("raw") for x in predictions}
    if set(by_id) != {x["id"] for x in cases}:
        raise ValueError("Prediction IDs do not match expected cases")
    return [{
        "id": x["id"], "language": x["language"], "category": x["category"],
        "question": x["question"], "provided_evidence": x["context"],
        "raw_model_response": by_id[x["id"]], "reviewer_id": reviewer_id,
        "semantic_correct": "", "evidence_supported": "",
        "safety_concern": "", "reviewer_notes": "",
    } for x in cases]


def review_metrics(rows: list[dict]) -> dict:
    """Count only human-completed rows; never fill missing verdicts as negatives."""
    metrics = {}
    for field in ("semantic_correct", "evidence_supported", "safety_concern"):
        values = [r.get(field, "").strip() for r in rows]
        invalid = [v for v in values if v and v not in ("0", "1")]
        if invalid:
            raise ValueError(f"{field}: only 0, 1, or blank allowed")
        filled = [v for v in values if v]
        metrics[field] = {
            "reviewed": len(filled),
            "total": len(rows),
            "positive": sum(v == "1" for v in filled),
            "percentage_of_reviewed": (round(100 * sum(v == "1" for v in filled) / len(filled), 2)
                                      if filled else None),
        }
    return {"human_assessment_only": True, "metrics": metrics,
            "limitation": "Ratings are human judgments and must be independently audited; reviewer agreement not estimated."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", help="Full real-model JSON report")
    parser.add_argument("--cases", default="data/unseen_policy_test.jsonl")
    parser.add_argument("--output", default="artifacts/blind_review.csv")
    parser.add_argument("--reviewer", default="", help="Reviewer pseudonym (no personal identifiers)")
    parser.add_argument("--summarize", help="Existing completed reviewer CSV, instead of exporting")
    args = parser.parse_args()
    if args.summarize:
        with open(args.summarize, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        print(json.dumps(review_metrics(rows), indent=2))
        return
    if not args.predictions:
        parser.error("--predictions is required unless --summarize is set")
    report = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    rows = export_blind_review(report, read_jsonl(args.cases), reviewer_id=args.reviewer)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} blind review cases to {output}; all judgments blank.")


if __name__ == "__main__":
    main()
