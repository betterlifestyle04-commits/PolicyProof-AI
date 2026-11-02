"""Re-score an existing baseline using the corrected deterministic evaluator.

No inference or GPU is required. The original JSON report is never modified.
The strict string-matching score remains separate from human semantic review.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from hashlib import sha256 as sha256_bytes
import json
from pathlib import Path
from policyproof.core import read_jsonl, score, aggregate, sha256


def rescore(report: dict, cases: list[dict], dataset_hash: str) -> dict:
    if report.get("engine") != "transformers" or report.get("is_real_model_result") is not True:
        raise ValueError("Only original real Transformers inference reports can be recalculated")
    if report.get("dataset_sha256") != dataset_hash:
        raise ValueError("Dataset checksum differs from original report; refusing to re-score")
    items = report.get("predictions", [])
    lookup = {p["id"]: p["raw"] for p in items}
    expected = {c["id"] for c in cases}
    if len(items) != len(expected) or len(lookup) != len(items) or set(lookup) != expected:
        raise ValueError("Missing, duplicated, or extra example IDs")
    output = dict(report)
    output["original_metrics"] = report.get("metrics")
    output["scores"] = [score(c, lookup[c["id"]]) for c in cases]
    output["metrics"] = aggregate(output["scores"])
    output["scorer_version"] = "conservative-output-audit-v2"
    output["rescored_utc"] = datetime.now(timezone.utc).isoformat()
    output["original_model_outputs_preserved"] = True
    output["warning"] = (
        "Recalculated deterministic metrics from original model responses; "
        "no new inference and no human semantic accuracy estimate. "
        "Original strict score is not a measure of semantic correctness."
    )
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--dataset", default="data/unseen_policy_test.jsonl")
    p.add_argument("--output", help="Defaults to a new -rescored-v2.json file beside the original")
    a = p.parse_args()
    source, dataset = Path(a.report), Path(a.dataset)
    output = Path(a.output) if a.output else source.with_name(source.stem + "-rescored-v2.json")
    if output.resolve() == source.resolve():
        raise SystemExit("Refusing to overwrite the original baseline")
    obj = json.loads(source.read_text(encoding="utf-8"))
    revised = rescore(obj, read_jsonl(dataset), sha256(dataset))
    revised["original_report_sha256"] = sha256_bytes(source.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(revised, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Original unchanged:", source)
    print("Re-scored report:", output)
    print(json.dumps(revised["metrics"], indent=2))


if __name__ == "__main__":
    main()
