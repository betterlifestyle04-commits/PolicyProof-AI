"""Validate a candidate final policy benchmark without model inference.

No final-holdout example may be used for prompting, tuning or stopping
criteria before the fixed comparison. Gold labels and translations still
require independent human review.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
import re
from policyproof.core import ROOT, read_jsonl, sha256

FINAL_LANGS = ("en", "no", "ar")


def validate_final_holdout(
    source_file: Path = ROOT / "data/final_holdout_sources_v1.json",
    holdout_file: Path = ROOT / "data/final_holdout_v1.jsonl",
    existing_train: Path = ROOT / "data/train.jsonl",
    prior_dev: Path = ROOT / "data/unseen_policy_test.jsonl",
) -> dict:
    data = json.loads(Path(source_file).read_text(encoding="utf-8"))
    policies = data["policies"]
    ids = {p["id"] for p in policies}
    if len(policies) != 8 or len(ids) != 8 or any(
        not re.fullmatch(r"POL-F1\d{2}", x) for x in ids
    ):
        raise ValueError("Expected eight independent, unique final source IDs")
    rows = read_jsonl(holdout_file)
    if len(rows) != 60 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected 60 unique final evaluation cases")
    lang_counts = Counter(r["language"] for r in rows)
    if lang_counts != {"en": 20, "no": 20, "ar": 20}:
        raise ValueError("Final language coverage differs from 20 per language")
    cat_counts = Counter(r["category"] for r in rows)
    if cat_counts != {"final_supported": 48, "final_unknown": 9, "final_injection": 3}:
        raise ValueError("Final benchmark categories unexpectedly changed")
    all_ids_in_final = set()
    for item in rows:
        if item.get("split") != "final_candidate_holdout_v1":
            raise ValueError("A final row has wrong split")
        if item.get("human_reviewed") is not False:
            raise ValueError("Human-reviewed label not supported without adjudication")
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError("Empty question found")
        if item.get("expected_source_id") is not None and item["expected_source_id"] not in ids:
            raise ValueError("Expected reference cites outside final-policy corpus")
        found = set(re.findall(r"\[(POL-[A-Za-z0-9-]+)\]", item["context"]))
        if found != ids:
            raise ValueError("Final context is incomplete, or includes other source IDs")
        if item["should_abstain"]:
            if (item["expected_source_id"] is not None
                    or item["expected_answer"] != "INSUFFICIENT_EVIDENCE"):
                raise ValueError("Abstention gold label mismatch")
        elif not item["expected_answer"].strip():
            raise ValueError("Missing supported answer")
        all_ids_in_final |= found

    for other_file in (existing_train, prior_dev):
        if Path(other_file).exists():
            existing = read_jsonl(other_file)
            used = set()
            for item in existing:
                used.update(re.findall(r"\[(POL-[A-Za-z0-9-]+)\]", item["context"]))
                if item.get("expected_source_id"):
                    used.add(item["expected_source_id"])
            if all_ids_in_final & used:
                raise ValueError(f"Policy source ID leaked from {other_file}")
    return {
        "status": "candidate_frozen_for_review",
        "cases": len(rows),
        "languages": dict(lang_counts),
        "categories": dict(cat_counts),
        "new_source_policies": sorted(ids),
        "sha256": sha256(holdout_file),
        "translations_human_reviewed": False,
        "final_results_measured": False,
        "caveat": "Public fixture with draft labels; independent bilingual review and no repeated tuning required.",
    }


def main():
    result = validate_final_holdout()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
