"""Fail-closed, stdlib-only audit of PolicyProof's versioned research fixtures.

Checks dataset integrity, policy-document separation, labels, and provenance.
This is not an evaluation of any AI model and does not establish safety.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

POLICY_RE = re.compile(r"\[(POL-[A-Za-z0-9-]+)\]")
LANGUAGES = {"en", "no", "ar"}
SPLITS = {"train": "train.jsonl", "paraphrase": "test.jsonl", "unseen_policy": "unseen_policy_test.jsonl"}


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"Missing dataset: {path.name}")
    try:
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid JSONL: {path.name}") from exc
    if not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"Empty or malformed dataset: {path.name}")
    return rows


def audit(root: str | Path) -> dict:
    """Return auditable diagnostics; raise on leakage, corruption or stale manifest."""
    root = Path(root)
    rows = {split: read_rows(root / name) for split, name in SPLITS.items()}
    all_ids: set[str] = set()
    source_ids: dict[str, set[str]] = {}
    problems: list[str] = []
    warnings: list[str] = []
    languages: dict[str, dict[str, int]] = {}
    exact_questions: dict[str, set[tuple[str, str]]] = {}
    for split, examples in rows.items():
        source_ids[split] = set()
        exact_questions[split] = set()
        languages[split] = dict(Counter(r.get("language", "") for r in examples))
        if set(languages[split]) != LANGUAGES:
            problems.append(f"{split}: missing or unexpected language labels")
        for index, r in enumerate(examples):
            key = f"{split}[{index}]"
            identifier = r.get("id")
            if not isinstance(identifier, str) or not identifier:
                problems.append(f"{key}: no stable ID")
            elif identifier in all_ids:
                problems.append(f"{key}: duplicate ID {identifier}")
            all_ids.add(identifier)
            lang = r.get("language")
            q = r.get("question")
            context = r.get("context")
            if not isinstance(q, str) or not q.strip():
                problems.append(f"{key}: empty question")
                continue
            if not isinstance(context, str) or not context.strip():
                problems.append(f"{key}: missing context")
                continue
            pair = (lang, q.strip().casefold())
            if pair in exact_questions[split] and r.get("category") != "context_injection":
                warnings.append(f"{key}: repeated question within split")
            exact_questions[split].add(pair)
            citations = POLICY_RE.findall(context)
            if len(citations) != len(set(citations)):
                problems.append(f"{key}: duplicate policy IDs in context")
            if not citations:
                problems.append(f"{key}: no source policies in context")
            source_ids[split].update(citations)
            expected_source_id = r.get("expected_source_id")
            abstain = r.get("should_abstain")
            expected_answer = r.get("expected_answer")
            if type(abstain) is not bool or not isinstance(expected_answer, str) or not expected_answer.strip():
                problems.append(f"{key}: invalid expected answer or abstention label")
            elif abstain:
                if expected_source_id is not None or expected_answer != "INSUFFICIENT_EVIDENCE":
                    problems.append(f"{key}: invalid abstention target")
            elif expected_source_id not in set(citations):
                problems.append(f"{key}: gold citation missing from provided evidence")
        if len(examples) < 12:
            warnings.append(f"{split}: unusually small fixture set")
    intersection = source_ids["train"] & source_ids["unseen_policy"]
    if intersection:
        problems.append(f"train-to-unseen policy leakage: {sorted(intersection)}")
    train_test = exact_questions["train"] & exact_questions["paraphrase"]
    if train_test:
        problems.append(f"{len(train_test)} exact repeated train/test questions")
    for split, count in (("train", 78), ("paraphrase", 42), ("unseen_policy", 48)):
        if len(rows[split]) != count:
            warnings.append(f"{split}: {len(rows[split])} rows, prototype baseline {count}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        problems.append("manifest.json missing")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksums = manifest.get("sha256", manifest.get("files_sha256", {}))
        if not isinstance(checksums, dict):
            problems.append("manifest lacks checksum mapping")
            checksums = {}
        for filename in SPLITS.values():
            expected = checksums.get(filename)
            actual = sha256((root / filename).read_bytes()).hexdigest()
            if not expected or actual != expected:
                problems.append(f"stale/missing SHA256 for {filename}")
    report = {
        "result": "fail" if problems else "pass",
        "purpose": "dataset and provenance integrity only; not model quality",
        "counts": {split: len(v) for split, v in rows.items()},
        "languages": languages,
        "source_policy_counts": {split: len(ids) for split, ids in source_ids.items()},
        "train_unseen_policy_overlap": len(intersection),
        "problems": problems,
        "warnings": warnings,
        "limits": [
            "Template-generated questions are not an independent human benchmark",
            "Six unseen fictional policies do not establish real-world safety",
            "English-only evidence means some target answers test translation too",
        ],
    }
    if problems:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description="Audit synthetic benchmark isolation and provenance")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    report = audit(args.data_dir)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
