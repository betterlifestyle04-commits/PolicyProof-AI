"""Offline, independent pre-inference review of the final 60-case benchmark.

This is a HUMAN annotation workflow, NOT model evaluation. The reviewer pack
includes proposed gold labels and fictional evidence but NO model predictions.
All outputs remain explicitly unsigned until people adjudicate them.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.validate_final_holdout import validate_final_holdout

REVIEW_FIELDS = (
    "evidence_support_yes_no",
    "translation_faithful_yes_no",
    "question_unambiguous_yes_no",
)
IDENTITY_FIELDS = (
    "case_id", "dataset_sha256", "language", "category", "question",
    "provided_evidence", "proposed_gold_answer", "proposed_source_id",
    "should_abstain", "reviewer_code",
)
HEADERS = (*IDENTITY_FIELDS, *REVIEW_FIELDS, "review_notes")
REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$")

MAX_REVIEW_FILE_BYTES = 1_500_000


def _private_new_output(root: Path, name: str, extension: str) -> Path:
    """Constrain CLI writes to a fresh, direct file under gitignored reviews/.

    Never allow a relative traversal, symlink, existing file, public Git path,
    or dataset path. Used only for owner-controlled output destinations; the
    lower-level writer also uses exclusive creation for test/library callers.
    """
    if (not isinstance(name, str) or len(name) > 180
        or not name.startswith("reviews/") or not name.endswith(extension)
        or "\\" in name or ":" in name or chr(0) in name
        or any(ord(ch) < 32 for ch in name)):
        raise ValueError("Review output must be a new private reviews/ file")
    tokens = name.split("/")
    if (len(tokens) != 2 or any(t in ("", ".", "..") for t in tokens)
        or len(PurePosixPath(name).parts) != 2):
        raise ValueError("Unsafe private review output path")
    root = Path(root).resolve()
    folder = root / "reviews"
    if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
        raise ValueError("Private reviews directory is not safe")
    target = folder / tokens[1]
    if target.is_symlink() or target.exists():
        raise FileExistsError("Refusing to overwrite existing human review evidence: " + name)
    return target



def check_reviewer_code(code: str) -> str:
    if not isinstance(code, str) or not REVIEWER_PATTERN.fullmatch(code):
        raise ValueError("Reviewer code must be 2-40 ASCII letters, digits, _ or -")
    return code


def make_review_pack(cases: list[dict], digest: str, reviewer_code: str) -> list[dict]:
    """Produce gold-validation rows without any saved model predictions."""
    check_reviewer_code(reviewer_code)
    if len(cases) != 60 or len({r["id"] for r in cases}) != 60:
        raise ValueError("Expected precisely 60 unique final holdout examples")
    return [
        {
            "case_id": row["id"],
            "dataset_sha256": digest,
            "language": row["language"],
            "category": row["category"],
            "question": row["question"],
            "provided_evidence": row["context"],
            "proposed_gold_answer": row["expected_answer"],
            "proposed_source_id": row["expected_source_id"] or "",
            "should_abstain": str(row["should_abstain"]).lower(),
            "reviewer_code": reviewer_code,
            **{field: "" for field in REVIEW_FIELDS},
            "review_notes": "",
        }
        for row in cases
    ]


def write_review_csv(output: Path, rows: list[dict]) -> None:
    """Create one review packet exclusively; never erase completed human votes."""
    output = Path(output)
    if output.is_symlink() or output.exists():
        raise FileExistsError("Refusing to overwrite existing human review evidence")
    if output.parent.is_symlink():
        raise ValueError("Private review parent is a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate first to avoid leaving a new partial file on malformed rows.
    if not isinstance(rows, list) or any(
        not isinstance(r, dict) or set(r) != set(HEADERS) for r in rows
    ):
        raise ValueError("Invalid review packet fields")
    created = False
    try:
        with output.open("x", encoding="utf-8", newline="") as handle:
            created = True
            writer = csv.DictWriter(handle, fieldnames=HEADERS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
    except BaseException:
        if created:
            output.unlink(missing_ok=True)
        raise


def read_and_validate_review(
    path: Path, cases: list[dict], dataset_digest: str
) -> tuple[str, dict[str, dict]]:
    """Reject changed gold, evidence, IDs, duplicate rows and verdict typos."""
    path = Path(path)
    if (path.is_symlink() or not path.is_file()
        or not 0 < path.stat().st_size <= MAX_REVIEW_FILE_BYTES):
        raise ValueError(f"{path}: reviewer CSV missing, symlinked, empty or oversized")
    csv.field_size_limit(2_000_000)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(HEADERS):
                raise ValueError(f"{path}: unexpected review columns or order")
            records = list(reader)
    except (csv.Error, UnicodeError) as exc:
        raise ValueError(f"{path}: malformed or unreadable review CSV") from exc
    if any(None in record or any(value is None for value in record.values())
           for record in records):
        raise ValueError(f"{path}: extra or missing CSV columns")
    expected = {r["id"]: r for r in cases}
    if len(expected) != 60 or len(records) != 60:
        raise ValueError(f"{path}: require all 60 benchmark cases exactly once")
    codes = {record.get("reviewer_code") for record in records}
    if len(codes) != 1:
        raise ValueError(f"{path}: reviewer code missing or inconsistent")
    reviewer = check_reviewer_code(next(iter(codes)))
    canonical = {
        r["case_id"]: r for r in make_review_pack(cases, dataset_digest, reviewer)
    }
    indexed: dict[str, dict] = {}
    for record in records:
        case_id = record.get("case_id")
        if case_id not in expected or case_id in indexed:
            raise ValueError(f"{path}: duplicate, unknown or missing case ID")
        fixed = canonical[case_id]
        for key in IDENTITY_FIELDS:
            if record.get(key) != fixed[key]:
                raise ValueError(
                    f"{path}: altered source, gold, dataset hash or reviewer metadata "
                    f"for {case_id}: {key}"
                )
        for field in REVIEW_FIELDS:
            verdict = record.get(field)
            if verdict not in ("yes", "no", ""):
                raise ValueError(
                    f"{path}: {case_id}/{field} must be yes, no or blank"
                )
        if record.get("review_notes") is None or len(record["review_notes"]) > 4000:
            raise ValueError(f"{path}: invalid review notes on {case_id}")
        indexed[case_id] = record
    if set(indexed) != set(expected):
        raise ValueError(f"{path}: benchmark case coverage mismatch")
    return reviewer, indexed


def consolidate_reviews(
    paths: list[Path], cases: list[dict], dataset_digest: str
) -> dict:
    """Require independent pseudonyms and disclose all unreviewed/negative cases.

    Distinct pseudonyms alone DO NOT prove distinct humans or language fluency.
    This function never signs off or changes benchmark/gold data.
    """
    if not 2 <= len(paths) <= 4:
        raise ValueError("Require 2-4 separate reviewer CSV files")
    if len({Path(path).resolve() for path in paths}) != len(paths):
        raise ValueError("Reviewer CSV source files must be distinct")
    submitted = [
        (Path(path), *read_and_validate_review(path, cases, dataset_digest))
        for path in paths
    ]
    codes = [code for _, code, _ in submitted]
    if len(set(codes)) != len(codes):
        raise ValueError("Each reviewer must use a distinct pseudonymous code")

    total_slots = len(cases) * len(REVIEW_FIELDS)
    incomplete, disagreements, rejected, unanimous = [], [], [], 0
    for item in cases:
        case_id = item["id"]
        for field in REVIEW_FIELDS:
            votes = {code: indexed[case_id][field] for _, code, indexed in submitted}
            filled = [vote for vote in votes.values() if vote]
            if len(filled) < len(submitted):
                incomplete.append({"case_id": case_id, "field": field})
            elif len(set(filled)) == 1:
                unanimous += 1
            if len(set(filled)) > 1:
                disagreements.append({"case_id": case_id, "field": field})
            if "no" in filled:
                rejected.append({"case_id": case_id, "field": field})

    counts = {
        code: {
            "judgments_completed": sum(
                bool(indexed[r["id"]][field])
                for r in cases for field in REVIEW_FIELDS
            ),
            "judgments_required": total_slots,
            "language_cases": dict(Counter(r["language"] for r in cases)),
        }
        for _, code, indexed in submitted
    }
    clear = not incomplete and not disagreements and not rejected
    status = (
        "READY_FOR_MANUAL_ADJUDICATION_AND_SIGNOFF" if clear
        else "REVIEW_INCOMPLETE" if incomplete
        else "REQUIRES_CORRECTIONS_OR_ADJUDICATION"
    )
    return {
        "status": status,
        "review_complete": False,
        "independent_human_identity_verified": False,
        "competency_verified": False,
        "model_inference_performed": False,
        "dataset_sha256": dataset_digest,
        "case_count": len(cases),
        "reviewer_count": len(submitted),
        "reviewers": codes,
        "review_file_sha256": {
            code: sha256(path) for path, code, _ in submitted
        },
        "coverage": counts,
        "judgment_slots_per_reviewer": total_slots,
        "unanimous_agreement_slots": unanimous,
        "unanimous_agreement_pct": round(100 * unanimous / total_slots, 2),
        "missing_judgments": incomplete,
        "disagreement_slots": disagreements,
        "negative_judgment_slots": rejected,
        "requires_manual_signoff": True,
        "warning": (
            "This is an offline consistency/coverage report, NOT an independent "
            "verification of reviewer identity, expertise, semantic correctness "
            "or benchmark readiness. Adjudicate all negative, incomplete and "
            "disputed labels; version the dataset before inference if altered. "
            "Never auto-populate final_benchmark_review_signoff.json."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export or compare independent blinded-to-model gold reviews"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--export", metavar="REVIEWER_CODE")
    group.add_argument(
        "--summarize", nargs="+", metavar="REVIEW_CSV",
        help="Two or more completed independent reviewer CSVs"
    )
    parser.add_argument("--cases", default=str(ROOT / "data/final_holdout_v1.jsonl"))
    parser.add_argument("--output", help="CSV pack or JSON report path")
    args = parser.parse_args()
    validate_final_holdout()
    cases_path = Path(args.cases)
    if cases_path.resolve() != (ROOT / "data/final_holdout_v1.jsonl").resolve():
        raise ValueError("Refusing to process a noncanonical holdout without a protocol revision")
    cases, digest = read_jsonl(cases_path), sha256(cases_path)
    if args.export:
        # Strict owner-only output guard applies even to user-supplied paths.
        # Review file generation must never replace another review or dataset.
        output = _private_new_output(
            ROOT, args.output or f"reviews/reviewer_{args.export}_final_holdout.csv",
            ".csv",
        )
        write_review_csv(output, make_review_pack(cases, digest, args.export))
        print(f"Created {len(cases)} blank review rows for {args.export} at {output}")
        return
    output = _private_new_output(ROOT, args.output, ".json") if args.output else None
    # Reject public and outside reviews input path conventions in CLI mode.
    for source in args.summarize:
        if (not isinstance(source, str) or not source.startswith("reviews/")
            or len(source.split("/")) != 2
            or source.split("/")[1] in ("", ".", "..")
            or "\\" in source or ":" in source or chr(0) in source):
            raise ValueError("Reviewer inputs must be direct private reviews/*.csv paths")
    # Resolve relative inputs under the canonical repository root, never the
    # process CWD; prevents accidental use of unrelated local files.
    inputs = [ROOT / name for name in args.summarize]
    report = consolidate_reviews(inputs, cases, digest)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
