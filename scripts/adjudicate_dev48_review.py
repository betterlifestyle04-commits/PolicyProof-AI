"""Private CPU-only masked human adjudication of disputed dev48 reviewer votes.

This tool never invents human votes. Export gives a genuinely independent
adjudicator a NEUTRAL subset of the v2 frozen review packet, without exposing
original reviewer votes, model A/B mapping, gold answers or UNKNOWN/INJECT
categories. A filled third-party CSV is separately validated before the
original three-source judgments can be combined.

The original two reviewers' files remain byte-for-byte frozen; the dispute
adjudicator decides ONLY fields where those reviewers disagreed. All other
fields preserve their unanimous original judgment. Descriptive results are
not final-blind model quality, independent reviewer attestation or release
approval.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from policyproof.core import ROOT, sha256
from scripts.paired_semantic_review import (
    FIELDS, IDENTITY, REVIEWER, SCHEMA, VERDICTS, _pair_metrics,
    _private_path, _read_csv, _read_json, summarize_reviews,
)

ADJ_SCHEMA = "policyproof_dev48_masked_adjudication_v1"
DEFAULT_PACKET = "reviews/dev48_blank.csv"
DEFAULT_KEY = "reviews/dev48_private_mapping.json"
DEFAULT_DISPUTES = "reviews/dev48_disputes_blank.csv"
DEFAULT_MANIFEST = "reviews/dev48_disputes_private_manifest.json"
DEFAULT_OUTPUT = "reviews/dev48_adjudicated_owner_only.json"


def _inputs(root: Path, packet: str, key: str, reviewer_files: list[str]):
    """Validate original v2 packet, sealed key and original FULL human votes."""
    if len(set(reviewer_files)) != len(reviewer_files):
        raise ValueError("Original reviewer filenames must be distinct")
    result = summarize_reviews(root, packet, key, reviewer_files)
    source = _private_path(root, packet, ".csv", new=False)
    mapping = _private_path(root, key, ".json", new=False)
    original = {
        name: _private_path(root, name, ".csv", new=False)
        for name in reviewer_files
    }
    if len({source, mapping, *original.values()}) != len(original) + 2:
        raise ValueError("Review source, mapping and reviewer paths must be distinct")
    return result, source, mapping, original


def _dispute_keys(summary: dict) -> list[tuple[str, str]]:
    keys = [(x["case_id"], x["slot"]) for x in summary["disputed"]]
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError("No disputed human judgments to independently adjudicate")
    return sorted(keys)


def export_disputes(root: Path, *, reviewer_files: list[str],
                    packet: str = DEFAULT_PACKET,
                    key: str = DEFAULT_KEY,
                    blank: str = DEFAULT_DISPUTES,
                    manifest: str = DEFAULT_MANIFEST) -> dict:
    """Export exactly disputed rows; NEVER export the first reviewers' votes."""
    root = Path(root).resolve()
    review, source, mapping, originals = _inputs(root, packet, key, reviewer_files)
    disputed = _dispute_keys(review)
    blank_file = _private_path(root, blank, ".csv", new=True)
    manifest_file = _private_path(root, manifest, ".json", new=True)
    if blank_file in {source, mapping, *originals.values()} or (
        manifest_file in {source, mapping, *originals.values(), blank_file}
    ):
        raise ValueError("Adjudication output may not reuse review evidence paths")
    original_packet = _read_csv(source)
    chosen = [r for r in original_packet
              if (r["case_id"], r["response_slot"]) in set(disputed)]
    if len(chosen) != len(disputed):
        raise ValueError("Dispute evidence does not match original blank packet")

    receipt = {
        "schema": ADJ_SCHEMA,
        "original_review_schema": SCHEMA,
        "source_packet_name": packet,
        "source_packet_sha256": sha256(source),
        "private_mapping_name": key,
        "private_mapping_sha256": sha256(mapping),
        "first_reviewers_sha256": {
            name: sha256(originals[name]) for name in reviewer_files
        },
        "disputed_response_ids": [
            {"case_id": case, "slot": slot} for case, slot in disputed
        ],
        "adjudication_packet_name": blank,
        "response_rows": len(disputed),
        "original_votes_in_packet": False,
        "original_case_ids_in_packet": False,
        "answerability_categories_in_packet": False,
        "model_origin_in_packet": False,
        "human_adjudication_completed": False,
        "reviewer_competence_or_independence_verified": False,
        "not_final_blind_benchmark": True,
    }
    created: list[Path] = []
    try:
        blank_file.parent.mkdir(parents=True, exist_ok=True)
        with blank_file.open("x", encoding="utf-8", newline="") as handle:
            created.append(blank_file)
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(chosen)
        receipt["adjudication_packet_sha256"] = sha256(blank_file)
        with manifest_file.open("x", encoding="utf-8") as handle:
            created.append(manifest_file)
            json.dump(receipt, handle, ensure_ascii=False, indent=2,
                      allow_nan=False)
            handle.write("\n")
    except BaseException:
        for target in reversed(created):
            target.unlink(missing_ok=True)
        raise
    return {
        "status": "BLIND_DISPUTED_ROWS_PACKET_READY_NO_ADJUDICATOR_VOTES",
        "masked_dispute_packet": str(blank_file),
        "private_manifest_DO_NOT_SHARE": str(manifest_file),
        "rows_needing_independent_adjudication": len(disputed),
        "original_votes_shared_with_adjudicator": False,
        "model_identities_shared_with_adjudicator": False,
        "human_adjudication_completed": False,
    }


def _review_rows(path: Path, expected: dict[tuple[str, str], tuple],
                 other_codes: set[str]) -> tuple[str, dict]:
    """Validate a genuinely FILLED, dispute-only third-reviewer CSV."""
    if (path.is_symlink() or not path.is_file()
        or not 0 < path.stat().st_size <= 4_000_000):
        raise ValueError("Missing, symlinked or oversized adjudication CSV")
    csv.field_size_limit(2_000_000)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Adjudication CSV columns differ from frozen template")
        rows = list(reader)
    if len(rows) != len(expected) or any(None in row for row in rows):
        raise ValueError("Adjudicator must complete every disputed response row")
    codes = {r["reviewer_code"] for r in rows}
    if len(codes) != 1 or not REVIEWER.fullmatch(next(iter(codes))):
        raise ValueError("Adjudicator must use one valid reviewer code")
    code = next(iter(codes))
    if code in other_codes:
        raise ValueError("Adjudicator code must differ from original reviewer codes")
    votes: dict[tuple[str, str], dict[str, int]] = {}
    for row in rows:
        identity = (row["case_id"], row["response_slot"])
        if (identity not in expected or identity in votes
            or tuple(row[k] for k in IDENTITY) != expected[identity]):
            raise ValueError("Adjudication changed, duplicated or omitted blinded evidence")
        if any(row[field] not in ("0", "1") for field in VERDICTS):
            raise ValueError("All adjudicator judgments must be filled with 0 or 1")
        votes[identity] = {field: int(row[field]) for field in VERDICTS}
    if set(votes) != set(expected):
        raise ValueError("Adjudicator missed a disputed response")
    return code, votes


def resolve_disputes(root: Path, *, reviewer_files: list[str],
                     adjudicator: str,
                     packet: str = DEFAULT_PACKET,
                     key: str = DEFAULT_KEY,
                     blank: str = DEFAULT_DISPUTES,
                     manifest: str = DEFAULT_MANIFEST) -> dict:
    """Combine authentic independent votes, applying arbiter only on conflicts."""
    root = Path(root).resolve()
    original, source, mapping, originals = _inputs(root, packet, key, reviewer_files)
    disputed = _dispute_keys(original)
    blank_file = _private_path(root, blank, ".csv", new=False)
    receipt_file = _private_path(root, manifest, ".json", new=False)
    filled = _private_path(root, adjudicator, ".csv", new=False)
    if len({source, mapping, *originals.values(), blank_file, receipt_file, filled}) != (
        len(originals) + 5
    ):
        raise ValueError("Adjudication evidence paths must be distinct")

    receipt = _read_json(receipt_file)
    expected_receipt_fields = {
        "schema", "original_review_schema",
        "source_packet_name", "source_packet_sha256",
        "private_mapping_name", "private_mapping_sha256",
        "first_reviewers_sha256", "disputed_response_ids",
        "adjudication_packet_name", "adjudication_packet_sha256",
        "response_rows", "original_votes_in_packet",
        "original_case_ids_in_packet", "answerability_categories_in_packet",
        "model_origin_in_packet", "human_adjudication_completed",
        "reviewer_competence_or_independence_verified", "not_final_blind_benchmark",
    }
    if (set(receipt) != expected_receipt_fields
        or receipt["schema"] != ADJ_SCHEMA
        or receipt["original_review_schema"] != SCHEMA
        or receipt["source_packet_name"] != packet
        or receipt["source_packet_sha256"] != sha256(source)
        or receipt["private_mapping_name"] != key
        or receipt["private_mapping_sha256"] != sha256(mapping)
        or receipt["adjudication_packet_name"] != blank
        or receipt["adjudication_packet_sha256"] != sha256(blank_file)
        or receipt["first_reviewers_sha256"] != {
            name: sha256(originals[name]) for name in reviewer_files
        }
        or receipt["disputed_response_ids"] != [
            {"case_id": c, "slot": s} for c, s in disputed
        ]
        or receipt["response_rows"] != len(disputed)
        or any(receipt[key] is not False for key in (
            "original_votes_in_packet", "original_case_ids_in_packet",
            "answerability_categories_in_packet", "model_origin_in_packet",
            "human_adjudication_completed",
            "reviewer_competence_or_independence_verified"
        ))
        or receipt["not_final_blind_benchmark"] is not True):
        raise ValueError("Adjudication receipt does not match sealed original review evidence")

    packet_rows = _read_csv(source)
    original_expected = {
        (row["case_id"], row["response_slot"]):
        tuple(row[k] for k in IDENTITY)
        for row in packet_rows
    }
    dispute_expected = {k: original_expected[k] for k in disputed}
    # Confirm the exported disputed rows were never given a vote, swapped with
    # outside evidence or edited before the adjudicator received the packet.
    with blank_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Frozen adjudication CSV header changed")
        blanks = list(reader)
    if (len(blanks) != len(disputed)
        or any(None in row for row in blanks)
        or {
            (row["case_id"], row["response_slot"]) for row in blanks
        } != set(disputed)):
        raise ValueError("Adjudication packet omits or adds disputed responses")
    for row in blanks:
        identity = (row["case_id"], row["response_slot"])
        if (tuple(row[k] for k in IDENTITY) != dispute_expected[identity]
            or row["reviewer_code"]
            or any(row[f] for f in (*VERDICTS, "reviewer_notes"))):
            raise ValueError("Adjudication blank packet leaked votes or changed source evidence")

    original_rows = [
        _read_csv(originals[name]) for name in reviewer_files
    ]
    reviewer_codes = {r[0]["reviewer_code"] for r in original_rows}
    adjudicator_code, arbiter_votes = _review_rows(
        filled, dispute_expected, reviewer_codes
    )
    # The original 2-4 reviews have all been independently validated and
    # pinned by summarize_reviews and the manifest hashes above. Preserve all
    # their unanimous field judgments. Override ONLY disputed field judgments
    # with the genuinely filled third-party adjudicator's separate vote.
    reviewer_vote_maps = [
        {
            (row["case_id"], row["response_slot"]):
            {field: int(row[field]) for field in VERDICTS}
            for row in rows
        } for rows in original_rows
    ]
    key_data = _read_json(mapping)
    final_by_case: dict[str, dict] = {}
    adjudicated_verdict_cells = 0
    for case_id, slot in sorted(original_expected):
        identity = (case_id, slot)
        values = {}
        for field in VERDICTS:
            original_field_votes = [
                reviewer[identity][field] for reviewer in reviewer_vote_maps
            ]
            if len(set(original_field_votes)) == 1:
                values[field] = original_field_votes[0]
            else:
                if identity not in arbiter_votes:
                    raise ValueError("Disputed human verdict lacks adjudicator")
                values[field] = arbiter_votes[identity][field]
                adjudicated_verdict_cells += 1
        model_identity = key_data["assignments"][case_id][slot]
        final_by_case.setdefault(case_id, {})[model_identity] = values
    if (len(final_by_case) != 48
        or any(set(pair) != {"base", "tuned"} for pair in final_by_case.values())):
        raise ValueError("Adjudicated paired review missing model-identity coverage")
    metadata = key_data["case_metadata"]
    cases = sorted(final_by_case)
    def stats(keys: list[str]) -> dict:
        return {
            field: _pair_metrics(final_by_case, keys, field)
            for field in VERDICTS
        }
    metrics = {
        "all_cases": stats(cases),
        "answerable_including_injection": stats([
            cid for cid in cases if metadata[cid]["category"] != "unseen_unanswerable"
        ]),
        "unsupported_only": stats([
            cid for cid in cases if metadata[cid]["category"] == "unseen_unanswerable"
        ]),
        "by_language": {
            lang: stats([cid for cid in cases
                         if metadata[cid]["language"] == lang])
            for lang in ("en", "no", "ar")
        },
    }
    # Recheck every original evidence digest after calculation. This is
    # source stability checking, NOT cryptographic proof of reviewer identity.
    if (sha256(source) != receipt["source_packet_sha256"]
        or sha256(mapping) != receipt["private_mapping_sha256"]
        or sha256(blank_file) != receipt["adjudication_packet_sha256"]
        or any(sha256(originals[name]) != h
               for name, h in receipt["first_reviewers_sha256"].items())):
        raise ValueError("Human review evidence changed during adjudication")
    return {
        "status": "ADJUDICATED_DEVELOPMENT_REVIEW_REQUIRES_INDEPENDENCE_ATTESTATION",
        "schema": ADJ_SCHEMA,
        "development_only": True,
        "final_blind_benchmark": False,
        "paired_development_cases": 48,
        "response_rows": 96,
        "original_reviewers_count": len(reviewer_files),
        "original_reviewer_codes": sorted(reviewer_codes),
        "independent_adjudicator_code_reported": adjudicator_code,
        "adjudicator_identity_or_independence_verified": False,
        "original_review_files_sha256": receipt["first_reviewers_sha256"],
        "adjudicator_filled_csv_sha256": sha256(filled),
        "private_original_mapping_sha256": receipt["private_mapping_sha256"],
        "masked_packet_sha256": receipt["source_packet_sha256"],
        "disputed_response_rows": len(disputed),
        "adjudicated_verdict_cells": adjudicated_verdict_cells,
        "original_unanimous_verdicts_preserved": True,
        "human_approval_inferred": False,
        "semantically_validated_model_improvement": False,
        "metrics": metrics,
        "limitations": [
            "Owner-reported third reviewer code does not prove human identity, qualifications or independence",
            "A third reviewer adjudicates only contested field values; all original reviewer votes stay frozen",
            "No original reviewer codes, votes or gold categories appear in the blank adjudicator packet",
            "Small previously inspected synthetic development data; NOT final blind model evaluation",
            "Reconciled judgments are a human review result, not an automatic ground-truth guarantee",
            "No inference, model weight update, external benchmark or submission confirmation was performed",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    actions = parser.add_subparsers(dest="action", required=True)
    export = actions.add_parser("export", help="Create masked dispute-only packet for another human")
    resolve = actions.add_parser("resolve", help="Validate completed third human review and report")
    for command in (export, resolve):
        command.add_argument("--packet", default=DEFAULT_PACKET)
        command.add_argument("--key", default=DEFAULT_KEY)
        command.add_argument("--reviews", nargs="+", required=True)
        command.add_argument("--blank", default=DEFAULT_DISPUTES)
        command.add_argument("--manifest", default=DEFAULT_MANIFEST)
    resolve.add_argument("--adjudicator", required=True)
    resolve.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.action == "export":
        result = export_disputes(
            args.root, reviewer_files=args.reviews,
            packet=args.packet, key=args.key,
            blank=args.blank, manifest=args.manifest,
        )
    else:
        output = _private_path(Path(args.root).resolve(), args.output, ".json", new=True)
        result = resolve_disputes(
            args.root, reviewer_files=args.reviews, adjudicator=args.adjudicator,
            packet=args.packet, key=args.key,
            blank=args.blank, manifest=args.manifest,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2,
                      allow_nan=False)
            handle.write("\n")
        result = {
            "status": result["status"],
            "private_report": str(output),
            "reconciled_disputes": result["disputed_response_rows"],
            "human_independence_verified": False,
            "semantically_validated_model_improvement": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
