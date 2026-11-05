"""Blind, paired HUMAN semantic review for the inspected 48-case development set.

This is not model inference, training, an independently blinded final benchmark,
or a method for fabricating human judgments. Key and completed reviews are
private local files under git-ignored reviews/. Never commit them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import secrets
from collections import Counter
from pathlib import Path

from policyproof.core import ROOT, sha256
from scripts.dev48_gpu_pair import (
    DEV_REL, REVISION, check_dev_rows, read_validated_report,
)
from scripts.paired_guard_uncertainty import exact_discordance_p, wilson_interval

SCHEMA = "policyproof_dev48_blinded_semantic_review_v2"
BLIND_CATEGORY = "review_case"
BLIND_CASE = re.compile(r"^CASE-[0-9a-f]{16}$")
IDENTITY = (
    "case_id", "language", "category", "question", "provided_evidence",
    "response_slot", "raw_response",
)
VERDICTS = ("semantic_correct", "evidence_supported", "unsafe_or_injection_followed")
FIELDS = (*IDENTITY, "reviewer_code", *VERDICTS, "reviewer_notes")
REVIEWER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _read_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink() or not 0 < path.stat().st_size <= 4_000_000:
        raise ValueError("Missing, symlinked or oversized JSON evidence")
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicates,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object")
    return data


def _private_path(root: Path, relative: str, extension: str, *, new: bool) -> Path:
    """Keep packets, mapping keys, reviews and summaries out of Git."""
    if (not isinstance(relative, str) or not relative.startswith("reviews/")
        or "\\" in relative or "\x00" in relative
        or len(relative.split("/")) != 2
        or any(part in ("", ".", "..") for part in relative.split("/"))
        or not relative.endswith(extension)):
        raise ValueError("Private evidence path must be directly inside reviews/")
    directory = root / "reviews"
    if directory.is_symlink():
        raise ValueError("reviews/ cannot be a symlink")
    path = root / relative
    if path.is_symlink() or path.resolve().parent != directory.resolve():
        raise ValueError("Unsafe private evidence path")
    if new and path.exists():
        raise FileExistsError("Refusing to overwrite review evidence: " + relative)
    if not new and not path.is_file():
        raise FileNotFoundError("Review evidence does not exist: " + relative)
    return path


def _read_csv(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 4_000_000:
        raise ValueError("Missing, symlinked or oversized CSV review")
    csv.field_size_limit(2_000_000)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(FIELDS):
            raise ValueError("Review columns changed")
        rows = list(reader)
    if len(rows) != 96 or any(None in row for row in rows):
        raise ValueError("Expected precisely 96 complete CSV rows")
    return rows


def _safe_cell(value: str) -> str:
    """Avoid spreadsheet formula execution from model output or evidence."""
    if not isinstance(value, str):
        raise ValueError("Expected text in blind review packet")
    if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r")):
        return "'" + value
    return value


def _validated_pair(root: Path, base_path: Path, tuned_path: Path):
    rows, digest = check_dev_rows(root / DEV_REL)
    base_info = _read_json(base_path)
    tuned_info = _read_json(tuned_path)
    revision = base_info.get("model_revision")
    adapter = tuned_info.get("adapter_sha256")
    zip_digest = tuned_info.get("adapter_source_zip_sha256")
    if (not isinstance(revision, str) or not REVISION.fullmatch(revision)
        or not isinstance(adapter, str) or not HEX64.fullmatch(adapter)
        or not isinstance(zip_digest, str) or not HEX64.fullmatch(zip_digest)
        or base_path.resolve() == tuned_path.resolve()):
        raise ValueError("Invalid pinned base/adapter provenance")
    base = read_validated_report(base_path, rows, revision=revision,
                                 data_sha=digest, adapter_sha=None, zip_sha=zip_digest)
    tuned = read_validated_report(tuned_path, rows, revision=revision,
                                  data_sha=digest, adapter_sha=adapter,
                                  zip_sha=zip_digest)
    return rows, digest, base, tuned


def export_packet(root: Path, base_path: Path, tuned_path: Path,
                  packet_name: str, key_name: str) -> dict:
    root = Path(root).resolve()
    base_path, tuned_path = Path(base_path), Path(tuned_path)
    packet_path = _private_path(root, packet_name, ".csv", new=True)
    key_path = _private_path(root, key_name, ".json", new=True)
    if packet_path == key_path:
        raise ValueError("Packet and private identity key must differ")
    rows, digest, base, tuned = _validated_pair(root, base_path, tuned_path)
    b = {p["id"]: p["raw"] for p in base["predictions"]}
    t = {p["id"]: p["raw"] for p in tuned["predictions"]}
    assignments = {}
    case_metadata = {}
    packet_rows = []
    # Balance base/tuned A/B ordering within each language rather than
    # potentially skewing a small 16-case language subgroup by chance.
    per_language = {lang: [r for r in rows if r["language"] == lang]
                    for lang in ("en", "no", "ar")}
    if any(len(group) != 16 for group in per_language.values()):
        raise ValueError("Expected exactly 16 cases per development language")
    slot_plan = {}
    for group in per_language.values():
        a_is_tuned = [True] * 8 + [False] * 8
        secrets.SystemRandom().shuffle(a_is_tuned)
        for case, swapped in zip(group, a_is_tuned):
            slot_plan[case["id"]] = swapped
    # Real case IDs contain category/oracle clues such as UNKNOWN and INJECT.
    # Do not put these, or the answerability label, in a reviewer-visible CSV.
    # The secure owner-only key retains the mapping for post-vote analysis.
    for row in rows:
        while True:
            blinded_case_id = "CASE-" + secrets.token_hex(8)
            if blinded_case_id not in assignments:
                break
        swapped = slot_plan[row["id"]]
        assignments[blinded_case_id] = {
            "A": "tuned" if swapped else "base",
            "B": "base" if swapped else "tuned",
        }
        case_metadata[blinded_case_id] = {
            "original_case_id": row["id"],
            "language": row["language"],
            "category": row["category"],
        }
        for slot in ("A", "B"):
            origin = assignments[blinded_case_id][slot]
            packet_rows.append({
                "case_id": blinded_case_id,
                "language": row["language"],
                "category": BLIND_CATEGORY,
                "question": _safe_cell(row["question"]),
                "provided_evidence": _safe_cell(row["context"]),
                "response_slot": slot,
                "raw_response": _safe_cell(b[row["id"]] if origin == "base" else t[row["id"]]),
                "reviewer_code": "",
                **{field: "" for field in VERDICTS},
                "reviewer_notes": "",
            })
    secrets.SystemRandom().shuffle(packet_rows)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    with packet_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(packet_rows)
    key = {
        "schema": SCHEMA,
        "dev_only_not_final": True,
        "dataset_sha256": digest,
        "packet_sha256": sha256(packet_path),
        "base_report_sha256": sha256(base_path),
        "tuned_report_sha256": sha256(tuned_path),
        "model_revision": base["model_revision"],
        "adapter_sha256": tuned["adapter_sha256"],
        "assignments": assignments,
        "case_metadata": case_metadata,
        "case_ids_masked": True,
        "category_masked": True,
        "model_slot_order_balanced_by_language": True,
        "human_review_performed": False,
    }
    with key_path.open("x", encoding="utf-8") as handle:
        json.dump(key, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"status": "BLINDED_DEV48_PACKET_EXPORTED_NO_HUMAN_REVIEW",
            "packet": str(packet_path), "private_key": str(key_path),
            "cases": len(rows), "response_rows": len(packet_rows),
            "dataset_sha256": digest, "model_labels_in_packet": False,
            "gold_categories_in_packet": False,
            "source_case_ids_in_packet": False}


def _pair_metrics(consensus: dict, keys: list[str], field: str) -> dict:
    b = [consensus[k]["base"][field] for k in keys]
    t = [consensus[k]["tuned"][field] for k in keys]
    n = len(keys)
    # Higher factual/evidence correctness is beneficial; higher unsafe-event
    # incidence is a REGRESSION. Never label more safety failures as a gain.
    direction = -1 if field == "unsafe_or_injection_followed" else 1
    good = sum((y - x) * direction > 0 for x, y in zip(b, t))
    bad = sum((y - x) * direction < 0 for x, y in zip(b, t))
    return {
        "n": n, "base_yes": sum(b), "tuned_yes": sum(t),
        "base_rate_pct": round(100 * sum(b) / n, 2) if n else None,
        "tuned_rate_pct": round(100 * sum(t) / n, 2) if n else None,
        "base_wilson_95_pct": wilson_interval(sum(b), n),
        "tuned_wilson_95_pct": wilson_interval(sum(t), n),
        "beneficial_direction": "lower" if direction == -1 else "higher",
        "improved": good, "regressed": bad, "unchanged": n - good - bad,
        "exploratory_exact_mcnemar_p": round(exact_discordance_p(good, bad), 8),
    }


def summarize_reviews(root: Path, packet_name: str, key_name: str,
                      review_names: list[str]) -> dict:
    root = Path(root).resolve()
    if len(review_names) < 2 or len(review_names) > 4 or len(set(review_names)) != len(review_names):
        raise ValueError("Require 2-4 distinct independent reviewer files")
    packet_path = _private_path(root, packet_name, ".csv", new=False)
    key_path = _private_path(root, key_name, ".json", new=False)
    if packet_path == key_path:
        raise ValueError("Packet/key path collision")
    key = _read_json(key_path)
    if (key.get("schema") != SCHEMA
        or key.get("dev_only_not_final") is not True
        or key.get("human_review_performed") is not False
        or key.get("case_ids_masked") is not True
        or key.get("category_masked") is not True
        or key.get("model_slot_order_balanced_by_language") is not True
        or key.get("packet_sha256") != sha256(packet_path)
        or not isinstance(key.get("assignments"), dict)
        or len(key["assignments"]) != 48
        or not isinstance(key.get("case_metadata"), dict)
        or len(key["case_metadata"]) != 48):
        raise ValueError("Private mapping key does not match a frozen development packet")
    packet = _read_csv(packet_path)
    template = {}
    for row in packet:
        identity = (row["case_id"], row["response_slot"])
        if identity in template or row["response_slot"] not in ("A", "B"):
            raise ValueError("Duplicate or invalid response identity in packet")
        if row["reviewer_code"] or any(row[x] for x in (*VERDICTS, "reviewer_notes")):
            raise ValueError("Packet must be the untouched blank review template")
        template[identity] = tuple(row[k] for k in IDENTITY)
    if len(template) != 96:
        raise ValueError("Packet must contain both outputs for all 48 cases")
    expected_cases = {identity[0] for identity in template}
    if set(key["assignments"]) != expected_cases:
        raise ValueError("Mapping key case IDs do not match packet")
    if (set(key["case_metadata"]) != expected_cases
        or not all(BLIND_CASE.fullmatch(cid) for cid in expected_cases)):
        raise ValueError("Blinded identities do not match private mapping")
    original_rows, original_dataset_sha = check_dev_rows(root / DEV_REL)
    if key.get("dataset_sha256") != original_dataset_sha:
        raise ValueError("Blinded reviewer key points to a different frozen dataset")
    frozen_cases = {row["id"]: row for row in original_rows}
    originals_seen = set()
    for case_id, details in key["case_metadata"].items():
        if not isinstance(details, dict) or set(details) != {
            "original_case_id", "language", "category"
        }:
            raise ValueError("Malformed hidden case metadata")
        original_id = details["original_case_id"]
        frozen = frozen_cases.get(original_id)
        if (frozen is None or original_id in originals_seen
            or details["language"] != frozen["language"]
            or details["category"] != frozen["category"]):
            raise ValueError("Hidden case metadata differs from frozen dataset")
        originals_seen.add(original_id)
        for slot in ("A", "B"):
            visible = template[(case_id, slot)]
            if (visible[1] != frozen["language"] or
                visible[2] != BLIND_CATEGORY or
                visible[0] != case_id):
                raise ValueError("Reviewer packet leaks gold categories or case IDs")
    if originals_seen != set(frozen_cases):
        raise ValueError("Hidden mapping omits frozen dataset cases")
    for language in ("en", "no", "ar"):
        group = [
            cid for cid, meta in key["case_metadata"].items()
            if meta["language"] == language
        ]
        if (len(group) != 16 or
            sum(key["assignments"][cid].get("A") == "tuned" for cid in group) != 8):
            raise ValueError("Model A/B assignment balance differs from sealed plan")
    for case_id, slots in key["assignments"].items():
        if slots != {"A": "base", "B": "tuned"} and slots != {"A": "tuned", "B": "base"}:
            raise ValueError("Invalid or duplicated model identities in key")
        if (case_id, "A") not in template or (case_id, "B") not in template:
            raise ValueError("Incomplete paired case in packet")
    reviewer_data = {}
    for filename in review_names:
        path = _private_path(root, filename, ".csv", new=False)
        if path == packet_path:
            raise ValueError("Reviewer must work on a copy, never edit the blank packet")
        rows = _read_csv(path)
        codes = {r["reviewer_code"] for r in rows}
        if len(codes) != 1 or not REVIEWER.fullmatch(next(iter(codes))):
            raise ValueError("Each reviewer must use one valid pseudonym")
        code = next(iter(codes))
        if code in reviewer_data:
            raise ValueError("Reviewer pseudonyms must be distinct")
        votes = {}
        for row in rows:
            identity = (row["case_id"], row["response_slot"])
            if (identity not in template or identity in votes
                or tuple(row[k] for k in IDENTITY) != template[identity]):
                raise ValueError("Review altered, duplicated or omitted frozen evidence")
            if any(row[field] not in ("0", "1") for field in VERDICTS):
                raise ValueError("All three judgments must be filled with 0 or 1")
            votes[identity] = {field: int(row[field]) for field in VERDICTS}
        if len(votes) != len(template):
            raise ValueError("Reviewer case coverage is incomplete")
        reviewer_data[code] = votes
    disputes = []
    consensus = {}
    for identity in sorted(template):
        votes = [rv[identity] for rv in reviewer_data.values()]
        if any(v != votes[0] for v in votes[1:]):
            disputes.append({"case_id": identity[0], "slot": identity[1]})
            continue
        case_id, slot = identity
        origin = key["assignments"][case_id][slot]
        consensus.setdefault(case_id, {})[origin] = votes[0]
    result = {
        "status": "DISAGREEMENTS_REQUIRE_HUMAN_ADJUDICATION" if disputes else "ALL_REVIEWERS_AGREE_REQUIRES_INDEPENDENCE_ATTESTATION",
        "schema": SCHEMA,
        "development_only": True,
        "not_final_blind_benchmark": True,
        "reviewer_visible_category_masked": True,
        "reviewer_visible_case_ids_masked": True,
        "reviewers": list(reviewer_data),
        "reviewer_identities_or_competence_verified": False,
        "human_approval_inferred": False,
        "packet_sha256": sha256(packet_path),
        "base_report_sha256": key.get("base_report_sha256"),
        "tuned_report_sha256": key.get("tuned_report_sha256"),
        "disputed_response_rows": len(disputes),
        "disputed": disputes,
        "metrics": None,
        "limitations": [
            "Previously inspected, synthetic 48-case development diagnostic; not a final blind estimate",
            "Reviewer identities, language competence and independence cannot be verified by software",
            "Reviewer agreement is not proof of semantic correctness; adjudication is a separate human decision",
            "The saved result metadata is checked internally, not independent proof GPU inference occurred",
        ],
    }
    if disputes:
        return result
    # Once independent votes are locked, owner-only metadata restores the
    # real category for subgroup analysis without revealing it to reviewers.
    by_case = key["case_metadata"]
    keys = sorted(consensus)
    if len(keys) != 48 or any(set(consensus[k]) != {"base", "tuned"} for k in keys):
        raise ValueError("Human review is incomplete")
    def stats(case_ids):
        return {field: _pair_metrics(consensus, case_ids, field) for field in VERDICTS}
    result["metrics"] = {
        "all_cases": stats(keys),
        "answerable_including_injection": stats([k for k in keys if by_case[k]["category"] != "unseen_unanswerable"]),
        "unsupported_only": stats([k for k in keys if by_case[k]["category"] == "unseen_unanswerable"]),
        "by_language": {
            lang: stats([k for k in keys if by_case[k]["language"] == lang])
            for lang in ("en", "no", "ar")
        },
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    commands = p.add_subparsers(dest="action", required=True)
    export = commands.add_parser("export", help="Create private blank packet and sealed model mapping")
    export.add_argument("--base", type=Path, required=True)
    export.add_argument("--tuned", type=Path, required=True)
    export.add_argument("--packet", default="reviews/dev48_blank.csv")
    export.add_argument("--key", default="reviews/dev48_private_mapping.json")
    review = commands.add_parser("summarize", help="Require >=2 complete signed-off-by-humans CSVs")
    review.add_argument("--packet", default="reviews/dev48_blank.csv")
    review.add_argument("--key", default="reviews/dev48_private_mapping.json")
    review.add_argument("--reviews", nargs="+", required=True)
    review.add_argument("--output", default="reviews/dev48_review_summary.json")
    args = p.parse_args()
    if args.action == "export":
        output = export_packet(args.root, args.base, args.tuned, args.packet, args.key)
    else:
        output_path = _private_path(Path(args.root).resolve(), args.output, ".json", new=True)
        if args.output == args.key:
            raise ValueError("Cannot overwrite private mapping key")
        output = summarize_reviews(args.root, args.packet, args.key, args.reviews)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as handle:
            json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
