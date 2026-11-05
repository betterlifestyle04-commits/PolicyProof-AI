"""Read-only evidence cross-checks for the Lexsi Track 1 submission gate.

This module is deliberately stricter than a file/flag existence check. A saved
receipt can be owner-edited, so even fully consistent evidence remains a
*manual-review candidate*, never independently authenticated GPU execution,
independent human competence, external license rights, or official submission.
"""
from __future__ import annotations

from pathlib import Path
import re

from policyproof.core import read_jsonl, score, sha256
from scripts.frozen_pair_inference import (
    FINAL_REL, GENERATION, MODEL_ID, SIGNOFF_REL, verify_review_signoff,
)
from scripts.paired_run_recovery import validate_saved_base, validate_saved_tuned
from scripts.final_pair_diagnostics import diagnostics_from_raw_reports
from scripts.verify_hf_collections import (
    _read_json, _url_id, receipt_is_current, validate_manifest,
)


def source_linked_review(root: Path) -> tuple[bool, str]:
    """Requires actual frozen candidate and source CSV/reconciliation provenance."""
    try:
        frozen = root / FINAL_REL
        if frozen.is_symlink() or not frozen.is_file():
            raise ValueError("Frozen 60-case candidate missing")
        receipt = verify_review_signoff(
            root / SIGNOFF_REL, sha256(frozen), root=root,
        )
        if (receipt.get("review_declared") is not True
            or receipt.get("review_source_evidence_validated") is not True
            or receipt.get("reviewers_declared", 0) < 2):
            raise ValueError("Independent reviewer source records incomplete")
        return True, (
            "Source-linked two-or-more reviewer files, full declared votes "
            "and SHA-matched reconciliation are structurally validated. "
            "Reviewer identities, competence and actual human action remain unverified"
        )
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return False, "Source-linked review evidence BLOCKED: " + str(exc)[:180]


def _verified_model_adapter(root: Path, training_receipts: list[tuple[Path, dict]]):
    """Select only an already structurally verified, Track 1 candidate adapter."""
    valid = [(path, info) for path, info in training_receipts
             if info.get("track1_submission_evidence_candidate") is True]
    if len(valid) != 1:
        raise ValueError("Require exactly one validated, eligible Track 1 adapter")
    adapter_root, info = valid[0]
    run = _read_json(adapter_root / "run_info.json", limit=100_000)
    revision = run.get("base_revision", run.get("model_revision"))
    if not isinstance(revision, str) or not re.fullmatch("[a-f0-9]{40}", revision):
        raise ValueError("Trained adapter base revision is unpinned")
    adapter_sha = info.get("adapter_sha256")
    if not isinstance(adapter_sha, str) or not re.fullmatch("[a-f0-9]{64}", adapter_sha):
        raise ValueError("Trained adapter digest is incomplete")
    return revision, adapter_sha


def _paired_deltas(rows: list[dict], base: dict, tuned: dict) -> dict:
    base_map = {x["id"]: x["raw"] for x in base["predictions"]}
    tuned_map = {x["id"]: x["raw"] for x in tuned["predictions"]}
    if set(base_map) != set(tuned_map) or len(base_map) != len(rows):
        raise ValueError("Paired prediction ID coverage incomplete")
    metrics = ("correct", "format_valid", "unsupported_claim",
               "raw_citation_correct", "abstention_correct")
    b = {r["id"]: score(r, base_map[r["id"]]) for r in rows}
    t = {r["id"]: score(r, tuned_map[r["id"]]) for r in rows}
    return {
        metric: {
            "baseline_count": sum(b[r["id"]][metric] for r in rows),
            "candidate_count": sum(t[r["id"]][metric] for r in rows),
            "count_delta": sum(t[r["id"]][metric] - b[r["id"]][metric] for r in rows),
        }
        for metric in metrics
    }


def paired_evaluation_from_sources(
    root: Path, training_receipts: list[tuple[Path, dict]],
    *, review_is_valid: bool,
) -> tuple[bool, str]:
    """Don't trust a stand-alone 'paired_deterministic_audit' JSON status."""
    if not review_is_valid:
        return False, "Source-linked independent gold-review declarations must validate first"
    try:
        revision, adapter_sha = _verified_model_adapter(root, training_receipts)
        benchmark = root / FINAL_REL
        if benchmark.is_symlink():
            raise ValueError("Frozen benchmark cannot be a symlink")
        rows = read_jsonl(benchmark)
        if len(rows) != 60 or len({r["id"] for r in rows}) != 60:
            raise ValueError("Expected exactly 60 distinct final candidate cases")
        digest = sha256(benchmark)
        base_file = root / "artifacts/final_base_inference.json"
        tuned_file = root / "artifacts/final_tuned_inference.json"
        pair_file = root / "artifacts/paired_auditkit.json"
        if (not pair_file.is_file() or pair_file.is_symlink()
            or not 0 < pair_file.stat().st_size <= 2_000_000):
            raise ValueError("Complete 60-case paired AuditKIT receipt missing")
        base = validate_saved_base(
            base_file, rows, dataset_sha256=digest, revision=revision,
            generation_config=GENERATION, model_id=MODEL_ID,
        )
        tuned = validate_saved_tuned(
            tuned_file, rows, dataset_sha256=digest, revision=revision,
            generation_config=GENERATION, model_id=MODEL_ID,
            adapter_sha256=adapter_sha,
        )
        paired = _read_json(pair_file, limit=2_000_000)
        if (paired.get("status") != "paired_deterministic_audit"
            or type(paired.get("sample_count")) is not int
            or paired["sample_count"] != 60
            or paired.get("evaluation_checksum") != digest
            or paired.get("baseline_model_revision") != revision
            or paired.get("adapter_sha256") != adapter_sha
            or paired.get("generation_config") != GENERATION
            or paired.get("verified_human_semantic_improvement") is not False
            or paired.get("semantics_independently_reviewed") is not False
            or not isinstance(paired.get("baseline_auditkit_fingerprint"), str)
            or not paired["baseline_auditkit_fingerprint"]
            or not isinstance(paired.get("candidate_auditkit_fingerprint"), str)
            or not paired["candidate_auditkit_fingerprint"]):
            raise ValueError("Paired audit provenance/safety disclaimer mismatch")
        if paired.get("case_matched_deterministic_counts") != _paired_deltas(rows, base, tuned):
            raise ValueError("Paired counts do not reproduce from full preserved raw reports")
        if (paired.get("case_matched_exploratory_diagnostics")
            != diagnostics_from_raw_reports(rows, base, tuned)):
            raise ValueError(
                "Language, category, or paired uncertainty diagnostics do not "
                "reproduce from complete preserved raw model predictions"
            )
        for label, path in (("base", base_file), ("tuned", tuned_file)):
            declared = paired.get(label + "_report_sha256")
            # Recovered comparisons contain file hashes; direct GPU-generated
            # comparisons do not. If present, they MUST match exactly.
            if declared is not None and declared != sha256(path):
                raise ValueError("Original " + label + " report hash changed")
        if sha256(benchmark) != digest:
            raise ValueError("Frozen benchmark changed during source inspection")
        return True, (
            "60 source-linked base+tuned outputs, scores, paired counts and "
            "exploratory language/category diagnostics "
            "recomputed against unchanged final candidate; GPU logs and "
            "semantic improvements still not independently authenticated"
        )
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return False, "60-case paired evidence BLOCKED: " + str(exc)[:180]


def public_hf_evidence(root: Path) -> tuple[bool, bool, str]:
    """Require a fresh *manifest-matching* anonymous access receipt, not flags.

    Even a stored receipt is unsigned/self-editable, so reverify network access
    live before submitting. Model collection membership is not model relevance.
    """
    manifest = root / "docs/hf_publication_manifest.json"
    receipt = root / "artifacts/hf_anonymous_access_receipt.json"
    if not receipt_is_current(receipt, manifest):
        return False, False, (
            "Missing/stale anonymous Hugging Face API receipt; publication "
            "manifest booleans and URL strings alone are not proof"
        )
    try:
        parsed = validate_manifest(manifest)
        links = _read_json(manifest, limit=100_000)
        record = _read_json(receipt, limit=100_000)
        adapter_url = links.get("adapter_url")
        if not isinstance(adapter_url, str) or len(adapter_url) > 400:
            raise ValueError("Published model adapter URL absent")
        from urllib.parse import urlsplit
        u = urlsplit(adapter_url)
        parts = [x for x in u.path.split("/") if x]
        if (u.scheme != "https" or u.netloc != "huggingface.co"
            or u.username or u.password or u.query or u.fragment
            or u.port or len(parts) != 2):
            raise ValueError("Adapter URL is not a canonical HF model repo URL")
        adapter_id = "/".join(parts)
        if (not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",
            adapter_id
        ) or not any(
            entry.get("id") == adapter_id
            and entry.get("anonymous_access_verified") is True
            for entry in record.get("model_repositories", ())
        )):
            raise ValueError("Adapter is absent from anonymously checked model collection")
        logs = _url_id(links.get("logs_url"), "dataset")
        if not any(
            entry.get("id") == logs and entry.get("anonymous_access_verified") is True
            for entry in record.get("dataset_repositories", ())
        ):
            raise ValueError("Logs repository not verified in dataset collection")
        if record.get("model_collection_requirement_verified") is not True:
            raise ValueError("Collection model requirement not supported by receipt")
        if not parsed.get("dataset_url"):
            raise ValueError("Published dataset URL missing")
        return True, True, (
            "Recent manifest-matched anonymous API receipt checks model+dataset "
            "collection membership and public access; live revalidation, model "
            "relevance and license rights still require manual checks"
        )
    except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
        return False, False, "Hugging Face public evidence BLOCKED: " + str(exc)[:180]
