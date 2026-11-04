"""Evidence summary is a faithful public TRANSLATION of an audited saved ZIP,
not proof that the original GPU inference occurred nor a new model benchmark.
"""
from __future__ import annotations
import json
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "data/dev48_paired_public_summary.json"


def test_public_evidence_has_correct_case_denominators_and_reported_values():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert data["schema"] == "policyproof_saved_dev48_paired_public_summary_v1"
    assert data["cases"] == 48
    assert data["composition"] == {"supported": 36, "unsupported": 9, "context_injection": 3}
    assert data["language_counts"] == {"en": 16, "no": 16, "ar": 16}
    checks = {
        "schema": (14, 48, 48, "higher"),
        "citation_id": (1, 39, 39, "higher"),
        "strict": (4, 6, 48, "higher"),
        "abstention": (4, 6, 9, "higher"),
        "supported_exact": (0, 0, 39, "higher"),
        "unsupported_claim": (5, 3, 9, "lower"),
    }
    actual = {m["key"]: (
        m["base"], m["tuned"], m["denominator"], m["beneficial_direction"]
    ) for m in data["metrics"]}
    assert actual == checks
    assert sum(entry["base"] for entry in data["strict_by_language"]) == 4
    assert sum(entry["tuned"] for entry in data["strict_by_language"]) == 6
    assert all(entry["denominator"] == 16 for entry in data["strict_by_language"])
    assert sum(data["language_counts"].values()) == data["cases"]


def test_saved_evidence_never_claims_final_science_or_official_submission():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    prohibited_claims = (
        "semantic_human_review_complete",
        "final60_human_review_complete",
        "final60_paired_inference_complete",
        "gpu_execution_independently_reproduced",
        "native_auditkit_independently_reexecuted",
        "public_hf_collections_verified",
        "official_submission_verified",
    )
    assert all(data[k] is False for k in prohibited_claims)
    assert data["verification_scope"] == "original_uploaded_saved_results_internally_recomputed_without_new_gpu_inference"
    assert len(data["paired_bundle_sha256"]) == 64
    assert len(data["dev_dataset_sha256"]) == 64
    assert len(data["adapter_sha256"]) == 64
    assert len(data["requested_base_revision"]) == 40
    assert data["source_document"].endswith("DEV48_PAIRED_GPU_RESULTS_2026-11-05.md")
