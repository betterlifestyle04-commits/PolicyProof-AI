"""CPU-only evidence-integrity regression tests using real fixture generation."""
import json
from pathlib import Path

import pytest

from policyproof.core import (
    ROOT, prepare_rows, score, aggregate, write_jsonl, sha256,
)
from scripts.verify_evidence import verify_baseline


@pytest.fixture
def evidence(tmp_path):
    policies = json.loads((ROOT / "data/policies.json").read_text(encoding="utf-8"))
    _train, _para, unseen = prepare_rows(policies)
    test_path = tmp_path / "unseen.jsonl"
    write_jsonl(test_path, unseen)
    predictions = [
        {
            "id": row["id"],
            "raw": json.dumps({
                "answer": row["expected_answer"],
                "source_id": row["expected_source_id"],
                "abstain": row["should_abstain"],
            }),
        } for row in unseen
    ]
    mapping = {p["id"]: p["raw"] for p in predictions}
    scores = [score(row, mapping[row["id"]]) for row in unseen]
    report = {
        "engine": "transformers",
        "is_real_model_result": True,
        "is_finetuned": False,
        "split": "unseen-policy",
        "model": "CohereLabs/tiny-aya-global",
        "dataset_sha256": sha256(test_path),
        "predictions": predictions,
        "scores": scores,
        "metrics": aggregate(scores),
    }
    report_path = tmp_path / "base.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return test_path, report_path, report


def test_evidence_recompute_matches_exact_prediction_rows(evidence):
    dataset, report_path, report = evidence
    validated = verify_baseline(report_path, dataset)
    assert validated["sample_count"] == 48
    assert validated["prediction_count"] == 48
    assert validated["recomputed_metrics"]["strict_accuracy_pct"] == 100.0
    assert validated["model_training_performed"] is False


def test_evidence_rejects_changed_prediction(evidence):
    dataset, report_path, report = evidence
    report["predictions"][0]["raw"] = '{}'
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Metric discrepancy|Stored per-row score"):
        verify_baseline(report_path, dataset)


def test_evidence_rejects_duplicate_id(evidence):
    dataset, report_path, report = evidence
    report["predictions"][1]["id"] = report["predictions"][0]["id"]
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Duplicate model prediction ID"):
        verify_baseline(report_path, dataset)


def test_evidence_rejects_dataset_hash_change(evidence):
    dataset, report_path, report = evidence
    dataset.write_bytes(dataset.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dataset SHA256"):
        verify_baseline(report_path, dataset)


def test_evidence_rejects_claimed_fake_engine(evidence):
    dataset, report_path, report = evidence
    report["engine"] = "mock-oracle"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="actual Transformers"):
        verify_baseline(report_path, dataset)
