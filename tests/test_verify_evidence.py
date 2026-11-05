"""CPU-only evidence-integrity regression tests using real fixture generation."""
import json
from pathlib import Path

import pytest

from policyproof.core import (
    ROOT, prepare_rows, score, aggregate, write_jsonl, sha256,
)
from scripts.verify_evidence import (
    MAX_BASELINE_REPORT_BYTES, MAX_BASELINE_DATASET_BYTES,
    read_evidence_json, verify_baseline,
)


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


@pytest.mark.parametrize("payload", [
    '{"engine":"transformers","engine":"mock-oracle"}',
    '{"metrics":{"strict_accuracy_pct":100,"strict_accuracy_pct":0}}',
    '{"metrics":{"a":NaN}}',
    '{"metrics":{"a":Infinity}}',
    '[{"engine":"transformers"}]',
    'null',
    '',
])
def test_source_json_rejects_ambiguous_or_non_object_evidence(tmp_path, payload):
    report = tmp_path / "conflicting.json"
    report.write_text(payload)
    with pytest.raises(ValueError):
        read_evidence_json(report)


def test_saved_baseline_refuses_duplicate_json_keys(evidence):
    dataset, report_path, _ = evidence
    text = report_path.read_text()
    report_path.write_text(text.replace(
        '"engine": "transformers"', '"engine":"transformers","engine":"mock-oracle"',
        1,
    ))
    with pytest.raises(ValueError, match="Malformed or ambiguous"):
        verify_baseline(report_path, dataset)


def test_saved_baseline_refuses_nonfinite_json_values(evidence):
    dataset, report_path, _ = evidence
    text = report_path.read_text()
    report_path.write_text(text.replace(
        '"engine": "transformers"', '"engine":"transformers","bogus":NaN',
        1,
    ))
    with pytest.raises(ValueError, match="Malformed or ambiguous"):
        verify_baseline(report_path, dataset)


@pytest.mark.parametrize("target", ["baseline", "dataset"])
def test_saved_baseline_refuses_symlinked_inputs(evidence, tmp_path, target):
    dataset, report_path, _ = evidence
    symlink = tmp_path / ("linked_base.json" if target == "baseline" else "linked_data.jsonl")
    symlink.symlink_to(report_path if target == "baseline" else dataset)
    with pytest.raises(ValueError, match="symlinked"):
        verify_baseline(symlink if target == "baseline" else report_path,
                        symlink if target == "dataset" else dataset)


@pytest.mark.parametrize("target,limit", [
    ("baseline", MAX_BASELINE_REPORT_BYTES),
    ("dataset", MAX_BASELINE_DATASET_BYTES),
])
def test_saved_baseline_refuses_oversized_evidence(evidence, target, limit):
    dataset, report_path, _ = evidence
    (report_path if target == "baseline" else dataset).write_bytes(b"x" * (limit + 1))
    with pytest.raises(ValueError, match="oversized"):
        verify_baseline(report_path, dataset)


def test_saved_baseline_refuses_malformed_prediction_record(evidence):
    dataset, report_path, report = evidence
    report["predictions"][0] = None
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Malformed model prediction"):
        verify_baseline(report_path, dataset)


@pytest.mark.parametrize("value", [None, "not-a-score-list", [{}]])
def test_saved_baseline_refuses_malformed_score_rows(evidence, value):
    dataset, report_path, report = evidence
    report["scores"] = value
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Malformed per-row baseline scores"):
        verify_baseline(report_path, dataset)


def test_saved_baseline_refuses_string_encoded_numeric_metric(evidence):
    dataset, report_path, report = evidence
    report["metrics"]["strict_accuracy_pct"] = "100.0"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Metric discrepancy"):
        verify_baseline(report_path, dataset)
