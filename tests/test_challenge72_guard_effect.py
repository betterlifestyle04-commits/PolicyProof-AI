"""Synthetic CPU-only checks. No genuine Tiny Aya GPU outputs are created."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from policyproof.core import make_gold, sha256
from scripts.challenge72_guard_effect import (
    REPORT_SCHEMA, analyze_72_guard_effect, audit_from_saved_files,
)
from scripts.generate_track2a_challenge import build_challenge
from scripts.score_track2a_72 import KNOWN_CHALLENGE_SHA256


def fixture(tmp_path):
    rows = build_challenge()
    data = tmp_path / "data" / "challenge_v1.jsonl"
    data.parent.mkdir(parents=True)
    data.write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True)+"\n" for x in rows),
        encoding="utf-8",
    )
    assert sha256(data) == KNOWN_CHALLENGE_SHA256
    report = {
        "engine": "transformers", "is_real_model_result": True,
        "is_finetuned": False, "model": "CohereLabs/tiny-aya-global",
        "model_revision": "a"*40,
        "created_utc": datetime(2026, 11, 4, tzinfo=timezone.utc).isoformat(),
        "dataset_sha256": KNOWN_CHALLENGE_SHA256,
        "generation_config": {"do_sample": False, "max_new_tokens": 160,
                              "num_beams": 1},
        "actual_gpu_inference_metadata": {
            "gpu": "FAKE_GPU_TEST_ONLY",
            "torch_version": "SYNTHETIC",
            "loaded_model_revision_verified": True,
            "loaded_model_revision_observed": "a"*40,
        },
        "predictions": [
            {"id": row["id"],
             "raw": make_gold(row["expected_answer"], row["expected_source_id"])}
            for row in rows
        ],
    }
    raw = tmp_path / "artifacts" / "original72.json"
    raw.parent.mkdir()
    raw.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return rows, data, raw, report


def test_synthetic_gold_outputs_produce_zero_guard_regressions(tmp_path):
    rows, data, raw, report = fixture(tmp_path)
    result = analyze_72_guard_effect(rows, report, sha256(data))
    assert result["schema"] == REPORT_SCHEMA
    assert result["samples"] == 72
    assert result["underlying_answerable_task_stems"] == 12
    assert result["training_or_weight_change_performed"] is False
    assert result["model_inference_executed_in_this_process"] is False
    assert result["human_semantic_review_completed"] is False
    assert result["predeclared_inference_guard_used_reference_labels"] is False
    overall = result["whole_sample_descriptive_only"]["metrics"]
    assert overall["strict_reference_match"]["raw_before"] == 72
    assert overall["strict_reference_match"]["raw_after"] == 72
    assert overall["schema_validity"]["raw_before"] == 72
    assert overall["schema_validity"]["raw_after"] == 72
    assert result["guard_action_counts"]["valid_abstention"] == 12
    assert sum(v["n"] for v in result["attack_family_slices"].values()) == 72
    assert sum(v["n"] for v in result["language_slices"].values()) == 72
    for family, pairs in result["matched_control_vs_attack_by_family"].items():
        assert pairs["n_matched_task_pairs"] == 12
        assert pairs["strict_match_before_guard"]["control_correct_attack_correct"] == 12
        assert pairs["strict_match_after_guard"]["control_correct_attack_correct"] == 12
        assert pairs["p_values_across_attack_families_not_independent"] is True


def test_guard_fixes_invalid_unsupported_json_but_cannot_repair_bad_citation(tmp_path):
    rows, data, raw, report = fixture(tmp_path)
    unsupported = next(p for p in report["predictions"] if "-UNSUPPORTED-" in p["id"])
    unsupported["raw"] = "INVALID_JSON_FROM_FAKE_MODEL"
    malicious = next(p for p in report["predictions"] if p["id"].endswith("role_spoof"))
    malicious["raw"] = '{"answer":"UNLIMITED","source_id":"POL-A999","abstain":false}'
    result = analyze_72_guard_effect(rows, report, sha256(data))
    groups = result["attack_family_slices"]
    unsupported_metric = groups["unsupported_question"]["metrics"]["correct_abstention_unsupported"]
    assert unsupported_metric["raw_before"] == 11
    assert unsupported_metric["raw_after"] == 12
    assert unsupported_metric["improved_cases"] == 1
    assert unsupported_metric["regressed_cases"] == 0
    attack_metric = groups["role_spoof"]["metrics"]["strict_reference_match"]
    assert attack_metric["raw_before"] == 11
    assert attack_metric["raw_after"] == 11
    assert result["guard_action_counts"]["unrecognized_source"] >= 1
    assert groups["role_spoof"]["literal_attack_marker_echoes_before"] == 1
    assert groups["role_spoof"]["literal_attack_marker_echoes_after"] == 0


def test_new_cpu_diagnostics_are_immutable_and_contain_no_raw_outputs(tmp_path):
    rows, data, saved, report = fixture(tmp_path)
    output = "artifacts/guard72_descriptive.json"
    first = audit_from_saved_files(data, saved, output=output, root=tmp_path)
    assert first["case_count"] == 72
    assert first["new_gpu_inference_performed"] is False
    assert first["reused_identical_report"] is False
    target = tmp_path / output
    blob = target.read_bytes()
    text = blob.decode("utf-8")
    assert "FAKE_GPU_TEST_ONLY" not in text
    assert "Within 14 calendar days" not in text
    assert "CHALLENGE-POL-A201-en-control" not in text
    assert '"source_model_report_sha256"' in text
    second = audit_from_saved_files(data, saved, output=output, root=tmp_path)
    assert second["reused_identical_report"] is True
    assert target.read_bytes() == blob
    # Source mutation after write cannot silently be repaired by a cached report.
    old = saved.read_bytes()
    report["predictions"][0]["raw"] = "CHANGED_FAKE_ORIGINAL"
    saved.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(FileExistsError, match="choose a NEW output filename"):
        audit_from_saved_files(data, saved, output=output, root=tmp_path)
    assert target.read_bytes() == blob
    assert saved.read_bytes() != old


@pytest.mark.parametrize("change", ["wrong_sha", "missing_case", "duplicate_case",
                                    "fake_oracle_status", "tuned"])
def test_missing_or_fabricated_72_inference_never_produces_results(tmp_path, change):
    rows, data, saved, report = fixture(tmp_path)
    report = deepcopy(report)
    if change == "wrong_sha":
        report["dataset_sha256"] = "f"*64
    elif change == "missing_case":
        report["predictions"].pop()
    elif change == "duplicate_case":
        report["predictions"][1]["id"] = report["predictions"][0]["id"]
    elif change == "fake_oracle_status":
        report["is_real_model_result"] = False
    elif change == "tuned":
        report["is_finetuned"] = True
    saved.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        audit_from_saved_files(data, saved, output="artifacts/guard72.json", root=tmp_path)
    assert not (tmp_path / "artifacts" / "guard72.json").exists()


@pytest.mark.parametrize("output", [
    "data/frozen.jsonl",
    "artifacts/nested/out.json",
    "artifacts/../data/challenge_v1.jsonl",
    "../escape.json",
])
def test_unsafe_output_refused_without_overwriting_source(tmp_path, output):
    rows, data, saved, report = fixture(tmp_path)
    original = data.read_bytes()
    with pytest.raises(ValueError):
        audit_from_saved_files(data, saved, output=output, root=tmp_path)
    assert data.read_bytes() == original


def test_symlinked_original_gpu_report_is_rejected(tmp_path):
    rows, data, saved, report = fixture(tmp_path)
    link = saved.parent / "linked.json"
    link.symlink_to(saved)
    with pytest.raises(ValueError, match="symlinked"):
        audit_from_saved_files(data, link, output="artifacts/guard72.json", root=tmp_path)
    assert not (saved.parent / "guard72.json").exists()


def test_repeated_group_control_id_is_required(tmp_path):
    rows, data, saved, report = fixture(tmp_path)
    damaged = deepcopy(rows)
    control = next(r for r in damaged if r["category"] == "control")
    control["question"] = "TAMPERED CONTROL QUESTION"
    # This catches an invalid matched-control comparison, even when an
    # attacker keeps all expected case ID strings intact.
    with pytest.raises(ValueError, match="matching frozen control"):
        analyze_72_guard_effect(damaged, report, sha256(data))
