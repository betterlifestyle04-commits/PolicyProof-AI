"""CPU-only Track 2a scoring handoff tests. Original responses are ORACLE FIXTURES.

No GPU work, actual inference, independent model assessment or HF publishing.
"""
from __future__ import annotations

import json

import pytest

from policyproof.core import sha256
from scripts import score_track2a_72 as scorer
from tests.test_score_track2a_72 import rows_and_report


def evidence_sources(tmp_path):
    rows, report, digest = rows_and_report()
    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "artifacts").mkdir()
    dataset = data / "challenge_v1.jsonl"
    dataset.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows), encoding="utf-8",
    )
    assert sha256(dataset) == digest == scorer.KNOWN_CHALLENGE_SHA256
    source = tmp_path / "artifacts" / "SIMULATED_NOT_REAL_GPU.json"
    source.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    return rows, report, dataset, source


def test_complete_cpu_audit_receipt_bound_to_exact_original_sources(tmp_path):
    pytest.importorskip("auditkit")
    rows, report, dataset, source = evidence_sources(tmp_path)
    original = (source.read_bytes(), dataset.read_bytes())
    metrics, audit = scorer.score_saved_72(
        dataset, source, "artifacts/cpu_72_analysis", root=tmp_path
    )
    folder = tmp_path / "artifacts/cpu_72_analysis"
    done = json.loads((folder / "EVIDENCE_COMPLETE.json").read_text(encoding="utf-8"))
    assert done["status"] == "SAVED_72_CPU_AUDIT_COMPLETED"
    assert done["model_inference_executed_in_this_process"] is False
    assert done["human_semantic_review_completed"] is False
    assert done["original_model_report_sha256"] == metrics["original_model_report_sha256"] == sha256(source)
    assert done["frozen_dataset_sha256"] == sha256(dataset)
    assert done["measured_report_sha256"] == sha256(folder / "measured_72_report.json")
    assert done["auditkit_receipt_sha256"] == sha256(folder / "auditkit_72_receipt.json")
    assert len(metrics["by_language"]) == 3
    assert {lang: info["n"] for lang, info in metrics["by_language"].items()} == {
        "en": 24, "no": 24, "ar": 24,
    }
    assert all(info["attack_family_counts"]["control"] == 4
               for info in metrics["by_language"].values())
    assert all(info["metrics"]["samples"] == 24
               for info in metrics["by_language"].values())
    assert audit["original_frozen_dataset_sha256"] == sha256(dataset)
    assert (source.read_bytes(), dataset.read_bytes()) == original
    with pytest.raises(FileExistsError, match="never overwrite"):
        scorer.score_saved_72(dataset, source, "artifacts/cpu_72_analysis", root=tmp_path)


@pytest.mark.parametrize("output", [
    "README.md", "docs/reviewer.csv", "data/final_holdout_v1.jsonl",
    "artifacts/../../data/final_holdout_v1.jsonl",
    "artifacts/nested/result", "artifacts//double", "artifacts/.hidden",
    "artifacts/created.json", "artifacts/dir:name", "/tmp/other_folder",
])
def test_output_folder_cannot_escape_artifacts_or_look_like_file(tmp_path, output):
    _, _, dataset, source = evidence_sources(tmp_path)
    with pytest.raises(ValueError, match="output directory|directory name"):
        scorer.score_saved_72(dataset, source, output, root=tmp_path)
    assert not (tmp_path / "artifacts" / "EVIDENCE_COMPLETE.json").exists()


def test_symlinked_output_folder_refused_with_original_unchanged(tmp_path):
    _, _, dataset, source = evidence_sources(tmp_path)
    target = tmp_path / "private-review"
    target.mkdir()
    (tmp_path / "artifacts" / "replay").symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="never overwrite"):
        scorer.score_saved_72(dataset, source, "artifacts/replay", root=tmp_path)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("bad", [
    '{"engine":"transformers","engine":"oracle","predictions":[]}',
    '{"a":{"source":"real","source":"forged"}}',
    '{"engine":"transformers","bad":NaN}',
    '{"engine":"transformers","bad":Infinity}',
])
def test_ambiguous_saved_json_never_runs_auditkit(tmp_path, monkeypatch, bad):
    _, _, dataset, source = evidence_sources(tmp_path)
    source.write_text(bad, encoding="utf-8")
    monkeypatch.setattr(
        scorer, "evaluate_real_run",
        lambda *_: pytest.fail("Ambiguous evidence reached AuditKIT"),
    )
    with pytest.raises(ValueError, match="Malformed or ambiguous"):
        scorer.score_saved_72(dataset, source, "artifacts/replay", root=tmp_path)
    assert not (tmp_path / "artifacts/replay").exists()


@pytest.mark.parametrize("which", ["dataset", "model"])
def test_symlinked_original_inputs_refused(tmp_path, which):
    _, _, dataset, source = evidence_sources(tmp_path)
    selected = dataset if which == "dataset" else source
    backup = tmp_path / "unchanged.txt"
    backup.write_bytes(selected.read_bytes())
    selected.unlink()
    selected.symlink_to(backup)
    with pytest.raises(ValueError, match="symlinked"):
        scorer.score_saved_72(dataset, source, "artifacts/replay", root=tmp_path)
    assert backup.read_bytes() == (selected.read_bytes())
    assert not (tmp_path / "artifacts/replay").exists()


@pytest.mark.parametrize("which", ["dataset", "model"])
def test_source_modified_during_scoring_blocks_publication(tmp_path, monkeypatch, which):
    _, _, dataset, source = evidence_sources(tmp_path)
    mutated = dataset if which == "dataset" else source
    def fake_scoring(*_):
        mutated.write_bytes(mutated.read_bytes() + b"x")
        return {"status": "SIMULATED_ONLY"}, {"status": "SIMULATED_ONLY"}
    monkeypatch.setattr(scorer, "evaluate_real_run", fake_scoring)
    with pytest.raises(ValueError, match="changed during AuditKIT"):
        scorer.score_saved_72(dataset, source, "artifacts/replay", root=tmp_path)
    assert not (tmp_path / "artifacts/replay").exists()


def test_interrupted_two_receipt_save_never_emits_complete_marker(tmp_path, monkeypatch):
    _, _, dataset, source = evidence_sources(tmp_path)
    monkeypatch.setattr(
        scorer, "evaluate_real_run",
        lambda *_: ({"status": "SIMULATED_ONLY"}, {"status": "SIMULATED_ONLY"}),
    )
    original_write = scorer.write_evidence_once
    def fail_second(path, content):
        if path.name == "auditkit_72_receipt.json":
            raise OSError("unit-test simulated disk failure")
        return original_write(path, content)
    monkeypatch.setattr(scorer, "write_evidence_once", fail_second)
    with pytest.raises(OSError, match="disk failure"):
        scorer.score_saved_72(dataset, source, "artifacts/partial", root=tmp_path)
    folder = tmp_path / "artifacts/partial"
    assert (folder / "measured_72_report.json").exists()
    assert not (folder / "EVIDENCE_COMPLETE.json").exists()
    assert source.is_file() and dataset.is_file()
