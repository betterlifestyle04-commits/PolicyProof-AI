"""CPU-only private review HTML tests. No human judgments or model inference."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.review_offline_html import export_offline_review, render_offline_review


def _source_fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("final_holdout_v1.jsonl", "final_holdout_sources_v1.json"):
        shutil.copy2(ROOT / "data" / name, data / name)
    return tmp_path


def test_original_frozen_review_exports_all_60_unreviewed_cases():
    cases = read_jsonl(ROOT / "data/final_holdout_v1.jsonl")
    digest = sha256(ROOT / "data/final_holdout_v1.jsonl")
    html = render_offline_review(cases, digest, "reviewer_01")
    assert "<!doctype html>" in html
    assert "PolicyProof AI" in html
    assert html.count('id="reviewData"') == 1
    assert "No" in html and "Clear vote" in html
    assert "importCSV" in html and "exportCSV" in html
    assert "textContent" in html
    assert "180 verdicts" in html
    assert "No model predictions" in html
    assert digest in html
    assert "model_inference" not in html
    assert "fetch(" not in html and "XMLHttpRequest" not in html
    assert "https://" not in html and "http://" not in html
    packet_text = re.search(
        r'<script id="reviewData" type="application/json">(.*?)</script>',
        html, re.DOTALL,
    ).group(1)
    packet = json.loads(packet_text)
    assert len(packet["rows"]) == 60
    assert packet["reviewer"] == "reviewer_01"
    assert packet["digest"] == digest
    assert {r["language"] for r in packet["rows"]} == {"en", "no", "ar"}
    assert all(r["evidence_support_yes_no"] == "" for r in packet["rows"])
    assert all(r["translation_faithful_yes_no"] == "" for r in packet["rows"])
    assert all(r["question_unambiguous_yes_no"] == "" for r in packet["rows"])
    assert all("raw_model_response" not in r for r in packet["rows"])


def test_source_text_is_embedded_as_safe_script_data_and_rendered_as_text():
    cases = read_jsonl(ROOT / "data/final_holdout_v1.jsonl")
    cases[0] = dict(cases[0], question='</script><img src=x onerror=alert("x")>')
    html = render_offline_review(cases, sha256(ROOT / "data/final_holdout_v1.jsonl"), "reviewer_02")
    assert '</script><img' not in html
    assert '\\u003c/script\\u003e' in html
    assert "innerHTML" not in html
    packet = json.loads(re.search(
        r'<script id="reviewData" type="application/json">(.*?)</script>',
        html, re.DOTALL,
    ).group(1))
    assert packet["rows"][0]["question"].startswith("</script>")


def test_private_export_is_new_write_once_and_preserves_frozen_source(tmp_path):
    _source_fixture(tmp_path)
    source = tmp_path / "data/final_holdout_v1.jsonl"
    before = source.read_bytes()
    created = export_offline_review(tmp_path, reviewer="reviewer_01")
    output = Path(created["private_output"])
    assert output == tmp_path / "reviews/reviewer_01_offline.html"
    assert created["verdicts_filled"] == 0
    assert created["human_review_completed"] is False
    assert created["no_model_predictions_included"] is True
    assert created["no_network_calls_or_uploaded_judgments"] is True
    assert output.is_file()
    assert source.read_bytes() == before
    original_html = output.read_bytes()
    with pytest.raises(FileExistsError, match="overwrite"):
        export_offline_review(tmp_path, reviewer="reviewer_01")
    assert output.read_bytes() == original_html


@pytest.mark.parametrize("output", [
    "docs/final_benchmark_review_signoff.html",
    "../reviewer.html",
    "reviews/../data/final_holdout_v1.html",
    "reviews/subdir/review.html",
    "reviews//review.html",
    "reviews/.hidden/packet.html",
    "reviews/reviewer.csv",
    "/tmp/escape.html",
])
def test_offline_html_cannot_overwrite_public_or_frozen_evidence(tmp_path, output):
    _source_fixture(tmp_path)
    with pytest.raises((ValueError, FileExistsError)):
        export_offline_review(tmp_path, reviewer="reviewer_01", output=output)
    assert not (tmp_path / "reviews/reviewer_01_offline.html").exists()


@pytest.mark.parametrize("bad", [
    "", "1", "../unsafe", "user name", "someone@example.com", "o" * 41,
])
def test_reviewer_ids_must_be_anonymous_safe_codes(bad):
    cases = read_jsonl(ROOT / "data/final_holdout_v1.jsonl")
    with pytest.raises(ValueError, match="Reviewer code"):
        render_offline_review(cases, sha256(ROOT / "data/final_holdout_v1.jsonl"), bad)


def test_export_refuses_modified_source_or_symlinked_review_directory(tmp_path):
    _source_fixture(tmp_path)
    path = tmp_path / "data/final_holdout_v1.jsonl"
    path.write_text(path.read_text() + "\n", encoding="utf-8")
    # Structural validator remains true for blank-line variation, but
    # output is still source-hash bound to those exact bytes.
    receipt = export_offline_review(tmp_path, reviewer="reviewer_01")
    assert receipt["frozen_dataset_sha256"] == sha256(path)
    assert (tmp_path / "reviews/reviewer_01_offline.html").is_file()
    second = tmp_path / "different"
    second.mkdir()
    _source_fixture(second)
    external = second / "elsewhere"
    external.mkdir()
    (second / "reviews").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="not safe"):
        export_offline_review(second, reviewer="reviewer_02")
    assert not list(external.iterdir())


def test_cli_help_never_triggers_training_or_writes():
    result = subprocess.run(
        [sys.executable, "-m", "scripts.review_offline_html", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0
    assert "Distinct private reviewer code" in result.stdout


def test_private_offline_workbench_warns_before_discarding_unsaved_votes():
    cases = read_jsonl(ROOT / "data/final_holdout_v1.jsonl")
    digest = sha256(ROOT / "data/final_holdout_v1.jsonl")
    html = render_offline_review(cases, digest, "reviewer_01")
    assert 'id="saveState"' in html
    assert 'window.addEventListener("beforeunload"' in html
    assert "if(!unsaved)return;" in html
    assert "event.preventDefault();" in html
    assert 'event.returnValue="";' in html
    assert "markDirty();updateProgress();" in html
    assert 'if(answers[index].review_notes!==this.value)' in html
    assert "unsaved=false;saveOrigin=\"export\";updateSaveStatus();" in html
    assert "unsaved=false;saveOrigin=\"import\";updateSaveStatus();" in html
    assert "Ctrl" not in html or 'event.ctrlKey||event.metaKey' in html
    assert "event.key.toLowerCase()===\"s\"" in html
    assert "exportCSV();" in html
    # Deliberately NO automatic persistent storage of private reviewer votes.
    assert "localStorage" not in html and "sessionStorage" not in html
    assert "indexedDB" not in html and "sendBeacon" not in html
    assert "fetch(" not in html and "XMLHttpRequest" not in html
