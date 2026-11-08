"""Synthetic packet-mechanics tests. Never treat these as actual ML evidence.

The independent real-toolkit CI job exercises unmocked source verification.
Here a fixed synthetic source assertion stands in for real CuratorKIT/AuditKIT.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile, ZIP_STORED, ZipInfo

import pytest

from policyproof.core import ROOT
from scripts import build_judge_evidence_packet as packet


GOOD_PROOF = {
    "48_saved_source_outputs_recomputed": True,
    "real_curatorkit_native_receipts_source_checked": True,
    "original_auditkit_native_receipt_source_checked": True,
    "72_curated_challenge_cases_verified_UNSCORED": True,
    "48_paired_dev_public_summary_NOT_original_archive": True,
    "track2a_scientific_status": "RESEARCH_READY_HF_PUBLICATION_BLOCKED",
}


def fake_sources(tmp_path, monkeypatch):
    original = {}
    for name in packet.SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            (ROOT / "data/dev48_paired_public_summary.json").read_bytes()
            if name == "data/dev48_paired_public_summary.json"
            else ("SYNTHETIC CPU RESEARCH FIXTURE NOT REAL " + name + "\n").encode()
        )
        path.write_bytes(content)
        original[name] = sha256(content).hexdigest()

    def synthetic_proof(root):
        for name, digest in original.items():
            p = Path(root) / name
            if not p.is_file() or sha256(p.read_bytes()).hexdigest() != digest:
                raise ValueError("Simulated original source replay rejected an altered file")
        return GOOD_PROOF.copy()

    monkeypatch.setattr(packet, "verify_sources", synthetic_proof)
    return original


def test_judge_packet_byte_identical_and_source_only(tmp_path, monkeypatch):
    original = fake_sources(tmp_path, monkeypatch)
    first = packet.make_packet(tmp_path)
    second = packet.make_packet(tmp_path)
    assert first == second
    assert len(first) <= packet.ZIP_LIMIT
    status = packet.inspect_packet_bytes(first)
    assert status["status"] == "JUDGE_PACKET_SOURCE_REPLAY_VERIFIED_NOT_SUBMITTED"
    assert status["original_scored_development_cases"] == 48
    assert status["new_challenge_cases_unscored"] == 72
    assert status["native_source_consistency_recomputed"] is True
    assert all(status[k] is False for k in (
        "original_gpu_inference_independently_attested",
        "human_semantic_review_verified", "public_hf_publication_verified",
        "official_lexsi_submission_verified",
    ))
    with ZipFile(BytesIO(first)) as z:
        files = set(z.namelist())
        assert files == packet.ALLOWED
        assert len(files) == len(packet.SOURCES) + 3
        assert not any("token" in x or "reviewer.csv" in x for x in files)
        manifest = json.loads(z.read(packet.MANIFEST))
        assert manifest["status"] == "CPU_SOURCE_REPLAYED_RESEARCH_UNPUBLISHED_NOT_SUBMITTED"
        assert manifest["final_model_cases"] == 0
        assert manifest["original_evaluated_cases"] == 48
        assert manifest["new_unscored_challenge_cases"] == 72
        assert manifest["unverified_requirements"] == {
            name: False for name in packet.UNCHECKED
        }
        assert manifest["private_reviewer_votes_included"] is False
        assert manifest["gated_model_weights_included"] is False
        assert manifest["paired_dev48_original_zip_included"] is False
        assert manifest["source_consistency_verified"] == GOOD_PROOF
        assert manifest["source_sha256"] == original
        assert b"NOT an official submission" in z.read(packet.GUIDE)


def test_write_new_packet_is_exclusive_and_preserves_original_files(tmp_path, monkeypatch):
    original = fake_sources(tmp_path, monkeypatch)
    receipt = packet.write_packet(tmp_path)
    target = tmp_path / "artifacts/policyproof_lexsi_judge_evidence_UNSUBMITTED.zip"
    assert target.is_file()
    assert receipt["zip_sha256"] == sha256(target.read_bytes()).hexdigest()
    assert receipt["official_lexsi_submission_verified"] is False
    packet_blob = target.read_bytes()
    with pytest.raises(FileExistsError, match="Never overwrite"):
        packet.write_packet(tmp_path)
    assert target.read_bytes() == packet_blob
    assert {
        name: sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in packet.SOURCES
    } == original


@pytest.mark.parametrize("output", [
    "README.zip", "docs/judge.zip", "artifacts/nested/judge.zip",
    "artifacts/.hidden.zip", "artifacts/../private.zip",
    "artifacts//two.zip", "artifacts/judge.json",
    "../artifacts/judge.zip", "/tmp/judge.zip",
    "artifacts/judge\\name.zip", "artifacts/judge:private.zip",
])
def test_output_path_cannot_override_source_or_escape_artifacts(tmp_path, output):
    with pytest.raises(ValueError, match="direct artifacts"):
        packet.write_packet(tmp_path, output)
    assert not list(tmp_path.iterdir())


def test_symlinked_public_source_rejected(tmp_path, monkeypatch):
    fake_sources(tmp_path, monkeypatch)
    path = tmp_path / "docs/TRACK2A_SUBMISSION_ABSTRACT.md"
    backup = tmp_path / "source_copy.txt"
    backup.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(backup)
    # Real validator and mock strict validator both refuse changed input
    # provenance; never package by following a symlink.
    with pytest.raises(ValueError, match="symlink"):
        packet._file_bytes(tmp_path, "docs/TRACK2A_SUBMISSION_ABSTRACT.md")


def test_source_modified_even_if_json_manifest_rewritten_cannot_qualify(
    tmp_path, monkeypatch,
):
    fake_sources(tmp_path, monkeypatch)
    safe = packet.make_packet(tmp_path)
    with ZipFile(BytesIO(safe)) as z:
        originals = {name: z.read(name) for name in z.namelist()}
    # Forge a source file, rehash both SHA256SUMS and the index as a forger might.
    changed = dict(originals)
    target = "docs/TRACK2A_SUBMISSION_ABSTRACT.md"
    changed[target] = b"FORGED INCOMPLETE MODEL SUCCESS!"
    manifest = json.loads(changed[packet.MANIFEST])
    manifest["source_sha256"][target] = sha256(changed[target]).hexdigest()
    changed[packet.MANIFEST] = (json.dumps(
        manifest, indent=2, ensure_ascii=False, sort_keys=True,
    ) + "\n").encode()
    changed[packet.HASHES] = "".join(
        sha256(changed[n]).hexdigest() + "  " + n + "\n"
        for n in sorted(changed) if n != packet.HASHES
    ).encode()
    stream = BytesIO()
    with ZipFile(stream, "w") as z:
        for name in sorted(changed):
            info = ZipInfo(name, (2026, 11, 8, 0, 0, 0))
            info.compress_type = ZIP_STORED
            info.external_attr = 0o100644 << 16
            z.writestr(info, changed[name])
    with pytest.raises(ValueError, match="original source replay"):
        packet.inspect_packet_bytes(stream.getvalue())


def test_truncated_or_extra_member_cannot_verify(tmp_path, monkeypatch):
    fake_sources(tmp_path, monkeypatch)
    raw = packet.make_packet(tmp_path)
    with pytest.raises((ValueError, OSError, EOFError, __import__("zipfile").BadZipFile)):
        packet.inspect_packet_bytes(raw[:-100])
    with ZipFile(BytesIO(raw)) as z:
        files = {n: z.read(n) for n in z.namelist()}
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_STORED) as z:
        for name, content in files.items():
            z.writestr(name, content)
        z.writestr("secrets/hf_token.txt", "hf_SOME_DANGEROUS_PRIVATE_DATA")
    with pytest.raises(ValueError, match="inventory|Unsafe"):
        packet.inspect_packet_bytes(stream.getvalue())


def test_fake_dev48_metric_can_never_enter_judge_packet(tmp_path, monkeypatch):
    fake_sources(tmp_path, monkeypatch)
    source = tmp_path / "data/dev48_paired_public_summary.json"
    payload = json.loads(source.read_text())
    payload["metrics"][0]["tuned"] = 777
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Simulated original|Public dev48"):
        packet.make_packet(tmp_path)


def test_potential_credentials_are_blocked(tmp_path, monkeypatch):
    fake_sources(tmp_path, monkeypatch)
    name = "docs/TRACK2A_SUBMISSION_ABSTRACT.md"
    (tmp_path / name).write_text("hf_" + "X" * 45)
    with pytest.raises(ValueError, match="apparent credential"):
        packet._file_bytes(tmp_path, name)


def test_default_preflight_cli_is_not_a_publication_action():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.build_judge_evidence_packet", "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0
    assert "--inspect" in proc.stdout and "--build" in proc.stdout
