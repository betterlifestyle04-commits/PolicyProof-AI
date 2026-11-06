"""Immutable deterministic CPU report storage, not model/GPU evidence."""
from __future__ import annotations

import pytest

from scripts.research_report_io import (
    deterministic_report_target, preserve_deterministic_report,
)


def test_report_created_once_then_reused_identically_without_rewriting(tmp_path):
    value = '{"source":"SYNTHETIC_ONLY","passed":false}\n'
    path, reused = preserve_deterministic_report(
        value, "artifacts/cpu-evidence.json", root=tmp_path
    )
    assert not reused
    before = path.read_bytes()
    after, reused = preserve_deterministic_report(
        value, "artifacts/cpu-evidence.json", root=tmp_path
    )
    assert reused and after == path
    assert path.read_bytes() == before


def test_changed_analysis_requires_new_filename_and_preserves_original(tmp_path):
    source = '{"old":"original"}\n'
    path, reused = preserve_deterministic_report(
        source, "artifacts/paired_auditkit.json", root=tmp_path
    )
    with pytest.raises(FileExistsError, match="differs from current source"):
        preserve_deterministic_report(
            '{"old":"changed"}\n', "artifacts/paired_auditkit.json", root=tmp_path
        )
    assert path.read_text() == source
    newer, _ = preserve_deterministic_report(
        '{"old":"changed"}\n', "artifacts/paired_auditkit_v2.json", root=tmp_path
    )
    assert newer.is_file() and path.read_text() == source


def test_absolute_report_destination_is_accepted_only_directly_under_project_artifacts(
    tmp_path,
):
    path = tmp_path / "artifacts" / "dev48_topic.json"
    resolved = deterministic_report_target(path, root=tmp_path)
    assert resolved == path
    created, reused = preserve_deterministic_report(
        '{"test_only":true}\n', path, root=tmp_path
    )
    assert created == path and reused is False


@pytest.mark.parametrize("destination", [
    "README.md",
    "data/final_holdout_v1.jsonl",
    "docs/final_benchmark_review_signoff.json",
    "reviews/owner.csv",
    "artifacts/../data/final_holdout_v1.jsonl",
    "artifacts/./report.json",
    "artifacts/long/nested.json",
    "artifacts//report.json",
    "artifacts/.hidden.json",
    "artifacts/bad.csv",
    "artifacts/bad:name.json",
    "artifacts\\bad.json",
    "artifacts/Réport.json",
])
def test_unsafe_report_destination_does_not_create_any_artifacts(tmp_path, destination):
    with pytest.raises(ValueError, match="Report output"):
        preserve_deterministic_report(
            '{"safe":true}\n', destination, root=tmp_path
        )
    assert not (tmp_path / "artifacts").exists()


def test_absolute_output_outside_artifacts_is_rejected(tmp_path):
    protected = tmp_path / "data"
    protected.mkdir()
    path = protected / "final_holdout_v1.jsonl"
    path.write_bytes(b"RETAIN")
    with pytest.raises(ValueError, match="Report output"):
        preserve_deterministic_report('{"data":"overwriting"}', path, root=tmp_path)
    assert path.read_bytes() == b"RETAIN"


def test_existing_report_symlink_is_never_followed_or_reused(tmp_path):
    other = tmp_path / "private_review.json"
    other.write_text('{"review":"human"}')
    folder = tmp_path / "artifacts"
    folder.mkdir()
    (folder / "report.json").symlink_to(other)
    with pytest.raises(ValueError, match="Unsafe existing report symlink"):
        preserve_deterministic_report(
            '{"review":"human"}', "artifacts/report.json", root=tmp_path
        )
    assert other.read_text() == '{"review":"human"}'


def test_symlinked_artifacts_folder_refused_even_if_empty(tmp_path):
    other = tmp_path / "private"
    other.mkdir()
    (tmp_path / "artifacts").symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe artifacts report directory"):
        preserve_deterministic_report(
            '{"test_only":true}', "artifacts/protected.json", root=tmp_path
        )
    assert list(other.iterdir()) == []


@pytest.mark.parametrize("payload", ["", "x" * 8_000_001])
def test_report_payload_must_be_bounded(tmp_path, payload):
    with pytest.raises(ValueError, match="empty or exceeds"):
        preserve_deterministic_report(payload, "artifacts/test.json", root=tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_existing_oversized_or_directory_target_never_overwritten(tmp_path):
    folder = tmp_path / "artifacts"
    folder.mkdir()
    target = folder / "report.json"
    target.mkdir()
    with pytest.raises(ValueError, match="bounded regular file"):
        preserve_deterministic_report("{}", "artifacts/report.json", root=tmp_path)
    assert target.is_dir()
