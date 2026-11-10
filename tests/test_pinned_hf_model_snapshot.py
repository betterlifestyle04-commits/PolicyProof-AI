"""Offline tests for immutable authenticated Tiny Aya cache-snapshot source binding.

No gated model download, GPU, token, optimizer or actual model weights here.
The mini shard bytes and model objects are synthetic test-only fixtures.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.frozen_pair_inference import (
    MODEL_ID, check_pinned_snapshot_path, obtain_pinned_model_snapshot,
    verify_loaded_model_revision,
)

SHA = "af89d219b53ed9b13b8a4645f9c8028973510324"


def fake_snapshot(tmp_path):
    p = (tmp_path / "models--CohereLabs--tiny-aya-global" /
         "snapshots" / SHA)
    p.mkdir(parents=True)
    (p / "config.json").write_text('{"model_type":"cohere2"}')
    (p / "tokenizer_config.json").write_text('{"test_only":true}')
    (p / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        }
    }))
    for shard in ("model-00001-of-00002.safetensors",
                  "model-00002-of-00002.safetensors"):
        (p / shard).write_bytes(b"SIMULATED_TENSOR_SHARD_NO_REAL_MODEL_WEIGHTS")
    return p


def model_with_reported_commit(commit):
    return SimpleNamespace(config=SimpleNamespace(_commit_hash=commit))


def test_checked_authenticated_snapshot_binds_missing_model_config_commit(tmp_path):
    snapshot = fake_snapshot(tmp_path)
    assert check_pinned_snapshot_path(snapshot, SHA) == snapshot
    assert verify_loaded_model_revision(
        model_with_reported_commit(None), SHA, pinned_snapshot=snapshot
    ) == SHA
    # No snapshot means no accepted silent missing commitment.
    with pytest.raises(RuntimeError, match="observed commit SHA"):
        verify_loaded_model_revision(model_with_reported_commit(None), SHA)


def test_direct_commit_metadata_remains_valid_and_mismatch_still_fails(tmp_path):
    snapshot = fake_snapshot(tmp_path)
    assert verify_loaded_model_revision(model_with_reported_commit(SHA), SHA) == SHA
    with pytest.raises(RuntimeError, match="does not match pinned SHA"):
        verify_loaded_model_revision(
            model_with_reported_commit("b"*40), SHA, pinned_snapshot=snapshot
        )
    with pytest.raises(RuntimeError, match="malformed"):
        verify_loaded_model_revision(
            model_with_reported_commit("not_sha"), SHA, pinned_snapshot=snapshot
        )


def test_snapshot_download_exact_repo_revision_token_and_filtered_file_types(tmp_path):
    snapshot = fake_snapshot(tmp_path)
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        return str(snapshot)
    result = obtain_pinned_model_snapshot(SHA, downloader=download)
    assert result == snapshot
    assert len(calls) == 1
    call = calls[0]
    assert call["repo_id"] == MODEL_ID
    assert call["revision"] == SHA
    assert call["token"] is True
    assert "*.safetensors" in call["allow_patterns"]
    assert "*.json" in call["allow_patterns"]
    assert "local_dir" not in call
    assert "force_download" not in call


@pytest.mark.parametrize("case", [
    "wrong_revision",
    "wrong_model_id",
    "custom_dir",
    "missing_config",
    "missing_tokenizer",
    "missing_shard",
    "traversal_shard",
    "weight_map_missing",
    "symlinked_snapshot",
])
def test_missing_or_misbound_pinned_snapshot_never_counts_as_loaded_commit(tmp_path, case):
    snapshot = fake_snapshot(tmp_path)
    expected = snapshot
    if case == "wrong_revision":
        expected = "b"*40
    elif case == "wrong_model_id":
        snapshot = snapshot.parent.parent.parent / "models--other--model" / "snapshots" / SHA
        snapshot.mkdir(parents=True)
    elif case == "custom_dir":
        snapshot = snapshot.parent.parent.parent / "random_file_dir"
        snapshot.mkdir()
    elif case == "missing_config":
        (snapshot / "config.json").unlink()
    elif case == "missing_tokenizer":
        (snapshot / "tokenizer_config.json").unlink()
    elif case == "missing_shard":
        (snapshot / "model-00002-of-00002.safetensors").unlink()
    elif case == "traversal_shard":
        (snapshot / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"a": "../../other.safetensors"}
        }))
    elif case == "weight_map_missing":
        (snapshot / "model.safetensors.index.json").write_text(
            '{"weight_map":{}}'
        )
    elif case == "symlinked_snapshot":
        real = snapshot
        snapshot = snapshot.parent / (SHA + "-alias")
        snapshot.symlink_to(real, target_is_directory=True)
    with pytest.raises((RuntimeError, ValueError)):
        check_pinned_snapshot_path(snapshot, expected)


def test_authenticated_download_cannot_accept_wrong_or_stale_snapshot(tmp_path):
    snap = fake_snapshot(tmp_path)
    with pytest.raises(RuntimeError, match="not bound"):
        obtain_pinned_model_snapshot(
            "b"*40, downloader=lambda **kwargs: str(snap)
        )
    with pytest.raises(ValueError):
        obtain_pinned_model_snapshot(
            "main", downloader=lambda **kwargs: "will_not_be_called"
        )


def test_model_loaded_only_from_snapshot_and_metadata_not_overstated():
    source = (Path(__file__).resolve().parents[1] /
              "scripts/frozen_pair_inference.py").read_text()
    assert "snapshot = obtain_pinned_model_snapshot(revision)" in source
    assert source.count("str(snapshot), local_files_only=True") >= 2
    assert "pinned_snapshot=snapshot" in source
    assert "authenticated_pinned_hf_snapshot_source" in source
    assert '"loaded_revision_independently_gpu_attested": False' in source
    assert "trainer.train(" not in source
