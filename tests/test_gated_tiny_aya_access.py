"""Gated Hugging Face model access must precede heavy GPU model loading.

All tests are offline and use injected fakes. No HF token, real network,
GPU, login, model weights or trainer.train() is used by these tests.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.restore_colab_session import verify_tiny_aya_gated_access

MODEL = "CohereLabs/tiny-aya-global"
REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"


def fake_access(tmp_path, *, initial_token="PRIVATE_TEST_TOKEN", revision=REVISION,
                allow_config=True, after_login_token="PRIVATE_TEST_TOKEN"):
    calls = []
    state = {"token": initial_token}
    config = tmp_path / "config.json"
    config.write_text('{"model_type":"cohere2"}')
    def get_token():
        return state["token"]
    def login(**kwargs):
        calls.append(("login", kwargs))
        assert kwargs == {"skip_if_logged_in": False}
        state["token"] = after_login_token
    def model_info(model_id, **kwargs):
        calls.append(("model_info", model_id, kwargs))
        assert model_id == MODEL and kwargs == {"token": True}
        return SimpleNamespace(sha=revision)
    def download(**kwargs):
        calls.append(("download", kwargs))
        assert kwargs == {
            "repo_id": MODEL, "filename": "config.json",
            "revision": revision, "token": True,
        }
        if not allow_config:
            raise RuntimeError("401 Unauthorized — GatedRepoError")
        return str(config)
    return calls, dict(get_token_fn=get_token, notebook_login_fn=login,
                       model_info_fn=model_info, download_fn=download)


def test_existing_approved_token_downloads_only_config_no_login(tmp_path, capsys):
    calls, kwargs = fake_access(tmp_path)
    assert verify_tiny_aya_gated_access(**kwargs) == REVISION
    assert [c[0] for c in calls] == ["model_info", "download"]
    output = capsys.readouterr().out
    assert "GATED_TINY_AYA_CONFIG_ACCESS_VERIFIED: TRUE" in output
    assert "PRIVATE_TEST_TOKEN" not in output


def test_absent_token_prompts_secure_widget_before_authenticated_download(tmp_path):
    calls, kwargs = fake_access(tmp_path, initial_token=None)
    assert verify_tiny_aya_gated_access(**kwargs) == REVISION
    assert [c[0] for c in calls] == ["login", "model_info", "download"]
    assert calls[0][1] == {"skip_if_logged_in": False}


def test_gated_401_rejected_even_when_public_model_info_has_valid_sha(tmp_path):
    calls, kwargs = fake_access(tmp_path, allow_config=False)
    with pytest.raises(PermissionError, match="gated Tiny Aya config.json"):
        verify_tiny_aya_gated_access(**kwargs)
    assert [c[0] for c in calls] == ["model_info", "download"]


def test_owner_denies_login_no_download_or_weights(tmp_path):
    calls, kwargs = fake_access(tmp_path, initial_token=None, after_login_token=None)
    with pytest.raises(PermissionError, match="No Hugging Face account token"):
        verify_tiny_aya_gated_access(**kwargs)
    assert [c[0] for c in calls] == ["login"]


@pytest.mark.parametrize("invalid_revision", [None, "", "123", "x"*40, 123, True])
def test_invalid_model_revision_fails_before_any_file_download(tmp_path, invalid_revision):
    calls, kwargs = fake_access(tmp_path, revision=invalid_revision)
    with pytest.raises(PermissionError):
        verify_tiny_aya_gated_access(**kwargs)
    assert [c[0] for c in calls] == ["model_info"]


def test_empty_config_cannot_pass_remote_access_gate(tmp_path):
    calls, kwargs = fake_access(tmp_path)
    (tmp_path / "config.json").write_bytes(b"")
    with pytest.raises(PermissionError):
        verify_tiny_aya_gated_access(**kwargs)


def test_gated_auth_helper_does_not_return_or_print_secrets(tmp_path, capsys):
    calls, kwargs = fake_access(tmp_path)
    observed = verify_tiny_aya_gated_access(**kwargs)
    assert observed == REVISION
    assert "PRIVATE_TEST_TOKEN" not in capsys.readouterr().out
    source = (Path(__file__).resolve().parents[1] /
              "scripts/restore_colab_session.py").read_text()
    assert "revision = verify_tiny_aya_gated_access()" in source
    assert "token=True" in source
    assert 'filename="config.json"' in source
