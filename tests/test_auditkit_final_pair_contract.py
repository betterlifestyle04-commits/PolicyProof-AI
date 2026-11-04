"""Strict final paired-result metadata tests; no model responses generated."""
from copy import deepcopy
import pytest
from scripts.auditkit_compare import validate_pair


def fixture():
    row = {
        "id":"synthetic-case-01", "language":"en", "category":"final_supported",
        "question":"What is the fictional rule?", "context":"[POL-F101] One token.",
        "expected_answer":"One token.", "expected_source_id":"POL-F101",
        "should_abstain":False,
    }
    common = {
        "engine":"transformers", "is_real_model_result":True,
        "model":"CohereLabs/tiny-aya-global", "model_revision":"a"*40,
        "split":"final_candidate_holdout_v1", "dataset_sha256":"test-hash",
        "generation_config":{"do_sample":False,"num_beams":1,"max_new_tokens":160},
        "predictions":[{"id":row["id"], "raw":
            '{"answer":"One token.","source_id":"POL-F101","abstain":false}'}],
    }
    original = dict(common, is_finetuned=False)
    adapted = dict(common, is_finetuned=True, adapter_sha256="b"*64)
    return row, original, adapted


def test_final_pair_accepts_identical_frozen_metadata_in_cpu_validator():
    row, base, tuned = fixture()
    b, c = validate_pair(base, tuned, [row], "test-hash")
    assert set(b) == set(c) == {row["id"]}


@pytest.mark.parametrize("target,value,match", [
    ("model_revision", "0"*40, "revisions"),
    ("model_revision", "n/a", "revisions"),
    ("split", "unseen-policy", "revisions"),
    ("adapter_sha256", "not-an-adapter-sha", "64-hex"),
    ("generation_config", {"do_sample":True}, "settings"),
    ("generation_config", {}, "settings"),
])
def test_final_pair_refuses_mismatched_provenance(target, value, match):
    row, base, candidate = fixture()
    candidate[target] = value
    with pytest.raises(ValueError, match=match):
        validate_pair(base, candidate, [row], "test-hash")


def test_final_pair_refuses_changed_base_settings():
    row, base, candidate = fixture()
    base["generation_config"] = {"do_sample":True}
    with pytest.raises(ValueError, match="settings"):
        validate_pair(base, candidate, [row], "test-hash")


def test_existing_development_replay_contract_preserved():
    row, base, candidate = fixture()
    base["split"] = candidate["split"] = "unseen-policy"
    base.pop("model_revision")
    candidate.pop("model_revision")
    base.pop("generation_config")
    candidate.pop("generation_config")
    assert validate_pair(base, candidate, [row], "test-hash")[0][row["id"]].startswith("{")
