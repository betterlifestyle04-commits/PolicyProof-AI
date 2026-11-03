"""The 72-case challenge is not model evaluated or human reviewed."""
import copy
from collections import Counter
import pytest

from scripts.generate_track2a_challenge import build_challenge,validate_challenge,attack_text


def test_fixed_72_case_distribution():
    rows=build_challenge()
    assert len(rows)==72
    assert len({r["id"] for r in rows})==72
    counts=Counter(r["category"] for r in rows)
    assert counts=={
        "control":12,"role_spoof":12,"source_spoof":12,
        "encoded_instruction":12,"language_override":12,
        "unsupported_question":12
    }
    assert all(r["model_evaluated"] is False and
               r["used_for_training"] is False and
               r["human_reviewed"] is False for r in rows)


def test_expected_answers_do_not_use_attack_payload():
    rows=build_challenge()
    for row in rows:
        assert "UNLIMITED" not in row["expected_answer"]
        assert "POL-A999" != row["expected_source_id"]
        if row["category"]=="unsupported_question":
            assert row["should_abstain"] is True
            assert row["expected_source_id"] is None
        else:
            assert row["should_abstain"] is False
            assert row["expected_source_id"] in row["context"]


def test_bad_review_or_training_flags_rejected():
    rows=build_challenge()
    bad=copy.deepcopy(rows)
    bad[0]["model_evaluated"]=True
    with pytest.raises(ValueError):
        validate_challenge(bad)
    bad=copy.deepcopy(rows)
    bad[0]["used_for_training"]=True
    with pytest.raises(ValueError):
        validate_challenge(bad)


def test_attack_families_have_fictional_noop_payloads():
    for kind in ("role_spoof","source_spoof","encoded_instruction","language_override"):
        assert attack_text(kind,"no").startswith("\n[UNTRUSTED-NOTE]")
    assert attack_text("control","en")==""
