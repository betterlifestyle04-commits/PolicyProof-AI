"""Evidence-action plan tests: all progress states are synthetic, never GPU proof."""

from scripts.submission_gate import _EVIDENCE_ACTIONS, next_evidence_actions


REQUIRED = tuple(action["check"] for action in _EVIDENCE_ACTIONS)


def _checks(*passed):
    return {key: {"passed": key in passed} for key in REQUIRED}


def test_empty_gate_lists_all_requirements_without_fabricating_completion():
    report = next_evidence_actions(_checks(), REQUIRED)
    assert [item["requirement"] for item in report] == list(REQUIRED)
    assert [item["priority"] for item in report] == list(range(1, len(REQUIRED) + 1))
    assert all(item["action_performed_by_report"] is False for item in report)
    indexed = {item["requirement"]: item for item in report}
    assert indexed["trained_adapter"]["state"] == "ACTIONABLE_LOCALLY"
    assert indexed["human_review_advisory"]["state"] == "ACTIONABLE_LOCALLY"
    assert indexed["paired_finetuned_evaluation"]["state"] == "WAITING_FOR_DEPENDENCIES"
    assert indexed["paired_finetuned_evaluation"]["waiting_on"] == [
        "trained_adapter", "human_review_advisory",
    ]
    assert indexed["public_hf_collections"]["waiting_on"] == ["huggingface_artifacts"]


def test_paired_evaluation_unblocks_only_after_both_training_and_review():
    only_training = _checks("trained_adapter")
    pair = {item["requirement"]: item for item in
            next_evidence_actions(only_training, REQUIRED)}["paired_finetuned_evaluation"]
    assert pair["waiting_on"] == ["human_review_advisory"]
    only_review = _checks("human_review_advisory")
    pair = {item["requirement"]: item for item in
            next_evidence_actions(only_review, REQUIRED)}["paired_finetuned_evaluation"]
    assert pair["waiting_on"] == ["trained_adapter"]
    both = _checks("trained_adapter", "human_review_advisory")
    pair = {item["requirement"]: item for item in
            next_evidence_actions(both, REQUIRED)}["paired_finetuned_evaluation"]
    assert pair["state"] == "ACTIONABLE_LOCALLY"
    assert pair["waiting_on"] == []


def test_publication_sequence_waits_for_adapter_then_hf_access():
    checks = _checks("trained_adapter", "human_review_advisory")
    actions = {item["requirement"]: item for item in next_evidence_actions(checks, REQUIRED)}
    assert actions["huggingface_artifacts"]["state"] == "ACTIONABLE_LOCALLY"
    assert actions["public_hf_collections"]["state"] == "WAITING_FOR_DEPENDENCIES"
    checks["huggingface_artifacts"]["passed"] = True
    actions = {item["requirement"]: item for item in next_evidence_actions(checks, REQUIRED)}
    assert "huggingface_artifacts" not in actions
    assert actions["public_hf_collections"]["state"] == "ACTIONABLE_LOCALLY"


def test_no_remaining_local_gates_still_requires_manual_release_review():
    actions = next_evidence_actions(_checks(*REQUIRED), REQUIRED)
    assert len(actions) == 1
    assert actions[0]["requirement"] == "manual_release_review"
    assert "official organizer submission confirmation" in actions[0]["proof_needed"]
    assert actions[0]["action_performed_by_report"] is False


def test_plan_does_not_include_already_completed_requirements():
    checks = _checks("saved_baseline", "trained_adapter", "public_repo_materials")
    actions = next_evidence_actions(checks, REQUIRED)
    action_names = {action["requirement"] for action in actions}
    assert "saved_baseline" not in action_names
    assert "trained_adapter" not in action_names
    assert "public_repo_materials" not in action_names
    assert "human_review_advisory" in action_names
