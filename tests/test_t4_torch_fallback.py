"""CPU tests of fallback data masking and explicit no-GradScaler semantics.

These tests do not pretend to run Tiny Aya or claim GPU compatibility.
"""
import ast
import inspect
import pytest

from scripts.t4_torch_fallback import (encode_pair, train_existing_t4,
    normalize_token_ids, preflight_tokenization, known_safe_encoding_retry,
    training_plan, check_loaded_revision)


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages[0]["role"] == "user"
        assert tokenize is True and add_generation_prompt is True
        return [10, 11, 12, 13]

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [20, 21, 22]


def test_mask_policy_prompt_and_keep_json_output_only():
    data = encode_pair(
        FakeTokenizer(),
        {"instruction": "Return JSON", "input": "Policy [POL-001]", "output": "{}"},
        max_length=20,
    )
    assert data["input_ids"] == [10, 11, 12, 13, 20, 21, 22, 99]
    assert data["labels"] == [-100, -100, -100, -100, 20, 21, 22, 99]
    assert len(data["attention_mask"]) == 8


def test_truncate_prompt_not_json_answer():
    data = encode_pair(FakeTokenizer(),
                       {"instruction": "hi", "input": "", "output": "{}"},
                       max_length=6)
    assert data["labels"] == [-100, -100, 20, 21, 22, 99]
    assert data["input_ids"][-4:] == [20, 21, 22, 99]


def test_fail_if_target_cannot_fit():
    with pytest.raises(ValueError, match="sequence limit"):
        encode_pair(FakeTokenizer(),
                    {"instruction": "hi", "input": "", "output": "{}"},
                    max_length=4)


def test_fallback_does_not_instantiate_trl_or_amp_gradscaler():
    source = inspect.getsource(train_existing_t4)
    tree = ast.parse(source)
    assert "GradScaler(" not in source
    assert "SFTTrainer(" not in source
    assert "trainer.train(" not in source
    assert "model.train()" in source  # PyTorch training mode is required
    assert "torch.autocast(" in source
    assert "optimizer.step()" in source


class EncodingLike:
    """Mimic tokenizers.Encoding returned by the user's installed tokenizer."""
    def __init__(self, ids):
        self.ids = ids


class EncodingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        super().apply_chat_template(messages, tokenize, add_generation_prompt)
        return EncodingLike([10, 11, 12, 13])

    def encode(self, text, add_special_tokens):
        super().encode(text, add_special_tokens)
        return EncodingLike([20, 21, 22])


def test_actual_encoding_objects_are_converted_to_plain_integer_lists():
    data = encode_pair(
        EncodingTokenizer(),
        {"instruction": "Use policy", "input": "", "output": '{"answer":"ok"}'}
    )
    assert data["input_ids"] == [10, 11, 12, 13, 20, 21, 22, 99]
    assert all(type(t) is int for t in data["input_ids"])


def test_batch_encoding_single_sequence_supported():
    assert normalize_token_ids(
        {"input_ids": [EncodingLike([1, 2, 3])]}, field="batch"
    ) == [1, 2, 3]


def test_multiple_encoding_sequences_rejected_not_flattened():
    with pytest.raises(TypeError, match="non-integer"):
        normalize_token_ids(
            [EncodingLike([1]), EncodingLike([2])], field="batch"
        )


def test_rows_preflight_before_gpu():
    rows = [{"instruction": f"Q{i}", "input": "", "output": "{}"}
            for i in range(78)]
    report = preflight_tokenization(EncodingTokenizer(), rows)
    assert report["checked_rows"] == 78
    assert report["training_performed"] is False
    assert report["assistant_loss_masking"] is True


def test_bad_encoding_fails_preflight():
    class Broken(EncodingTokenizer):
        def encode(self, text, add_special_tokens):
            return object()

    with pytest.raises(ValueError, match="row 0"):
        preflight_tokenization(
            Broken(),
            [{"instruction": "test", "input": "", "output": "{}"}]
        )


def test_tensors_will_receive_plain_lists():
    sample = encode_pair(EncodingTokenizer(),
                         {"instruction": "x", "input": "", "output": "{}"})
    assert isinstance(sample["input_ids"], list)
    assert isinstance(sample["labels"], list)
    assert isinstance(sample["attention_mask"], list)
    assert all(isinstance(v, int) for key in sample for v in sample[key])


def test_only_prior_zero_step_tokenizer_typeerror_allows_targeted_retry():
    original = {
        "status": "failed",
        "completed_steps": 0,
        "error_type": "TypeError",
        "error": "'tokenizers.Encoding' object cannot be interpreted as an integer",
    }
    assert known_safe_encoding_retry(original) is True
    assert known_safe_encoding_retry({**original, "completed_steps": 1}) is False
    assert known_safe_encoding_retry({**original, "status": "completed"}) is False
    assert known_safe_encoding_retry({**original, "error": "GPU out of memory"}) is False
    assert known_safe_encoding_retry({**original, "error_type": "RuntimeError"}) is False



def test_extended_training_uses_only_training_source_split_and_pinned_steps():
    plan = training_plan(steps=40)
    assert plan["steps"] == 40
    assert plan["gradient_accumulation"] == 4
    assert plan["training_only_validation"] is True
    assert plan["final_benchmark_or_challenge_used_for_training"] is False
    train = plan["train_indices"]
    val = plan["validation_indices"]
    assert len(train) == 66
    assert len(val) == 12
    assert len(set(train)) == 66
    assert len(set(val)) == 12
    assert not set(train).intersection(val)
    assert set(train + val) == set(range(78))
    assert plan == training_plan(steps=40)


def test_existing_ten_step_smoke_uses_all_78_without_validation():
    plan = training_plan()
    assert plan["steps"] == 10
    assert len(plan["train_indices"]) == 78
    assert plan["validation_indices"] == []
    assert plan["training_only_validation"] is False


@pytest.mark.parametrize("bad_steps", [0, 1, 9, 39, 41, 80, -1, "40", True])
def test_training_plan_rejects_unapproved_counts(bad_steps):
    with pytest.raises(ValueError, match="Only 10-step smoke"):
        training_plan(steps=bad_steps)


@pytest.mark.parametrize("rows", [0, 77, 79, "78", True])
def test_training_plan_never_trains_on_unrecognized_dataset_shape(rows):
    with pytest.raises(ValueError, match="Exactly 78"):
        training_plan(rows, steps=40)


def test_extended_fallback_reuses_same_preflight_and_optimizer_without_trl():
    source = inspect.getsource(train_existing_t4)
    assert "optimizer_steps == 40" in source
    assert "model_sha" in source
    assert "train_indices" in source
    assert "val_indices" in source
    assert '"policyproof_t4_torch_extended.zip"' in source
    assert '"artifacts/torch_fallback_qlora_extended"' in source
    assert "model.eval()" in source
    assert "torch.no_grad()" in source
    assert "optimizer.step()" in source
    assert "trainer.train(" not in source
    assert "SFTTrainer(" not in source



class DummyConfig:
    def __init__(self, revision=None):
        self._commit_hash = revision


class DummyModel:
    def __init__(self, revision=None, *, nested=None):
        self.config = DummyConfig(revision)
        self.base_model = nested


def test_loaded_model_revision_can_be_verified_from_nested_config():
    sha = "a" * 40
    result = check_loaded_revision(DummyModel(nested=DummyModel(sha)), sha)
    assert result["loaded_model_revision_verified"] is True
    assert result["loaded_model_revision_observed"] == sha


def test_hub_model_info_by_itself_is_not_weight_revision_proof():
    result = check_loaded_revision(DummyModel(), "a"*40)
    assert result["loaded_model_revision_verified"] is False
    assert result["loaded_model_revision_observed"] is None


def test_mismatched_actual_loaded_model_revision_fails_closed():
    with pytest.raises(RuntimeError, match="differs from requested"):
        check_loaded_revision(DummyModel("b"*40), "a"*40)
