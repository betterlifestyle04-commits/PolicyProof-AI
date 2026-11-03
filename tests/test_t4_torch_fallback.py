"""CPU tests of fallback data masking and explicit no-GradScaler semantics.

These tests do not pretend to run Tiny Aya or claim GPU compatibility.
"""
import ast
import inspect
import pytest

from scripts.t4_torch_fallback import (encode_pair, train_existing_t4,
    normalize_token_ids, preflight_tokenization)


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
    assert ".train()" not in source
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
