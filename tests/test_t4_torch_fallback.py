"""CPU tests of fallback data masking and explicit no-GradScaler semantics.

These tests do not pretend to run Tiny Aya or claim GPU compatibility.
"""
import ast
import inspect
import pytest

from scripts.t4_torch_fallback import encode_pair, train_existing_t4


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
