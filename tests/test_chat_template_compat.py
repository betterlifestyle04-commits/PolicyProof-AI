"""CPU-only contract regression for Transformers 4.x/5.x chat-template output.

Fake tensors are purely local test doubles, not real Tiny Aya predictions.
No checkpoint, network, GPU, tokenizer or training is executed.
"""
from __future__ import annotations

import inspect
from collections import UserDict

import pytest

from scripts.frozen_pair_inference import (
    _predict_on_gpu, normalize_chat_template_batch,
)


class FakeTensor:
    def __init__(self, shape, dtype="int64", label="tokens"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.label = label
        self.device = "cpu"

    def unsqueeze(self, dimension):
        assert dimension == 0
        return FakeTensor((1, *self.shape), self.dtype, self.label)

    def to(self, device=None, *, dtype=None):
        new = FakeTensor(self.shape, dtype if dtype is not None else self.dtype, self.label)
        new.device = self.device if device is None else device
        return new


class FakeTorch:
    Tensor = FakeTensor
    long = "int64"

    @staticmethod
    def ones_like(tensor):
        return FakeTensor(tensor.shape, dtype="int64", label="generated-all-ones-mask")


def normalize(value):
    return normalize_chat_template_batch(value, FakeTorch, "cuda:0")


def test_transformers_v5_batch_encoding_preserves_supplied_attention_mask():
    tokens = FakeTensor((1, 128), label="v5_ids")
    attention = FakeTensor((1, 128), label="v5_attention_mask")
    # UserDict is a Mapping, just as the real HF BatchEncoding is.
    ids, mask = normalize(UserDict({
        "input_ids": tokens, "attention_mask": attention,
        "token_type_ids": FakeTensor((1, 128), label="ignored-extra-field"),
    }))
    assert ids.shape == (1, 128)
    assert ids.device == "cuda:0"
    assert ids.label == "v5_ids"
    assert mask.shape == ids.shape
    assert mask.device == "cuda:0"
    assert mask.label == "v5_attention_mask"


def test_transformers_v4_legacy_tensor_is_still_supported():
    ids, mask = normalize(FakeTensor((1, 64), label="v4-tensor"))
    assert ids.shape == (1, 64)
    assert mask.shape == (1, 64)
    assert mask.label == "generated-all-ones-mask"


def test_single_1d_tensor_is_batched_without_changing_token_count():
    ids, mask = normalize(FakeTensor((42,)))
    assert ids.shape == (1, 42)
    assert mask.shape == (1, 42)


def test_v5_1d_attention_mask_is_batched_to_match_ids():
    ids, mask = normalize({
        "input_ids": FakeTensor((27,)),
        "attention_mask": FakeTensor((27,), label="original-mask"),
    })
    assert ids.shape == (1, 27)
    assert mask.label == "original-mask"
    assert mask.shape == ids.shape


def test_attention_mask_integer_type_normalization_only():
    ids, mask = normalize({
        "input_ids": FakeTensor((1, 8)),
        "attention_mask": FakeTensor((1, 8), dtype="int32", label="cast-mask"),
    })
    assert ids.dtype == "int64"
    assert mask.dtype == "int64"
    assert mask.label == "cast-mask"


@pytest.mark.parametrize("bad", [
    None,
    [],
    {"attention_mask": FakeTensor((1, 8))},
    {"input_ids": [[1, 2, 3]]},
    {"input_ids": FakeTensor((2, 12))},
    {"input_ids": FakeTensor((1, 0))},
    {"input_ids": FakeTensor((1, 4097))},
    {"input_ids": FakeTensor((1, 12, 1))},
    {"input_ids": FakeTensor((1, 12), dtype="float32")},
])
def test_malformed_chat_template_ids_never_get_passed_to_generate(bad):
    with pytest.raises(RuntimeError, match="chat template|Chat template"):
        normalize(bad)


@pytest.mark.parametrize("bad_attention", [
    "not-a-tensor",
    FakeTensor((2, 10)),
    FakeTensor((1, 9)),
    FakeTensor((1, 10, 1)),
])
def test_bad_attention_masks_fail_closed(bad_attention):
    with pytest.raises(RuntimeError, match="attention_mask|attention mask"):
        normalize({
            "input_ids": FakeTensor((1, 10)),
            "attention_mask": bad_attention,
        })


def test_shared_real_gpu_runner_calls_compat_layer_and_keeps_decoding():
    source = inspect.getsource(_predict_on_gpu)
    assert "return_dict=True" in source
    assert "normalize_chat_template_batch(" in source
    assert "attention_mask=attention_mask" in source
    assert "input_ids=input_ids" in source
    assert "input_ids.shape[-1]:]" in source
    assert "Unexpected tokenizer chat-template encoding" not in source
    assert "torch.inference_mode()" in source
    assert "torch.cuda.empty_cache()" in source
