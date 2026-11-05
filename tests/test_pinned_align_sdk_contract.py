"""Synthetic, offline contract drift tests; no SDK import, model or optimizer."""
import ast
from pathlib import Path

import pytest

from scripts.check_pinned_align_sdk_contract import SOURCES, inspect_sdk_source


BASE_SOURCE = {
    "aligntune_builder": '''
def build_sft_config(config, train_dataset, eval_dataset=None, output_dir="./output", **kwargs):
    config_dict = {"max_steps": config.train.max_steps, "output_dir": output_dir}
    sft_config = SFTConfig(**config_dict)
    missing = extract_extra_and_missing_params(
        backend_config=sft_config, config=config, algorithm="sft"
    )
    for key, value in missing.items():
        setattr(sft_config, key, value)
    return sft_config
''',
    "aligntune_factory": '''
def create_sft_trainer(model_name, dataset_name, output_dir="./output", **kwargs):
    config = SFTConfig(
        train=SFTTrainingConfig(
            max_steps=kwargs.get("max_steps"),
            extra_params=kwargs,
        )
    )
    return config
''',
    "aligntune_extractor": '''
class ConfigExtractor:
    @classmethod
    def extract_extra_and_missing_params(cls, backend_config, config, algorithm):
        extra_params = config.train.extra_params
        final_params = {**extra_params}
        valid = cls.get_valid_params(type(backend_config))
        return {k: v for k, v in final_params.items() if k in valid}
''',
    "aligntune_config": '''
class PrecisionType(Enum):
    FP32 = "fp32"
''',
    "aligntune_trl_train": '''
class TRLSFTTrainer:
    def train(self):
        if self.model is None:
            self.setup_model()
        if self.dataset is None:
            self.setup_data()
            self.dataset = self.train_dataset
        train_result = self.trainer.train()
        return {"steps": getattr(train_result, "global_step", 0)}
''',
    "trl_config": '''
class SFTConfig(TrainingArguments):
    max_length: int = 1024
''',
}


@pytest.fixture
def source_files(tmp_path):
    result = {}
    for name, (_package, relative) in SOURCES.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(BASE_SOURCE[name], encoding="utf-8")
        result[name] = path
    return result


def test_synthetic_declared_sdk_contract_has_honest_limits(source_files):
    report = inspect_sdk_source(source_files)
    assert report["status"] == "PINNED_REAL_SDK_SOURCE_CONTRACT_MATCHED"
    assert report["check_kind"] == "AST_SOURCE_ONLY_NO_GPU_NO_SDK_RUNTIME_IMPORT"
    assert report["real_trl_sftconfig_instantiated"] is False
    assert report["align_trainer_optimizer_steps_executed"] == 0
    assert report["quantized_tiny_aya_model_loaded"] is False
    assert report["model_improvement_verified"] is False
    assert report["aligntune_train_dataset_cache_branch_verified"] is True
    assert report["aligntune_train_returned_step_counter_source_verified"] is True
    assert set(report["packages"]) == set(BASE_SOURCE)
    assert all(len(item["source_sha256"]) == 64
               for item in report["packages"].values())


@pytest.mark.parametrize("name,before,after,message", [
    ("aligntune_builder", '"max_steps"', '"max_optimizer_steps"', "builder"),
    ("aligntune_builder", "extract_extra_and_missing_params", "old_extractor", "builder"),
    ("aligntune_builder", "backend_config=sft_config", "backend_config=None", "builder"),
    ("aligntune_factory", "extra_params=kwargs", "extra_params={}", "factory"),
    ("aligntune_factory", 'kwargs.get("max_steps")', 'kwargs.get("unknown")', "factory"),
    ("aligntune_extractor", "final_params = {**extra_params}", "merged = {**extra_params}", "merge"),
    ("aligntune_config", "class PrecisionType", "class OldPrecisionType", "PrecisionType"),
    ("aligntune_trl_train", "if self.dataset is None:", "if self.dataset is not None:", "dataset cache"),
    ("aligntune_trl_train", "self.setup_data()", "self.download_different_data()", "dataset cache"),
    ("aligntune_trl_train", "self.dataset = self.train_dataset", "self.dataset = []", "dataset cache"),
    ("aligntune_trl_train", '"steps": getattr(train_result, "global_step", 0)',
     '"steps": 40', "result steps contract"),
    ("trl_config", "max_length:", "old_length:", "max_length"),
])
def test_read_only_sdk_contract_rejects_upstream_drift(
    source_files, name, before, after, message,
):
    path = source_files[name]
    path.write_text(path.read_text().replace(before, after), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        inspect_sdk_source(source_files)


def test_sdk_checker_requires_full_five_source_files(source_files):
    source_files.pop("trl_config")
    with pytest.raises(ValueError, match="incomplete"):
        inspect_sdk_source(source_files)


def test_sdk_checker_rejects_symlinked_published_sources(source_files, tmp_path):
    original = source_files["aligntune_builder"]
    target = tmp_path / "real_copy.py"
    target.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(target)
    with pytest.raises(ValueError, match="symlinked"):
        inspect_sdk_source(source_files)


def test_sdk_checker_rejects_oversized_and_malformed_source(source_files):
    path = source_files["aligntune_builder"]
    path.write_bytes(b"x" * 800001)
    with pytest.raises(ValueError, match="large"):
        inspect_sdk_source(source_files)
    path.write_text("def bad_syntax(:")
    with pytest.raises(ValueError, match="malformed"):
        inspect_sdk_source(source_files)
