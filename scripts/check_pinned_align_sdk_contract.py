"""Inspect the REAL installed AlignTune 0.1.28 and TRL 1.7.1 source contracts.

Uses importlib.metadata + AST, deliberately WITHOUT importing a heavyweight
SDK, downloading model weights, running a GPU, or triggering an optimizer.
Fail-closed if the exact pinned SDK no longer exposes the required training
configuration path or if a different version was installed.

The resulting receipt is *source/API compatibility evidence only*. It does
NOT prove that real TRL SFTConfig can be instantiated, the quantized Tiny Aya
model fits a T4, 40 training steps run, or semantic accuracy improves.
"""
from __future__ import annotations

import argparse
import ast
from importlib import metadata
import hashlib
import json
from pathlib import Path

ALIGN_VERSION = "0.1.28"
TRL_VERSION = "1.7.1"
SOURCES = {
    "aligntune_builder": ("aligntune", "aligntune/core/sft/training_args_builder.py"),
    "aligntune_factory": ("aligntune", "aligntune/core/backend_factory.py"),
    "aligntune_extractor": ("aligntune", "aligntune/utils/config_extractor.py"),
    "aligntune_config": ("aligntune", "aligntune/core/sft/config.py"),
    "aligntune_trl_train": ("aligntune", "aligntune/backends/trl/sft/sft_generation.py"),
    "trl_config": ("trl", "trl/trainer/sft_config.py"),
}


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    functions = [node for node in module.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == name]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise ValueError("Pinned SDK source missing a unique function: " + name)
    return functions[0]


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    classes = [node for node in module.body
               if isinstance(node, ast.ClassDef) and node.name == name]
    if len(classes) != 1:
        raise ValueError("Pinned SDK source missing a unique class: " + name)
    return classes[0]


def _has_call(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, ast.Call) and
        (isinstance(n.func, ast.Name) and n.func.id == name or
         isinstance(n.func, ast.Attribute) and n.func.attr == name)
        for n in ast.walk(tree)
    )


def _contains_named_keyword(tree: ast.AST, keyword: str, identifier: str) -> bool:
    return any(
        isinstance(n, ast.keyword) and n.arg == keyword
        and isinstance(n.value, ast.Name) and n.value.id == identifier
        for n in ast.walk(tree)
    )


def _contains_dict_constant_key(tree: ast.AST, key: str) -> bool:
    return any(
        isinstance(n, ast.Dict)
        and any(isinstance(k, ast.Constant) and k.value == key for k in n.keys)
        for n in ast.walk(tree)
    )


def _has_assignment_target(tree: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, (ast.Assign, ast.AnnAssign))
        and any(isinstance(x, ast.Name) and x.id == name
                for target in (n.targets if isinstance(n, ast.Assign) else [n.target])
                for x in ast.walk(target))
        for n in ast.walk(tree)
    )


def _sft_train_reloads_missing_dataset(module: ast.Module) -> bool:
    """Verify exact upstream TRLSFTTrainer.train() second-load semantics.

    A separate self.dataset cache, not self.train_dataset, controls whether
    AlignTune runs DataManager.load_dataset a second time inside .train().
    We deliberately prepopulate that cache with the exact earlier validated
    training-only object, so this branch must remain recognizable upstream.
    """
    cls = _class(module, "TRLSFTTrainer")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name == "train"]
    if len(methods) != 1:
        return False
    for branch in (n for n in ast.walk(methods[0]) if isinstance(n, ast.If)):
        test = branch.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Attribute)
            and isinstance(test.left.value, ast.Name)
            and test.left.value.id == "self"
            and test.left.attr == "dataset"
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Is)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            continue
        call_reload = any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "self"
            and n.func.attr == "setup_data"
            for stmt in branch.body for n in ast.walk(stmt)
        )
        assigns_cache = any(
            isinstance(stmt, ast.Assign)
            and any(isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr == "dataset"
                    for target in stmt.targets)
            and isinstance(stmt.value, ast.Attribute)
            and isinstance(stmt.value.value, ast.Name)
            and stmt.value.value.id == "self"
            and stmt.value.attr == "train_dataset"
            for stmt in branch.body
        )
        if call_reload and assigns_cache:
            return True
    return False


def inspect_sdk_source(paths: dict[str, Path]) -> dict:
    """Inspect actual installed module sources without importing package code."""
    if set(paths) != set(SOURCES):
        raise ValueError("SDK source set is incomplete")
    modules: dict[str, ast.Module] = {}
    receipts = {}
    for label, path in paths.items():
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"SDK source absent or symlinked: {label}")
        if not 0 < path.stat().st_size < 800_000:
            raise ValueError(f"SDK source unexpectedly empty/large: {label}")
        raw = path.read_bytes()
        try:
            source = raw.decode("utf-8")
            modules[label] = ast.parse(source, filename=str(path))
        except (UnicodeError, SyntaxError) as exc:
            raise ValueError(f"SDK source malformed: {label}") from exc
        receipts[label] = {
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_bytes": len(raw),
        }

    builder = _function(modules["aligntune_builder"], "build_sft_config")
    factory = _function(modules["aligntune_factory"], "create_sft_trainer")
    config_extractor = _class(modules["aligntune_extractor"], "ConfigExtractor")
    method = [x for x in config_extractor.body
              if isinstance(x, ast.FunctionDef)
              and x.name == "extract_extra_and_missing_params"]
    _class(modules["aligntune_config"], "PrecisionType")
    trl_sft = _class(modules["trl_config"], "SFTConfig")
    if not _sft_train_reloads_missing_dataset(modules["aligntune_trl_train"]):
        raise ValueError(
            "AlignTune TRL dataset cache/reload contract changed; "
            "do not assume preverified training data are reused"
        )

    # Check BOTH layers. AlignTune can calculate a correct max_steps initially
    # but then overwrite it while copying train.extra_params to real TRL.
    if (not _contains_dict_constant_key(builder, "max_steps")
        or not _contains_dict_constant_key(builder, "output_dir")
        or not _has_call(builder, "SFTConfig")
        or not _has_call(builder, "extract_extra_and_missing_params")
        or not _contains_named_keyword(builder, "backend_config", "sft_config")
        or not _contains_named_keyword(builder, "config", "config")
        or not _has_assignment_target(builder, "missing")):
        raise ValueError("AlignTune real SFT builder contract changed")
    if (not _has_call(factory, "SFTConfig")
        or not _contains_named_keyword(factory, "extra_params", "kwargs")
        or not _has_call(factory, "get")
        or not any(isinstance(n, ast.Constant) and n.value == "max_steps"
                   for n in ast.walk(factory))):
        raise ValueError("AlignTune real SFT factory kwargs/step contract changed")
    if (len(method) != 1 or not _has_call(method[0], "get_valid_params")
        or not any(isinstance(n, ast.Name) and n.id == "extra_params"
                   for n in ast.walk(method[0]))
        or not _has_assignment_target(method[0], "final_params")):
        raise ValueError("AlignTune extra_params merge contract changed")
    if not any(
        (isinstance(n, ast.AnnAssign)
         and isinstance(n.target, ast.Name)
         and n.target.id == "max_length")
        or (isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "max_length"
                    for target in n.targets for t in ast.walk(target)))
        for n in trl_sft.body
    ):
        raise ValueError("Pinned TRL SFTConfig missing max_length")
    return {
        "status": "PINNED_REAL_SDK_SOURCE_CONTRACT_MATCHED",
        "check_kind": "AST_SOURCE_ONLY_NO_GPU_NO_SDK_RUNTIME_IMPORT",
        "aligntune_factory_kwargs_and_max_steps": True,
        "aligntune_late_training_config_extraction": True,
        "aligntune_train_dataset_cache_branch_verified": True,
        "trl_sft_config_class_and_max_length": True,
        "packages": receipts,
        "real_trl_sftconfig_instantiated": False,
        "quantized_tiny_aya_model_loaded": False,
        "align_trainer_optimizer_steps_executed": 0,
        "model_improvement_verified": False,
    }


def check_installed() -> dict:
    versions = {}
    paths = {}
    for label, (package, relative) in SOURCES.items():
        try:
            dist = metadata.distribution(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Install pinned {package} distribution first") from exc
        version = dist.version
        if version != (ALIGN_VERSION if package == "aligntune" else TRL_VERSION):
            raise ValueError(f"Incorrect {package} version: {version}")
        versions[package] = version
        paths[label] = Path(dist.locate_file(relative))
    return {**inspect_sdk_source(paths), "installed_versions": versions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(check_installed(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
