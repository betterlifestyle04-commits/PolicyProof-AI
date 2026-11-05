"""Statically validate committed notebooks without running model or spending GPU.

Checks JSON v4 structure, Python code cell syntax, clean saved outputs,
training-disabled defaults, and absence of obvious credential literals.
Does not assert that Colab/Kaggle providers or GPU training work.
"""
from __future__ import annotations

import ast
import argparse
import json
from pathlib import Path
import re


def validate_notebook(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4 or not isinstance(notebook.get("cells"), list):
        raise ValueError(f"{path}: invalid notebook v4 format")
    code_cells = 0
    magics = 0
    opt_in_flags: set[str] = set()
    forbidden = re.compile(r"(?:hf_[A-Za-z0-9]{24,}|sk-[A-Za-z0-9]{24,})")
    for i, cell in enumerate(notebook["cells"]):
        source = cell.get("source")
        if isinstance(source, list):
            source = "".join(source)
        if not isinstance(source, str):
            raise ValueError(f"{path} cell {i}: invalid source")
        if forbidden.search(source):
            raise ValueError(f"{path} cell {i}: suspicious hardcoded token")
        if cell.get("cell_type") not in ("markdown", "code"):
            raise ValueError(f"{path} cell {i}: unknown cell type")
        if cell["cell_type"] == "markdown":
            continue
        code_cells += 1
        if cell.get("execution_count") is not None:
            raise ValueError(f"{path} cell {i}: output execution counter committed")
        if cell.get("outputs"):
            raise ValueError(f"{path} cell {i}: cell outputs committed")
        if re.search(r"(?m)^\s*RUN_(?:TRAINING|TORCH_FALLBACK|EXTENDED_TRAINING|DEV48_GPU_EVAL|CPU_AUDIT_RECOVERY)\s*=\s*True\b", source):
            raise ValueError(f"{path} cell {i}: automatic training is enabled")
        for flag in re.findall(
            r"(?m)^\s*(RUN_TRAINING|RUN_TORCH_FALLBACK|RUN_MODEL_LOAD|RUN_EXTENDED_TRAINING|RUN_DEV48_GPU_EVAL|RUN_CPU_AUDIT_RECOVERY)\s*=\s*False\b",
            source,
        ):
            opt_in_flags.add(flag)
        # IPython magics are valid notebook code but not Python syntax.
        # Only transform a full-line magic, not arbitrary operators or strings.
        normalized = []
        for line in source.splitlines(keepends=True):
            if line.lstrip().startswith(("%pip ", "%cd ", "!")):
                normalized.append("pass\n")
                magics += 1
            else:
                normalized.append(line)
        try:
            ast.parse("".join(normalized), filename=f"{path}:{i}")
        except SyntaxError as exc:
            raise ValueError(f"{path} cell {i}: invalid Python syntax: {exc}") from exc
    if code_cells < 3:
        raise ValueError(f"{path}: insufficient executable example cells")
    if path.name == "03_aligntune_qlora_t4.ipynb":
        # Prevent a regression to the previous direct write_bytes extraction
        # that could overwrite source data BEFORE its exact SHA was checked.
        notebook_code = "\n".join(
            "".join(cell["source"]) if isinstance(cell["source"], list)
            else cell["source"]
            for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        if ("restore_curated_archive(" not in notebook_code
            or "check_loaded_revision(" not in notebook_code
            or "(CURATED/name).write_bytes" in notebook_code):
            raise ValueError(
                f"{path}: unsafe curator extraction or missing loaded revision check"
            )
    if path.name in ("03_aligntune_qlora_t4.ipynb", "04_kaggle_gpu_fallback.ipynb"):
        if not {"RUN_TRAINING", "RUN_EXTENDED_TRAINING"}.issubset(opt_in_flags):
            raise ValueError(f"{path}: smoke and extended training must both default OFF")
    if path.name == "06_dev48_pair_eval_t4.ipynb" and not {
        "RUN_DEV48_GPU_EVAL", "RUN_CPU_AUDIT_RECOVERY"
    }.issubset(opt_in_flags):
        raise ValueError(f"{path}: paired GPU inference and CPU audit must default OFF")
    return {"file": str(path), "cells": len(notebook["cells"]), "code_cells": code_cells,
            "magic_lines": magics, "training_flags_off": sorted(opt_in_flags),
            "syntax_verified": True, "executed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebooks", default="notebooks")
    args = parser.parse_args()
    notebooks = sorted(Path(args.notebooks).glob("*.ipynb"))
    if len(notebooks) < 3:
        raise ValueError("Expected at least three committed demonstration notebooks")
    result = [validate_notebook(path) for path in notebooks]
    print(json.dumps({"status":"pass", "notebooks":result}, indent=2))


if __name__ == "__main__":
    main()
