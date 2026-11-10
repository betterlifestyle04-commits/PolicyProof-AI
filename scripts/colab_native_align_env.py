"""Repair Colab text-only native AlignTune import environment without GPU work.

Colab 2026.10 uses Python 3.13 and can leave an incompatible optional
torchvision installed. Transformers lazily imports torchvision through image
utilities when initializing some text-only SDKs, surfacing a misleading
BloomPreTrainedModel error. Training Tiny Aya text-only requires torch, NOT
torchvision. This removes only that optional vision package in Colab and
checks the exact requested AlignTune/TRl/BNB versions in a NEW interpreter.

No model weights downloaded, training, GPU inference, HF token or data read.
Changing packages requires a Colab runtime RESTART to clear poisoned imports.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
import subprocess
import sys

PINNED = {"aligntune": "0.1.28", "trl": "1.7.1", "bitsandbytes": "0.50.2"}

FRESH_IMPORT_CODE = """\
from aligntune.core.backend_factory import create_sft_trainer
from trl import SFTTrainer
from peft import PeftModel
from transformers import BloomPreTrainedModel
import bitsandbytes
assert callable(create_sft_trainer)
print("FRESH_INTERPRETER_TEXT_ONLY_NATIVE_IMPORTS_PASSED")
"""


def installed(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def plan_changes(packages: dict[str, str | None]) -> tuple[list[str], bool]:
    """Pure decision: install pinned trio if needed; remove optional torchvision.

    Never replace torch / torchvision with mismatched wheels or execute model
    code. Runtime restart required if any package changed.
    """
    required_install = any(packages.get(name) != want for name, want in PINNED.items())
    remove_vision = packages.get("torchvision") is not None
    return (
        ["aligntune==0.1.28", "trl==1.7.1", "bitsandbytes==0.50.2", "jedi"]
        if required_install else [],
        remove_vision,
    )


def verify_text_only_native_env(*, runner=subprocess.run, python: str | None = None,
                                package_versions: dict | None = None) -> dict:
    """Install exact requested packages and remove *optional* torchvision.

    Called only by Stage 11 Step 2, never after a live trainer is built.
    Uses a subprocess for import probe; a failed import cannot poison the
    active notebook kernel. After installation/uninstall ALWAYS stop and
    require Runtime > Restart session, then rerun Steps 1–2.
    """
    executable = python or sys.executable
    packages = package_versions or {
        name: installed(name) for name in (*PINNED, "torchvision", "torch")
    }
    to_install, remove_vision = plan_changes(packages)
    if to_install:
        runner(
            [executable, "-m", "pip", "install", "--prefer-binary", *to_install],
            check=True,
        )
    # Even if pip install brought torchvision back as a transitive package,
    # text-only training cannot tolerate a broken optional vision extension.
    # Uninstalling it is safer than guessing which CUDA torchvision wheel
    # matches Colab's preinstalled torch; torch itself stays untouched.
    tv_after = installed("torchvision") if package_versions is None else packages.get("torchvision")
    if remove_vision or tv_after is not None:
        runner([executable, "-m", "pip", "uninstall", "-y", "torchvision"], check=True)
    changes = bool(to_install or remove_vision or tv_after is not None)
    if changes:
        raise RuntimeError(
            "Colab training dependencies changed, including optional torchvision. "
            "RESTART THE RUNTIME now (Runtime > Restart session), then rerun "
            "Stage 11 Sections 1 and 2 only. This clears partial torchvision/"
            "Transformers imports; no model or optimizer has been run. "
            "If imports still fail, select Runtime Version 2026.07 (Python 3.12) "
            "and a T4 GPU in Change runtime type, then rerun on a fresh session."
        )
    after = {name: installed(name) for name in PINNED} if package_versions is None else {
        name: packages.get(name) for name in PINNED
    }
    if after != PINNED:
        raise RuntimeError("Pinned AlignTune / TRL / BNB versions not installed: " + str(after))
    try:
        proc = runner(
            [executable, "-c", FRESH_IMPORT_CODE],
            check=False, text=True, capture_output=True,
            timeout=180,
            env={**os.environ, "USE_TF": "0", "USE_FLAX": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Fresh isolated native AlignTune import timed out; no training started") from exc
    if proc.returncode != 0:
        stderr = (proc.stderr or "")[-4500:]
        raise RuntimeError(
            "Native AlignTune import preflight FAILED before training. "
            "Captured diagnostic:\n" + stderr
            + "\nIf caused by broken torch/torchvision, make sure Runtime "
            "was restarted after uninstall. Otherwise choose Colab runtime "
            "2026.07 (Python 3.12) with T4 GPU and rerun setup. Do NOT "
            "start the optimizer or install BloomPreTrainedModel separately."
        )
    return {
        "status": "FRESH_TEXT_ONLY_NATIVE_ALIGNTUNE_IMPORT_PASSED_NO_TRAINING",
        "pinned_versions": after,
        "python_version": sys.version.split()[0],
        "optional_torchvision_installed": False,
        "torch_version_unchanged_by_helper": installed("torch"),
        "training_started": False,
        "model_weights_loaded": False,
        "gpu_inference_performed": False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(verify_text_only_native_env(), indent=2))
