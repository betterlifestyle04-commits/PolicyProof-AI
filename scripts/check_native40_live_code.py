"""Fail-fast check for stale Stage 13 in-memory inference after a Git pull.

The model is multi-GB and user GPU compute is costly. Colab can update the
source .py files with git pull while retaining the old function objects in
sys.modules. Tracebacks then display NEW linecache source alongside OLD code's
error, disguising the real cause. This check interrogates LIVE code objects,
not inspect.getsource() or source-file text, and performs no model access.

Never reload the partly initialized ML dependency graph in place: require a
normal Colab Runtime > Restart session, followed by Sections 1–5.
"""
from __future__ import annotations

import inspect

FAILURE_GUIDANCE = (
    "STALE_STAGE13_PYTHON_CODE: GitHub source was updated but the Colab "
    "kernel still holds the older paired inference functions. "
    "Select Runtime > Restart session (NOT factory reset), then rerun "
    "Stage 13 Sections 1–5. Verify the updated code before re-enabling "
    "Section 6. Do not rerun training or delete saved evidence."
)


def assert_live_pinned_snapshot_inference(*, frozen_module=None,
                                         dev_module=None) -> dict:
    """Check function-code identity without doing a network/GPU call."""
    if frozen_module is None:
        from scripts import frozen_pair_inference as frozen_module
    if dev_module is None:
        from scripts import dev48_gpu_pair as dev_module

    verifier = frozen_module.verify_loaded_model_revision
    predictor = frozen_module._predict_on_gpu
    dev_predictor = dev_module._predict_on_gpu
    if (
        "pinned_snapshot" not in inspect.signature(verifier).parameters
        or "check_pinned_snapshot_path" not in verifier.__code__.co_names
        or "obtain_pinned_model_snapshot" not in predictor.__code__.co_names
        or "pinned_snapshot" not in predictor.__code__.co_names
        or dev_predictor is not predictor
    ):
        raise RuntimeError(FAILURE_GUIDANCE)
    return {
        "status": "LIVE_STAGE13_PINNED_SNAPSHOT_CODE_CONFIRMED",
        "dev48_uses_current_frozen_predictor": True,
        "current_predictor_uses_authenticated_pinned_snapshot": True,
        "model_loaded": False,
        "gpu_inference_performed": False,
        "training_performed": False,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(assert_live_pinned_snapshot_inference(), indent=2))
