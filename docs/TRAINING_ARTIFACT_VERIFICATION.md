# Training checkpoint verification — CPU-only release gate

**Status (November 4, 2026):** The project has **no verified successfully trained checkpoint** in the public repository. This module prepares us to validate a future Colab/Kaggle artifact; it does not train Tiny Aya, use GPU, or prove model improvement.

## Why this matters

A text file with `{"status":"completed","optimizer_steps":10}` and a nonempty fake `adapter_model.safetensors` can look plausible to a basic existence check. The new `scripts.verify_training_archive` refuses to trust that combination.

The verifier checks these structural and provenance properties **without requiring torch, an HF token or GPU**:

- ZIP path safety, bounded unpacked size, missing/duplicate files and symlink safety.
- Real `safetensors` header structure, dtype/shape/offsets, contiguous tensor data and LoRA A/B tensors. For a Track 1 release candidate, the adapter must additionally match the audited Tiny Aya QLoRA configuration of **288 LoRA tensors and 3,833,856 trainable elements**.
- Actual SHA-256 of the adapter tensor file and PEFT config against values in `run_info.json`.
- Completed-run receipt with at least ten **self-reported** optimizer steps, original model ID, audited 78-row CuratorKIT training SHA-256, full 40-character source commit, and accepted training backend.
- Distinction between a **declared AlignTune trainer** and a **manual PyTorch fallback**; the latter cannot be called an executed AlignTune Track 1 training step.
- A full 40-character base-model revision is required for a Track 1 submission-evidence *candidate*. Missing revision makes the release flag false rather than silently guessing.

## How to check a completed training ZIP

On a local trusted Python 3.11+ environment after getting a genuine training ZIP:

```bash
python -m scripts.verify_training_archive \
  --archive /path/to/policyproof_align_smoke.zip
```

For a strict Track 1 candidate check:

```bash
python -m scripts.verify_training_archive \
  --archive /path/to/policyproof_align_smoke.zip --strict-track1
```

The strict variant exits nonzero if the backend is a manual fallback or the original base-model revision is not pinned. For an extracted `run_info.json` + `adapter/` directory:

```bash
python -m scripts.verify_training_archive \
  --directory artifacts/align_qlora_smoke
```

The ZIP is read in a temporary directory without overwriting source or research files.

## What a valid artifact still does **not** prove

A self-reported training run is not independent evidence that the model actually performed gradient updates. Hashes prove **internal consistency**, not data authenticity. The verifier therefore always emits `training_independently_verified: false` and `model_improvement_verified: false`, even for a structurally valid adapter.

For a scientifically valid final submission, still obtain actual GPU console/job evidence, pinned base-model commit, trainer logs, genuine adapter weights, followed by **paired base-vs-tuned inference** on a human-reviewed frozen benchmark. Preserve the raw outputs and AuditKIT receipts. Do not call successful loading, a mock trainer, or the manual fallback a completed AlignTune pipeline. Do not put gated base model weights into a public adapter repo.

## AlignTune recovery and PyTorch fallback differences

- `scripts.complete_t4_training.recover_train_save` now declares `trainer_backend: aligntune_wrapper` only after its wrapper reports a completed ten-step run. Pass a real 40-character model revision when known.
- `scripts.t4_torch_fallback.train_existing_t4` saves its adapter configuration hash and explicitly declares `trainer_backend: manual_torch_smoke_fallback` and `aligntune_trainer_completed: false`.
- Both save adapter weight digests; neither should invent a revision or claim improved safety.

The existing `scripts.submission_gate` now checks adapter file integrity before it will acknowledge a trained-adapter candidate. It remains `NOT_READY` until legitimate model/paired evaluation and external Hugging Face publishing evidence is available.

## Verification tests

```bash
python -m pytest -q tests/test_verify_training_archive.py tests/test_submission_gate.py
```

The fixtures in these tests are **synthetic two-tensor files for testing the verifier only**. They are not model weights, performance metrics or evidence of completed training.

## Extended 40-step research adapter (PyTorch fallback; not official AlignTune execution)

The new `manual_torch_extended_fallback` receipt is accepted for **CPU artifact-integrity checks only** when it records exactly **40 optimizer steps**, a pinned 40-character original Tiny Aya Git commit, the audited 78-row dataset digest, a deterministic **66-train / 12-training-only-validation** split and finite positive internal validation-loss values at steps 10, 20, 30 and 40. It does not qualify as a completed AlignTune Track 1 trainer run, and internal training-source validation loss cannot establish improvement in behavior on independent evaluation cases.

On actual GPU success, run `python -m scripts.verify_training_archive --archive policyproof_t4_torch_extended.zip`. Preserve the entire original zip and GPU console logs. [How to execute the GPU training](GPU_TRAINING_NEXT_STEPS.md).
