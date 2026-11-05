# Safer Colab recovery: curated data integrity and model-revision provenance

**Implementation status (Nov 5, 2026):** Code and CPU-only tests are available. **No new GPU training has been performed by this change.** This workflow does not convert the previously saved 40-step direct-PyTorch fallback into a native AlignTune result.

## Why this was necessary

The previous Stage 3 notebook copied allowlisted CuratorKIT ZIP entries directly into its training folder with `write_bytes`; the old recovery helper copied data before checking the exact curated-data digest. A partial, corrupted or different ZIP could disturb files left from an earlier Colab runtime. Separately, a Hugging Face `model_info().sha` tells us the revision returned by the Hub API, not independently which model revision the local runtime actually loaded.

## What now happens

1. The CuratorKIT ZIP must pass the existing full archive/member inventory and checksum validator.
2. An allowlisted subset is **staged in a temporary private folder**, not the target training folder.
3. The full 78-row training data are validated and their SHA-256 must equal `4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c`.
4. Before any write, all existing target files are compared byte-for-byte. Conflicts and symlinks **abort**, preserving owner evidence.
5. New files are created only with exclusive mode, and partial files created by this attempt are cleaned up on failure. Restoring the identical archive twice is idempotent.
6. The requested Tiny Aya model revision must be a complete 40-hex SHA **before** importing the heavyweight GPU stack. After loading the quantized PEFT model, the existing `check_loaded_revision` inspects `config._commit_hash` on the loaded model and nested models.
7. A conflicting observed SHA aborts. If the loaded model does not expose a SHA, the report says `loaded_model_revision_verified=false`. This is **not** silently upgraded to proof of pinned model weights.

This hardens experimental provenance. It does **not** cryptographically authenticate the hosted model's provenance or claim that a native AlignTune trainer successfully completed optimizer steps.

## Run the updated original notebook

[PolicyProof Stage 3 — AlignTune T4](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/03_aligntune_qlora_t4.ipynb)

In a fresh notebook, use Stage 2 to upload the original `policyproof_curatorkit_train_v1.zip`. The upload helper will validate it **before** creating any training file. If the prior file is different, it refuses the restore rather than overwriting. For a cleared Colab runtime, use the one-call `restore_interactive` cell, which also verifies the 78 training examples and displays the requested versus observed base revision.

Before intentionally running training, examine `loaded_revision_inspection`:

- `loaded_model_revision_verified: true` means the observed model config hash matches the requested SHA (subject to self-reported metadata provenance).
- `false` means the loaded revision was **not observable**; proceed only for exploratory work and never claim the weights are independently pinned.
- A hash **mismatch raises an error** and the model must not be used for evidence claims.

The Stage 3 notebook defaults to **training OFF**. It must not rerun an existing completed checkpoint in-place or overwrite original failure receipts. The actual outstanding milestone is a genuine successful AlignTune-native optimizer run under its licensed/training environment, followed by independent human review and a frozen paired evaluation.

## Reproduce safeguards in CI

```bash
python -m pytest -q tests/test_restore_colab_session.py
python -m scripts.validate_notebooks
```

These tests use fake small archives and fake model configs. They verify *software behavior*, **not** live T4 operation, Hugging Face connectivity, model quality or completion of a submission.
