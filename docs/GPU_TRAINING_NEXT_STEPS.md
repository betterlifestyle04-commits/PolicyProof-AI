# Continue REAL Tiny Aya training on a GPU — 40-step route

**Status — November 4, 2026:** The repository provides a tested *execution plan*, not a completed GPU training run. The authoring session and GitHub Actions use CPU only. A real Colab/Kaggle T4/P100 accelerator and access to the gated original model are still required.

## Why continue

The first ten optimizer steps in the existing notebook were only intended to test the training integration. We now support a **separate 40-step PyTorch QLoRA fallback**, using the **already-loaded AlignTune 4-bit Tiny Aya LoRA model** after previous TRL automatic mixed-precision failures.

**Do not confuse this with a completed AlignTune training pipeline.** The direct PyTorch route deliberately bypasses TRL trainer initialization; receipts say `manual_torch_extended_fallback`, `aligntune_trainer_completed: false`, and `performance_improvement_verified: false`. A trained adapter is not yet proof of improved safety.

The 78 examples come only from the original verified CuratorKIT synthetic training export (SHA-256 `4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c`). The extended route deterministically divides them into **66 optimizer-training examples and 12 internal validation-loss examples**, with random seed 42. It performs at most 40 optimizer steps and accumulates gradients over four microbatches per optimizer update. A loss on the 12 internal rows is computed every 10 steps.

**The 48-case already-inspected development set, 60-case proposed final benchmark and 72-case unscored challenge are completely excluded from the training schedule and internal validation.** The small internal loss check is *not* a fair blind model evaluation: all 78 rows originated from a small synthetic training corpus, with correlated prompts and policies. It cannot prove semantic accuracy or safety.

## Option A — Google Colab (recommended if a T4 is available)

Open the actual GitHub-hosted notebook:

**[Open Tiny Aya training notebook in Colab](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/03_aligntune_qlora_t4.ipynb)**

1. **Runtime → Change runtime type → T4 GPU**, then connect; run the GPU check. Colab free GPU access and duration vary by account and demand.
2. Make sure your Hugging Face account has accepted the model's license and access conditions for [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global). Authenticate *privately* with a read-only token through the notebook's authorized login flow, never paste a token into Git or chat. The model is gated and governed by CC-BY-NC and Cohere's Acceptable Use Policy.
3. Upload **the original**, already generated `policyproof_curatorkit_train_v1.zip` with **78 training rows** when requested by notebook Step 2 or the recovery path. Do not upload a safety benchmark or ad-hoc 120-case combined dataset.
4. Follow the notebook's documented dependency and restore-model steps until it says **RESTORE COMPLETE**, zero optimizer steps; verify a 40-character `MODEL_SHA`, 4-bit quantization, 78 training rows and expected LoRA adapter size. *Do not load a second copy of the model if one is already in memory.*
5. **Do not run** the old ten-step AlignTune training or PyTorch fallback cells. Navigate to the final **Optional: genuine 40-step QLoRA training run** cell. Leave `RUN_TRAINING=False` and `RUN_TORCH_FALLBACK=False`. Set **only** `RUN_EXTENDED_TRAINING=True` when ready, then execute that final cell.
6. On actual success, the cell downloads **`policyproof_t4_torch_extended.zip`**. Preserve the original ZIP, complete training logs, GPU name, pinned base Git revision, 40 recorded optimizer steps, all training losses and training-only validation losses. **Do not reset the runtime or claim success before the adapter is downloaded.**

If training fails with an OOM, NaN or account GPU quota, stop and preserve the failure receipt; repeated blind retries can waste GPU and overwrite evidence. Do not use a GPU or paid compute that you haven't explicitly authorized.

## Option B — Kaggle if Colab GPU unavailable

Open the [Kaggle notebook source](../notebooks/04_kaggle_gpu_fallback.ipynb) in a Kaggle notebook; enable **Settings → Internet** and **Accelerator → T4 x2 or P100** according to available free quota.

Attach your **private** `policyproof_curatorkit_train_v1.zip` through **Add Input**, or upload it privately. Access the gated Hugging Face model with a private `HF_TOKEN` in **Kaggle Secrets**, not in source cells.

Run notebook steps 1–3 to restore the existing 4-bit PEFT model and 78 training rows. Keep `RUN_TRAINING=False`; in the new optional extended-training cell, explicitly set `RUN_EXTENDED_TRAINING=True`. On success, download `policyproof_t4_torch_extended.zip` from Kaggle working/output Files.

If an accelerator is unavailable, **do not represent CPU tests as model training.**

## Verify the returned adapter on CPU

After a real GPU session generates the archive, use the project's structural verifier:

```bash
python -m scripts.verify_training_archive --archive policyproof_t4_torch_extended.zip
```

A successful result confirms **internal file format, hashes, 40 self-reported steps, the exact training-only split, and recorded loss values**. It still explicitly reports that training execution and model improvement are **not independently verified**. Never pass this manual PyTorch fallback artifact off as successful AlignTune training or a strict Track 1 release.

When an actual ZIP is available, next steps are to inspect its exact adapter, replay paired **base-vs-adapter** generation using an identical pinned base revision and deterministic decoding, use native AuditKIT on both saved outputs, and report strict and qualitative metrics. Our existing final 60-case gate requires **genuine independent reviewer files** first. If those are not yet ready, evaluate the already inspected **48-case development set** with new, honestly labeled paired inference; it is not a blind final benchmark.

## Strategic deadline

The initial Lexsi submission deadline is **November 14, 2026, end of day AoE**, according to the official organizer email. Use a parallel path: **dataset publication and Track 2a organizer clarification must not be delayed by GPU retries.** If 40-step fine-tuning fails, the benchmark-only submission may still be viable if the organizers confirm eligibility. If the optimizer run succeeds, real adapter evaluation becomes the next scientific step, not immediate evidence of improvement.
