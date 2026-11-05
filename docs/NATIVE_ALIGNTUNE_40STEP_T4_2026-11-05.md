# Native 40-step AlignTune QLoRA route — preflight and GPU evidence (November 5, 2026)

**Status: IMPLEMENTED, NOT EXECUTED on a real T4 GPU in repository CI.**
The existing 40-step model adaptation was run with a **direct PyTorch fallback over an AlignTune-loaded PEFT model**, which must not be presented as an official AlignTune optimizer result. The existing successful saved format/citation metrics do not prove supported answer semantics.

This new path is an **independent fresh-run experiment** using the official AlignTune TRL trainer. It cannot retroactively convert the earlier manual result into an official run.

## Official dependency and workflow

AlignTune's [SFT guide](https://aligntune.lexsi.ai/user-guide/sft/) documents `create_sft_trainer(..., max_steps=..., backend="trl")`, `trainer.train()`, and `trainer.save_model()`. PolicyProof uses a previously inspected compatibility environment (AlignTune **0.1.28**, TRL **1.7.1**, BitsAndBytes **0.50.2**) to work around the prior BF16 GradScaler mismatch with T4 LoRA training.

**Open a fresh T4 Colab session:**

[Standalone Colab Stage 11 — native AlignTune 40-step QLoRA](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/11_native_aligntune_40step_t4.ipynb)

1. Select **Runtime → T4 GPU**, run notebook step 1, clone the current pinned GitHub commit.
2. Install and verify the pinned environment in step 2. If Colab requires a runtime restart after package changes, restart and run notebook cells from the top.
3. In step 3 upload **only** the original `policyproof_curatorkit_train_v1.zip` if prompted; the archive and all 78 training-only Alpaca rows must match SHA256 `4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c`. Sign into Hugging Face through its login prompt if needed, never by inserting tokens into a code cell.
4. Confirm 252 4-bit layers, 288 trainable LoRA tensors, **3,833,856** trainable LoRA parameters and zero existing optimizer steps. Inspect the observed **loaded-model** revision independently from the Hub's remote SHA; missing observed hash remains flagged *UNVERIFIED*.
5. Leave `RUN_NATIVE_TRAINING = False` while inspecting. Change it to **True** only when ready to spend GPU time and run step 5 **once**. The script validates the final **TRL SFTConfig**, with **`max_steps=40`, `fp16=False`, `bf16=False`**, and the separate output directory before calling the native `wrapper.train()`.
6. If and only if the actual TRL trainer reaches exactly 40 completed optimizer steps, the notebook saves a LoRA adapter and metadata into `/content/policyproof_align_native40.zip`; it then checks archive integrity using the CPU-only verifier and downloads it. Keep the original ZIP and unedited Colab stdout/stderr logs. **Do not restart/resume the model or combine this with 10-step smoke or PyTorch fallback training in the same session.**

The package uses `artifacts/align_qlora_native40` internally; the existing 10-step smoke uses `artifacts/align_qlora_smoke`. Both are non-overwriting and refuse stale output/checkpoints or preexisting ZIP files.

## What the wrapper actually checks

- Calls `scripts.check_align_config.check_train_data()` on the real training file then compares the **exact immutable bytes SHA** before a GPU optimizer is possible.
- Verifies an approved QLoRA base with no existing optimizer steps and the exact original LoRA parameter count.
- Synchronizes both `config.train.max_steps` and AlignTune's later-applied `config.train.extra_params["max_steps"]` to **40**. This closes the accidental 10-step fallback to default `max_steps=10` when a caller requests 40.
- Repairs only trainable LoRA adapters to FP32, retains the frozen base quantized at 4-bit and forbids FP16/BF16 automatic mixed precision/GradScaler on the T4. Asserts the final built TRL `SFTConfig` has the exact budget, no AMP and the intended distinct output directory.
- Because the 78-row experiment has **no evaluation dataset**, explicitly synchronizes 'load_best_model_at_end=False' and 'eval_strategy=no'. A best-model checkpoint requires real eval measurements; the code now aborts if AlignTune re-enables an incompatible best-checkpoint setting.

- Calls `AlignTune wrapper.train()` **once** and checks the actual resulting `wrapper.trainer.state.global_step` equals 40. The public CPU tests replace this with a fake; they do **not** show real GPU execution.
- Calls `wrapper.save_model()`, expects PEFT adapter weights and config, hashes them, records pinned training SHA, Git commit, package versions, output GPU identity and session-reported native backend, and packages a create-once ZIP.
- Any training exception leaves a `run_info.json` marked **failed** rather than a completed adapter ZIP. Do not suppress failure output or auto-claim training.
- The archive verifier checks shape, hashes and 288-tensor/3,833,856-element geometry, but it **never** claims independent GPU execution or model improvement. For native 40-step artifacts, a remote Hub model revision by itself no longer qualifies as a pinned *loaded* model; an observed matching loaded model config commit is required for the local candidate flag.

## Real pinned upstream source compatibility check

The [new sixth CI job](REAL_ALIGN_SDK_SOURCE_CONTRACT_2026-11-05.md) installs the exact **AlignTune 0.1.28 and TRL 1.7.1 published distributions** without ML dependencies and examines their original factory/SFTConfig/config-extractor AST. This strengthens the mocked late-configuration tests and fails if the pinned upstream contract drifts. **No real SFTConfig is instantiated in this job, and no actual T4 optimizer step has been run.**

## Tests and reproducibility without GPU

```bash
python -m pytest -q tests/test_native_aligntune_40_budget.py \
  tests/test_complete_t4_training.py tests/test_recover_t4_training.py \
  tests/test_verify_training_archive.py tests/test_notebook_validation.py
python -m scripts.validate_notebooks
```

The no-AMP/preflight tests deliberately simulate AlignTune's `train.extra_params` precedence and an SFTConfig builder with no live GPU, trainer, model weights or optimizer call. The artifact tests use toy adapter bytes and synthetic receipts and **must not** appear in published training claims.

## After a genuine native adapter

Run independent 48-case base-vs-trained evaluation **without using it to train**, then the separate human-blinded semantic review. The 60-case candidate final set remains held back until independent gold-label review is completed. The final paired inference, HF collection publication, license check, organizer approval and official submission still require separate evidence. Merely completing 40 steps or achieving a schema metric is not proof that PolicyProof AI learned to reason about the fictional policy text accurately.

**Remaining blocker:** We cannot execute the remote Colab T4 from GitHub Actions and therefore this new path is still **untested end-to-end on GPU**. If it fails on Colab, retain the complete error and failure receipt, and diagnose the real trainer rather than relabeling the old manual fallback. This research pipeline must not be used as a production financial-answer authority.
