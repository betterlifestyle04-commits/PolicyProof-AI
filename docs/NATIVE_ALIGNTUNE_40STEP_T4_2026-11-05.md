# Native 40-step AlignTune QLoRA route — preflight and GPU evidence (November 5, 2026)

**Status: IMPLEMENTED, NOT EXECUTED on a real T4 GPU in repository CI.**
The existing 40-step model adaptation was run with a **direct PyTorch fallback over an AlignTune-loaded PEFT model**, which must not be presented as an official AlignTune optimizer result. The existing successful saved format/citation metrics do not prove supported answer semantics.

This new path is an **independent fresh-run experiment** using the official AlignTune TRL trainer. It cannot retroactively convert the earlier manual result into an official run.

## November 9 — safer owner recovery after a native 40-step run

[Stage 11 guided native AlignTune T4 Colab notebook](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/11_native_aligntune_40step_t4.ipynb) now **separates the optimizer action from private backup/browser download**. All action switches still default OFF.

- The optimizer cell first checks for an **existing original** `/content/policyproof_align_native40.zip`. When a complete ZIP exists, it checks that original ZIP **without calling `recover_train_save` or `trainer.train()` again**. If the native working directory already exists but the complete ZIP is missing, the cell blocks a second optimizer call and directs the owner to the existing read-only interruption audit.
- The new **separate Stage 6 recovery cell** defaults to `SAVE_PRIVATE_DRIVE_BACKUP=False` and `DOWNLOAD_NATIVE40_ZIP=False`. Enable only the requested action. It can be rerun after a failed browser download **without retraining**.
- For an authorized private backup, Stage 6 mounts the owner's private Google Drive using the secure Google prompt and invokes `scripts.preserve_native40_owner_archive.preserve_native40_archive`. The tool CPU-validates the exact original 40-step AlignTune adapter ZIP, copies it once to a hash-named private file, reads the copy back, revalidates adapter integrity, and refuses to replace altered, partial or symlinked backups. Google Drive may not immediately sync writes; keep a second original and the unedited Colab log.
- A file hash and an internal `run_info.json` receipt still **do not independently prove GPU training took place**, that 40 optimizer steps executed, that a specific model SHA was loaded, or that model safety improved. These remain separate scientific proof requirements.

**If a download dialog fails:** rerun **only Stage 6**, turn on `DOWNLOAD_NATIVE40_ZIP=True`, and preserve the ZIP. **If Colab disconnects after training without a verified backup, do not restart optimizer as a recovery attempt.** Inspect the original evidence (if any) with the Stage 8 read-only diagnostic, and preserve interrupted local receipts. Do not claim a successful native 40-step run until its genuine original ZIP and log have been checked.

## Official dependency and workflow

AlignTune's [SFT guide](https://aligntune.lexsi.ai/user-guide/sft/) documents `create_sft_trainer(..., max_steps=..., backend="trl")`, `trainer.train()`, and `trainer.save_model()`. PolicyProof uses a previously inspected compatibility environment (AlignTune **0.1.28**, TRL **1.7.1**, BitsAndBytes **0.50.2**) to work around the prior BF16 GradScaler mismatch with T4 LoRA training.

**Open a fresh T4 Colab session:**

[Standalone Colab Stage 11 — native AlignTune 40-step QLoRA](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/11_native_aligntune_40step_t4.ipynb)

1. Select **Runtime → T4 GPU**, run notebook step 1, clone the current pinned GitHub commit.
2. Install and verify the pinned environment in step 2. If Colab requires a runtime restart after package changes, restart and run notebook cells from the top.
3. In step 3 upload **only** the original `policyproof_curatorkit_train_v1.zip` if prompted; the archive and all 78 training-only Alpaca rows must match SHA256 `4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c`. Sign into Hugging Face through its login prompt if needed, never by inserting tokens into a code cell.
4. Confirm 252 4-bit layers, 288 trainable LoRA tensors, **3,833,856** trainable LoRA parameters and zero existing optimizer steps. Inspect the observed **loaded-model** revision independently from the Hub's remote SHA; missing observed hash remains flagged *UNVERIFIED*.
5. Leave `RUN_NATIVE_TRAINING = False` while inspecting. Change it to **True** only when ready to spend GPU time and run step 5 **once**. The script validates the final **TRL SFTConfig**, with **`max_steps=40`, `fp16=False`, `bf16=False`**, and the separate output directory before calling the native `wrapper.train()`.
6. If and only if the actual TRL trainer reaches exactly 40 completed optimizer steps, the notebook saves a LoRA adapter and metadata into `/content/policyproof_align_native40.zip`; it checks archive integrity using the CPU-only verifier. **Use the separate Stage 6 recovery cell** to download the original ZIP and, if explicitly approved, copy a checksum-verified backup to private Drive without retriggering training. Keep the unchanged original and Colab stdout/stderr logs. **Do not restart/resume the model or combine this with 10-step smoke or PyTorch fallback training in the same session.**

The package uses `artifacts/align_qlora_native40` internally; the existing 10-step smoke uses `artifacts/align_qlora_smoke`. Both are non-overwriting and refuse stale output/checkpoints or preexisting ZIP files.

## Verified dataset handoff and small adapter archive (November 5)

Reading the **actual AlignTune 0.1.28 TRL SFT trainer** revealed a second dataset cache: after setup, `wrapper.train_dataset` contains the 78 verified training rows but `wrapper.dataset` is still `None`. The real `TRLSFTTrainer.train()` then calls `setup_data()` again if `wrapper.dataset is None`, potentially reloading or reprocessing an already-verified dataset. We now bind the **same Python dataset object** into `wrapper.dataset` immediately before `wrapper.train()`, reject a conflicting existing cache or any evaluation dataset, and verify the object identity again afterward. This blocks the redundant second DataManager load under the pinned implementation. The sixth CI job reads the REAL installed upstream trainer source and fails if its branch changes. These are still local source and runtime-object checks, **not cryptographic proof of what each token contributed to loss**.

AlignTune's own trainer also saves intermediate checkpoint, tokenizer and metadata files into its output directory. Previously our ZIP recursively copied everything under that directory, risking an oversized download or rejected/unsafe archive. The original files **remain in the local Colab output directory**, but the portable release ZIP now contains **exactly** `adapter/adapter_model.safetensors`, `adapter/adapter_config.json`, and `run_info.json`. It rejects symlinks, missing safetensors, unusually large individual files or archives that would exceed the CPU verifier bounds. Never upload intermediate optimizer-state checkpoints or machine-local private logs with the adapter.

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

## Transactional native adapter release (November 5, 2026)

Actual AlignTune 0.1.28 `TRLSFTTrainer.train()` returns its independent `TrainOutput.steps` in a dictionary. The native 10/40-step launcher now requires **the returned integer `steps` to match both `trainer.state.global_step` and the requested optimizer budget**. The ZIP receipt records the returned value; the CPU-only archive verifier checks it for native40. This is a consistency check between *self-reported* local counters, not independent proof of GPU execution.

Training completion and safe archive publication are now separate states. After verified training the local `run_info.json` is written with `status=packaging` through a temporary file and atomic replace. The ZIP is assembled under a private `.partial` filename, contains only the original safe PEFT safetensors, adapter config and the **completed** JSON receipt, and is checked for exact members and CRC before publication. A same-filesystem atomic **hard link** publishes the finished ZIP under its final name **only if that name is still unused**. Any interrupted ZIP operation, malformed CRC or destination race fails without overwriting another archive or exposing a partial ZIP under the intended filename; cleanable temporary files are removed on ordinary exceptions. The local receipt becomes `completed` only after archive publication succeeds. A failure is recorded `failed`, even if the optimizer had already completed 40 steps. On a hard process kill, a temporary `.partial` file may remain, but it is never an approved final artifact.

The original on-disk AlignTune checkpoints and logs remain in the Colab working directory. The portable ZIP has exactly three files and uses the existing size/format bounds. All logs and ZIP metadata remain **self-reported** until an actual independent GPU/semantic verification.

## Interrupted Colab evidence reconciliation (November 5, 2026)

Native 10/40-step training now saves an **`in_progress` receipt before its first optimizer step**. A Colab disconnect or `KeyboardInterrupt` therefore leaves evidence of an attempted but **unproven** training run, rather than an absent receipt that might later be confused with a successful run. A normal failure is `failed`; during archive creation the local receipt is `packaging`. A published ZIP followed by a failed local receipt update is **`archive_published_local_receipt_failed`**, not proof that the optimizer itself failed. An additional receipt-write failure can leave the earlier `packaging` record intact; the ZIP is not deleted if already sealed.

To triage **without restarting GPU training**, run the new **Stage 11 Step 7** diagnostic cell, or use:

```bash
python -m scripts.reconcile_native_training_evidence \
  --directory artifacts/align_qlora_native40 \
  --archive /content/policyproof_align_native40.zip
```

Omit `--archive` only when the ZIP is genuinely absent. The CPU-only checker verifies all archived adapter structure/hashes via the existing safe verifier and compares the original ZIP receipt, local receipt, and adapter content hashes. It never modifies evidence or fabricates a missing training result. A fully matching report is `MATCHED_SELF_REPORTED_COMPLETION` **only when** the verified ZIP also meets the Track 1 candidate checks (native AlignTune backend, expected 288-tensor Tiny Aya LoRA geometry, and pinned base revision with the observed loaded-model commit for native40). Matching receipts with an unverified loaded revision are `MATCHED_UNVERIFIED_LOADED_BASE_REVISION`; matching manual PyTorch fallback receipts are `MATCHED_NON_ALIGNTUNE_BACKEND`; neither is a Track 1 candidate. The report exposes `consistent_track1_artifact_candidate` and `archive_loaded_base_revision_verified` separately from mere hash agreement. **Every status, including a candidate, requires owner review of genuine GPU execution and independent response quality**; no CPU check authenticates those. `ARCHIVE_VALID_LOCAL_RECEIPT_UNRECONCILED` also requires investigation even when the ZIP is internally sound. `INTERRUPTED_OR_RUNNING_NO_ZIP`, `FAILED_NO_ZIP`, contradictory fields and missing adapter files all block success claims. Preserve the original archive, receipt, and complete Colab logs. **Never rerun the 40-step optimizer simply to make a receipt pass.**

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

## November 10 — original 78-example training ZIP now produced by verified CI

The Stage 11 native40 Colab notebook requires the **original** `policyproof_curatorkit_train_v1.zip`. To avoid asking the owner to find an old attachment or accidentally use the 72-case challenge, the CI job `Real CuratorKIT then AuditKIT safety benchmark (CPU)` now builds it from fresh **official CuratorKIT 0.1.22** exports, verifies both Alpaca and ShareGPT views, validates the unchanged **78-row training-data SHA256 `4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c`**, and independently validates the sealed ZIP with `scripts.verify_curator_archive.check_zip`.

**Download:** On the latest successful [main GitHub Actions run](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/workflows/ci.yml), find **Artifacts → `policyproof-native-align-original-78-CuratorKIT-INPUT-NOT-MODEL`**. GitHub downloads an *outer Actions artifact ZIP*; extract the single `policyproof_curatorkit_train_v1.zip` from it. Upload **that inner ZIP**, unchanged, when Stage 11's **Step 3** secure Colab upload prompt requests it.

The inner ZIP contains only the original CuratorKIT source-derived training export files; no adapter, model weights, GPU results, reviewer notes, 48-case evaluation, 60-case holdout or 72-case challenge. It is **training input**, not a trained model or research result. Weights still need legitimate Tiny Aya access and license approval in the owner's fresh GPU runtime. The GitHub runner never downloads a gated model, accesses owner HF tokens, or performs optimizer steps.

For a reproducible local CPU-only rebuild from a checked-out project with CuratorKIT 0.1.22 installed:

```bash
python -m scripts.prepare
python -m scripts.run_curator --output artifacts/curator_train_dual_export
python -m scripts.build_native_train_owner_zip
python -m scripts.verify_curator_archive artifacts/policyproof_curatorkit_train_v1.zip
```

**Never train on the frozen 72-case adversarial evaluation, final-60 holdout, or original 48-case evaluation examples.** If Stage 11 has already started training in its current Colab runtime, do not rerun the optimizer to replace a ZIP; use its read-only interruption audit.

