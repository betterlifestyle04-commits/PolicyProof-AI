# Evaluate the genuine 40-step Tiny Aya checkpoint — step-by-step

**Current status — November 4, 2026:** We have a locally uploaded, structurally valid PEFT adapter with 40 reported optimizer updates, but **no actual paired inference evidence or established safety improvement**. This protocol is built for an authorized Google Colab T4 GPU and uses the already-inspected **48-case synthetic development set**, not the unreviewed 60-case final candidate or 72 unscored challenges.

## What to keep

Keep a backup of the original exact `policyproof_t4_torch_extended.zip` you uploaded. Its verified archive SHA-256 is:

```text
f76750c31e26db8603dab7f0eaef481929403ff96fd3ec6aaea1361a842fba3f
```

The adapter weights are internally consistent, with 288 LoRA tensors and 3,833,856 trainable elements. The 40-step receipt claims model revision `af89d219b53ed9b13b8a4645f9c8028973510324`, but the **exact revision of weights loaded during training could not be independently observed**. It is therefore incorrect to claim a perfectly pinned train-time baseline or proven generalization; the paired inference will deliberately load the same declared revision freshly for both new passes.

The old `artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json` has `model_revision: null`. It is **not an acceptable paired control** for this checkpoint. The new runner generates an entirely fresh baseline.

## 1. Start the evaluation notebook (NOT the training notebook)

Open the direct [48-case evaluation notebook in Google Colab](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/06_dev48_pair_eval_t4.ipynb).

In Colab use **Runtime → Change runtime type → T4 GPU**. Start an independent notebook/runtime so stale training objects cannot interfere with the original-model inference. Using the same existing model with LoRA enabled as the base is an invalid experimental comparison. Training remains OFF throughout this notebook.

## 2. Run cells 1–2: source and dependencies

Run the repo setup cell, which clones/pulls the current main source into `/content/PolicyProof-AI`. It checks for CUDA without running training. The next cell installs the minimal inference stack and **real AuditKIT 1.1.3**. It does not call any training optimizer.

If Colab demands a runtime restart after installing packages, restart and rerun the setup cells. Avoid accidentally clicking `Run all` after manually changing any execution toggles.

## 3. Run cell 3: upload the SAME trained ZIP

Upload exactly `policyproof_t4_torch_extended.zip`, with the digest shown above. Do **not** upload the original CuratorKIT training ZIP here, because the model is already trained; these are distinct archives.

The notebook refuses source mismatch, an unsupported backend, bad LoRA tensor geometry, changed adapter file hashes, incorrect optimizer-step claims, or an altered ZIP. Avoid renaming or repacking the archive to make a hash check pass.

## 4. Run cell 4: generate the fixed development rows and read-only preflight

The notebook runs `python -m scripts.prepare`, reconstructing the original 48 synthetic development prompts from public fictional policies, with exact file SHA256:

```text
d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c
```

It should print:

```text
DEV48_GPU_EVALUATION_PREPARED_NOT_EXECUTED
cases: 48
model_inference_performed: false
model_improvement_verified: false
```

This is a **CPU-only safety check** and is not a model result.

## 5. Run cell 5: authorize access to the gated base model

Authenticate using Hugging Face's supported notebook login. Never paste a token in a source cell, public Git commit or ChatGPT message. Make sure your HF account has accepted [CohereLabs/tiny-aya-global's model license and access conditions](https://huggingface.co/CohereLabs/tiny-aya-global). The notebook checks the pinned base commit. Original upstream Tiny Aya weights are gated and must **not** be redistributed with the checkpoint.

## 6. Run cell 6: authorize actual paired GPU inference

**Only after the previous five cells succeed**, go to **"6. Explicitly authorize TWO 48-case GPU inference passes"**.

Change just:

```python
RUN_DEV48_GPU_EVAL = True
RESUME_SAVED_BASE = False
```

Run that single cell. The script will:

1. Load original 4-bit Tiny Aya revision `af89d219b53ed9b13b8a4645f9c8028973510324` and generate 48 fresh responses with deterministic decoding; save `artifacts/dev48_paired_base.json` as soon as this pass is complete.
2. Release original model memory. Validate and safely stage the original ZIP, load a new copy of the same base revision with the adapter, and generate the exact same 48 prompts; save `artifacts/dev48_paired_tuned.json`.
3. Evaluate the original raw outputs in **native AuditKIT** and write `artifacts/dev48_paired_auditkit.json`. Preserve per-case paired exact/format/citation/abstention changes, including regressions.
4. Create a **separate** `policyproof_dev48_paired_gpu_results.zip` with the three original reports and a checksum manifest; Colab will offer it as a download. This results package does NOT contain original model weights or your PEFT checkpoint.

All outputs are **write-once**. It cannot overwrite an existing report, the benchmark source, or the trained ZIP. Do not confuse the two 48-case inference passes with re-training; no optimizer steps run here.

## 7. If the GPU pass errors

**If the base result was saved but tuned inference fails:** inspect the exception, correct the underlying issue and rerun *only* cell 6 with `RESUME_SAVED_BASE = True`. This rechecks original inputs, model revision, generation config and all baseline raw predictions/scores. Never manually change saved predictions or delete the base to fake success.

**If both base and tuned results exist but AuditKIT fails:** inspect the error, then use the optional *CPU-only recovery* cell with `RUN_CPU_AUDIT_RECOVERY = True`. This rescored both genuine saved reports without repeating the expensive model calls.

**If the runtime disconnects:** download any completed JSON reports before the VM is deleted, if possible. A partial 48-case generation is not a completed report.

**If the model is gated, GPU quota is exhausted or CUDA memory is insufficient:** stop and preserve the error; don't invent results, and do not spend paid compute without approving it.

## 8. Upload the RESULT ZIP for independent analysis

Upload **`policyproof_dev48_paired_gpu_results.zip`** back into this chat. Retain **`policyproof_t4_torch_extended.zip`** separately. We can then:

- Recalculate all per-case metrics and inspect where the model improved or regressed.
- Validate AuditKIT fingerprints and exact SHA256 of every result file.
- Compare accuracy, source IDs, schema validity and abstention by English/Norwegian/Arabic plus prompt-injection cases, and flag paraphrases that exact-string scores may penalize.
- Decide whether another training run is justified or whether the existing checkpoint is preferable.
- Prepare the final, independent 60-case review process *without leaking final labels into model tuning*.

## Explicit scientific limitations

The earlier saved 48-case results are not a correctly pinned comparison for this new adapter. The 48-case prompts were seen during development and thus are **not a new blind holdout**. The internal 12-row training-source validation loss is **not** an external independent safety metric. Correct source IDs are **not** semantic citation entailment. Exact supported answers may be penalized for legitimate paraphrases; human review is needed for meaning. The new runner's predictions and AuditKIT comparison are meaningful diagnostic evidence but **cannot establish broad real-world model safety**.

This work is independent of the required owner-approved Hugging Face Track 2a dataset publication and official Lexsi submission due November 14, 2026 AoE. Do not delay those tasks for further iterative GPU experiments.
