# PolicyProof AI — Lexsi Labs Hackathon 2026

## Current project milestones (updated November 3, 2026)

| Workstream | Actual status |
| --- | --- |
| Baseline inference | **Completed:** 48 genuine Tiny Aya outputs on Colab T4; honest limitations documented |
| CuratorKIT | **Completed:** 78/78 synthetic examples, three languages and audited archive checksums |
| AlignTune training | **Not completed:** 4-bit LoRA model/config loaded but original trainer hit BF16 GradScaler |
| Direct PyTorch fallback | **Fix committed:** `tokenizers.Encoding` conversion and full preflight; actual GPU retry pending |
| AuditKIT | **Offline replay script ready:** baseline predictions can be evaluated without GPU; execution pending |
| Candidate final holdout | **Draft frozen in GitHub:** 60 independent multilingual cases; reviewer approval pending |
| Submission | **Draft created:** [write-up](docs/LEXSI_SUBMISSION_DRAFT.md); trained adapter and final metrics still missing |

**Fresh evaluation:** [60-case candidate](data/final_holdout_v1.jsonl) · [integrity checker](scripts/validate_final_holdout.py). Do not train on or tune against this set.  
**AuditKIT:** [real-output replay runner](scripts/auditkit_replay.py), based on the official `ak.evaluate(model="precomputed")` interface; no fabricated model generations.  
**Submission tracker:** [readiness and risk register](docs/SUBMISSION_READINESS.md).

## GPU quota reached — free Kaggle alternative (November 3, 2026)

Google Colab currently returned **"Cannot connect to GPU backend"** for the user. The original **78-row CuratorKIT ZIP restored successfully** on CPU, but no new Tiny Aya model has been loaded or trained since the Colab quota restriction. Google does **not** publish a guaranteed GPU quota reset time.

**Kaggle route:** [open the Kaggle GPU fallback notebook](notebooks/04_kaggle_gpu_fallback.ipynb). In Kaggle Settings, enable Internet and use GPU T4 x2 if offered (alternatively P100); add the original CuratorKIT training ZIP as a private input; configure an approved read-only HF token in Kaggle Secrets; then run the notebook in order. It prepares the quantized model and tokenizes all 78 examples, with *actual training disabled* until explicitly opted in. Hardware availability and project-specific success on Kaggle have **not been verified**.

**No GPU needed:** `python -m scripts.prepare`, `python -m scripts.validate_final_holdout`, and `python -m pytest -q` perform fixture generation and structural tests on CPU. They do **not** prove model improvement.

**Do not retry the old `recover_train_save` cell.** In the user's previous run, AlignTune/TRL still encountered BF16/FP16 precision conflicts. The documented direct PyTorch fallback was built to avoid those training-loop issues, but has not yet produced a successfully verified checkpoint.

## Last-resort T4 fallback: plain PyTorch optimizer on the existing LoRA adapter

**Why this exists:** In the exact installed TRL **1.7.1** source, `SFTTrainer` unconditionally casts trainable parameters to **BF16** when the model is 4-bit quantized; coupled with FP16 `GradScaler`, it caused the reported `_amp_foreach_non_finite_check_and_unscale_cuda` error. Attempts to alter AlignTune's factory flags alone did not resolve the second trainer-level behavior.

A [guarded Torch fallback](scripts/t4_torch_fallback.py) avoids the TRL/AlignTune **trainer loop** entirely while reusing the **already-loaded AlignTune PEFT/4-bit model**, tokenizer, and validated **CuratorKIT 78-row training export**. It masks the user/policy prompt from the loss, uses FP16 autocast with FP32 LoRA master parameters and **no GradScaler**, performs at most ten optimizer steps, and saves/checks adapter hashes. It refuses to overwrite an existing training receipt. This fallback does **not** count as demonstrated AlignTune trainer execution, nor is it evidence of improved safety. **November 3 tokenizer fix:** Hugging Face `tokenizers.Encoding` objects are now normalized to plain integer IDs; a CPU-side preflight checks all 78 rows first. The exact initial zero-step TypeError report is preserved if the corrected code is retried once.

In the same live Colab T4 kernel, after `git pull --ff-only`:

```python
from scripts.t4_torch_fallback import train_existing_t4
from google.colab import files
archive = train_existing_t4(trainer, model_sha=globals().get("MODEL_SHA"))
files.download(str(archive))
```

A successful run prints ten numbered optimizer steps and makes `policyproof_t4_torch_fallback.zip` available in the runtime's working directory. **This GPU step has not been executed or verified here**. Do not run repeatedly: the fixed runner permits at most one targeted retry when the previous receipt confirms the known zero-step `tokenizers.Encoding` TypeError. All other prior failures require inspection.

## Stage 3: controlled AlignTune QLoRA compatibility test

**[Open Stage 3 T4 training notebook in Colab](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/03_aligntune_qlora_t4.ipynb)** — starts with GPU check and [CuratorKIT archive verification](scripts/verify_curator_archive.py); training and heavyweight dependencies are OFF by default. Requires separately confirming package installation and trainer compatibility. **No successful training or adapter claimed yet.**

**[Verified CuratorKIT run: 78 synthetic examples](docs/CURATION_2026-11-03.md)** — contains archival hashes, version, provenance caveats, and the research risks to resolve before judging.

## Stage 2: CuratorKIT curation in Google Colab

**[Open the CuratorKIT data curation notebook in Colab](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/02_curatorkit_training_data.ipynb)** — use a fresh CPU runtime. This prepares *training data only*, verifies manifests and exports a ZIP for inspection. **No training or GPU needed.**

## Start the real Tiny Aya baseline in Colab

**[Open the secure T4 baseline notebook in Google Colab](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/01_tiny_aya_baseline_t4.ipynb)** — runs GPU preflight, private interactive Hugging Face login, dataset integrity checks, and unchanged-model inference. Never commit or paste access tokens. This notebook does not train a model.


**Track 1 · Fine-tuning & Data Curation**. Reproducible research into multilingual, evidence-grounded policy answering, citation correctness and abstention.

Status: active research prototype; **no actual GPU model training, validated improvements or final hackathon submission claimed**. All policies and evaluation examples are fictional, never actual bank information.

## Quick start

```bash
python -m scripts.prepare
python -m pip install pytest
python -m pytest -q
python -m scripts.audit_dataset --data-dir data --output artifacts/audit.json
python -m scripts.evaluate --split unseen-policy --engine mock-abstain --output artifacts/mock.json
python -m scripts.train # safe dry run; --execute required for GPU
```

The fixture builder creates 78 training examples, 42 paraphrase tests, and 48 unseen-policy tests (including 3 prompt-injection probes). Synthetic question templates in the generated fixture are *not* human-reviewed; stronger real evaluation must be added before claims. The unseen-policy split uses distinct fictional documents and validates ID separation.

## Scientific protocol

1. Prepare synthetic, versioned data with policy-level split isolation.
2. Curate with Lexsi CuratorKIT and save its provenance report.
3. Run base-model evaluation for CohereLabs/tiny-aya-global on unmodified tests.
4. Run AlignTune supervised fine-tuning on an approved GPU with the accepted HF model license.
5. Evaluate with AuditKIT and deterministic citation/abstention scorers; compare real model results via paired bootstrap intervals.
6. Publish data cards, model checkpoints, raw predictions, configs, limitations and a research write-up.

## Integrity and caveats

Mock evaluation checks software only, not real model quality. Direct external prediction imports are unverified and cannot be used as confirmed model improvement. Never commit Hugging Face tokens or private banking data. Review the Lexsi library licenses before use outside research. Full experimental training remains blocked on GPU and Hugging Face access.

Official resources: https://hackathon.lexsi.ai · https://github.com/Lexsi-Labs/CuratorKIT · https://github.com/Lexsi-Labs/aligntune · https://github.com/Lexsi-Labs/AuditKIT

## Inspect and review the experiment

Install the optional dashboard dependency and start the read-only research observer:

```bash
python -m pip install "streamlit>=1.40"
streamlit run research_dashboard.py
```

The dashboard shows real source fixtures and recorded outputs and prominently distinguishes mock evaluation runs from genuine model inference. It does not perform inference itself.

See [Research Protocol](docs/RESEARCH_PROTOCOL.md) and [Submission Readiness](docs/SUBMISSION_READINESS.md). The dataset integrity guard now rejects source-policy leakage, malformed labels, duplicated sample IDs, and stale checksums. Run it before using a dataset to train.

**Current local verification:** The sister working copy passed 15 software tests; the GitHub Actions run status has not been independently confirmed. Training and official Lexsi package integration remain pending.

## Resource preflight

Run `python -m scripts.preflight` to inspect Python, installed ML packages, CUDA device availability and whether an HF environment token exists (never its value). The optional `--strict` flag exits nonzero when prerequisites are missing. A passing preflight does **not** verify Hugging Face gated-license acceptance or compute credit balance.
