# PolicyProof AI — Lexsi Labs Hackathon 2026

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
