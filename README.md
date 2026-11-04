# PolicyProof AI — Lexsi Labs Hackathon 2026

## Public research dashboard

**[Open the live PolicyProof AI Observatory](https://policyproof-ai-observatory.onrender.com)** — hosted on a free Render service with nine tested pages. Build and start verified by Render (service `srv-dbku0dek1f9s73e0cnt0`, initial deploy `dep-dbku0duk1f9s73e0cp80`). It uses preserved base-model outputs and CPU-generated evaluation artifacts, **not** a tuned model or real-time AI inference. Free-tier services may sleep and take extra time to wake.

**[Judge walkthrough](docs/JUDGE_QUICKSTART.md)** · **[Track 2a research report](docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md)** · **[Guard experiment](docs/INFERENCE_GUARD_RESULTS_2026-11-03.md)** · **[Track 1 submission status](docs/SUBMISSION_READINESS.md)**

## Verified browser usability — November 3, 2026

A complete **headless Chromium** session navigated the live-style local Streamlit site, loaded the Track 2a safety benchmark, submitted a synthetic forged citation appearing inside an untrusted note and observed the guard reject it, then opened the judge submission gate and confirmed NOT READY. This is a real browser interaction, **not** a cloud-hosted third-party visitor test or a live model evaluation.

- [Real browser workflow and execution logs](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41706273198)
- [Browser navigation and guard test source](qa/browser_smoke.py)
- Screenshots are retained as downloadable GitHub Actions artifacts.

## Future 72-case evaluation runner (NOT executed on a real model yet)

The [72-case measurement protocol](docs/TRACK2A_72_EVALUATION_PROTOCOL.md) and [strict AuditKIT scorer](scripts/score_track2a_72.py) are ready for **actual** Tiny Aya inference responses once GPU compute is available. The runner requires 72 unique raw model outputs, the exact frozen benchmark SHA-256, a pinned 40-character model revision, and a generation record. It fails closed on missing/synthetic/incorrectly labeled reports, and explicitly separates matched control-versus-attack outcomes. **No new-model result or 72-case performance statistic is claimed at present.**

## Dataset publishing handoff (CPU only; not yet published)

The **48 evaluated + 72 unscored** Track 2a dataset package passed offline ZIP and SHA-256 integrity verification. The [owner-only CPU Colab publisher](notebooks/05_publish_track2a_hf_cpu_only.ipynb) and [step-by-step Hugging Face guide](docs/TRACK2A_HF_PUBLICATION_GUIDE.md) are ready, with publication **OFF by default**. The publisher requires explicit legal license/rights approval, a private HF_TOKEN and creates a real public dataset repo and dataset collection only after owner opt-in. It verifies anonymous access afterwards. It does **not** publish model weights, a fake adapter, the required separate model collection or the final Lexsi submission.

## Automated QA — November 3, 2026

**Previously verified CPU QA:** [135 passed, 2 skipped](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41706273198) on the standard runner, plus the [dedicated PyTorch CPU safety job's 7 passing tests](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41635240908). The expanded runner now validates and replays the 48 actual saved Tiny Aya baseline responses with AuditKIT 1.1.3, and measures a real inference-time guard (not model training). This covers CPU fixtures, training safety guards, 60-case benchmark integrity, Colab/Kaggle notebook syntax, nine Streamlit dashboard pages and genuine AuditKIT precomputed evaluation with **synthetic oracle** outputs. The main runner skips two PyTorch-dependent test modules, but the dedicated CPU-only PyTorch job executed their seven tests successfully. **Public deployment and HTTP health checks verified; CUDA training not completed.**

[Read the complete verified QA report](docs/QA_VERIFICATION_2026-11-03.md). Baseline and CuratorKIT evidence are independently checked for consistency, but the trained adapter and model improvement remain outstanding.

## Current project milestones (updated November 3, 2026)

| Workstream | Actual status |
| --- | --- |
| Baseline inference | **Completed:** 48 genuine Tiny Aya outputs on Colab T4; honest limitations documented |
| CuratorKIT | **Completed:** 78/78 synthetic examples, three languages and audited archive checksums |
| AlignTune training | **Not completed:** 4-bit LoRA model/config loaded but original trainer hit BF16 GradScaler |
| Direct PyTorch fallback | **Fix committed:** `tokenizers.Encoding` conversion and full preflight; actual GPU retry pending |
| AuditKIT | **Executed:** 48 saved Tiny Aya base outputs successfully replayed with real AuditKIT 1.1.3; fingerprint ae297dd84f990566 |
| Track 2a safety benchmark | **Actual CuratorKIT → AuditKIT chain completed on 48 recorded Tiny Aya outputs**; additional **72 curated synthetic attack cases are UNSCORED**. [Research report](docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md) · [alternate abstract](docs/TRACK2A_SUBMISSION_ABSTRACT.md); official HF collections and submission pending |
| Candidate final holdout | **Draft frozen in GitHub:** 60 independent multilingual cases; reviewer approval pending |
| Submission | **Research article and HF cards drafted:** [publication draft](docs/HF_ARTICLE_DRAFT.md); trained adapter, paired model evaluation, HF public URLs and official submission still missing |

**Fresh evaluation:** [60-case candidate](data/final_holdout_v1.jsonl) · [integrity checker](scripts/validate_final_holdout.py). Do not train on or tune against this set.  
**AuditKIT:** [48 actual saved baseline predictions](artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json) were replayed successfully through real AuditKIT 1.1.3 on GitHub Actions, with no newly generated inference. [Guarded output results](docs/INFERENCE_GUARD_RESULTS_2026-11-03.md) are a separate deterministic *inference-time* experiment, not a trained adapter.  
**Submission tracker:** [Track 1 status](docs/SUBMISSION_READINESS.md) · [Track 2a separate gate](scripts/track2a_readiness.py). Neither track is officially submitted. The Track 2a gate does NOT require a fictitious fine-tuned model.

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
python -m scripts.train  # safe dry run; --execute intentionally DISABLED
python -m scripts.verify_evidence --baseline artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json --dataset data/unseen_policy_test.jsonl
python -m scripts.auditkit_replay --report artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json --dataset data/unseen_policy_test.jsonl
python -m scripts.inference_guard --baseline artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json --dataset data/unseen_policy_test.jsonl
python -m scripts.submission_gate  # must print NOT_READY before a legitimate checkpoint
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

**Current verification:** Official GitHub Actions successfully reproduced original saved-model baseline diagnostics, executed the AuditKIT integration, and ran the deterministic output guard. Separate PyTorch CPU tests passed. Real GPU training and a tuned-model comparison remain pending; see [QA report](docs/QA_VERIFICATION_2026-11-03.md).

## Resource preflight

Run `python -m scripts.preflight` to inspect Python, installed ML packages, CUDA device availability and whether an HF environment token exists (never its value). The optional `--strict` flag exits nonzero when prerequisites are missing. A passing preflight does **not** verify Hugging Face gated-license acceptance or compute credit balance.


## Independent final-holdout review packets (CPU only)

The dashboard's **Final benchmark review** page now prepares downloadable, independent reviewer CSVs for the 60-case fictional final candidate. Each CSV contains the proposed gold labels, original source evidence, three empty yes/no judgment fields per case, the dataset hash and a reviewer pseudonym, but **no model predictions**. Two or more reviewers can complete separate CSVs and reconcile them offline with `python -m scripts.review_holdout --summarize reviews/reviewer_A.csv reviews/reviewer_B.csv`. The validator detects missing, negative, inconsistent and duplicate judgments and changes to the frozen source/gold data. Even unanimous agreement **never** automatically generates a human sign-off or claims semantic accuracy. See [the independent-review protocol](docs/FINAL_REVIEW_WORKFLOW.md). The candidate currently remains **unreviewed**.

## New: verify any future fine-tuning adapter without a GPU

The public source now includes a **CPU-only checkpoint verifier**. After a genuine Colab/Kaggle training attempt produces an adapter ZIP, run:

```bash
python -m scripts.verify_training_archive --archive path/to/actual_checkpoint.zip
python -m scripts.verify_training_archive --archive path/to/actual_checkpoint.zip --strict-track1
```

It checks PEFT `safetensors` layout, LoRA A/B tensors, both file hashes, audited training dataset SHA-256, source commit, self-reported optimizer-step metadata, pinned base model revision and the declared training backend. A manual PyTorch fallback is **not** presented as completed AlignTune training; malformed dummy files no longer satisfy the submission gate. Even a passing check **does not prove genuine optimizer execution or a model-quality improvement**—those require separately verified GPU logs and paired model evaluations. Full instructions: [Training artifact verification](docs/TRAINING_ARTIFACT_VERIFICATION.md). Current project status remains **NOT READY**.


## Gated paired base-vs-adapter evaluation (new)

A future completed AlignTune run can now be evaluated against the same **60-case frozen final candidate** through [`scripts.frozen_pair_inference`](scripts/frozen_pair_inference.py). The **default command is CPU-only preflight** and reports missing human-review signoff and a valid adapter without downloading weights or starting model inference:

```bash
python -m scripts.frozen_pair_inference
```

Only after genuinely completed independent benchmark review and a hash-verified, 40-character-revision-pinned LoRA adapter are present can an operator explicitly authorize separate real GPU base and adapter predictions with identical prompts and generation parameters, followed by paired AuditKIT scoring. The runner never uses the already inspected 48-case development outputs as the 60-case baseline and does not auto-report improvement. **Neither this feature nor its CPU tests represent completed GPU inference or fine-tuning.** [Read the frozen evaluation runbook](docs/FROZEN_PAIRED_INFERENCE.md).


### Recovery if real GPU evaluation is interrupted

After a genuine, authorized 60-case paired evaluation begins, the original-base report is now **persisted immediately after a complete base pass**. If adapted-model inference or AuditKIT fails, the base results are preserved. An explicit `--resume-base` rerun can reuse those same outputs **only** after strict revalidation of frozen dataset bytes, pinned revision, generation settings, 60 unique case IDs, raw outputs, scores, and declared inference metadata. The validator rejects corrupt or mismatched evidence and refuses to overwrite any existing result. This capability reduces duplicate GPU work; it is **not evidence that real GPU inference has occurred**. [Recovery instructions](docs/RECOVERABLE_PAIRED_EVALUATION.md).


### New: recover CPU AuditKIT without repeating either model pass

The paired evaluator now preserves **both complete original-base and tuned-model raw outputs** before AuditKIT scores them. If scoring fails *after both GPU passes completed*, `python -m scripts.finish_saved_pair` runs a read-only preflight. The explicit `--finalize --confirm-saved-inference` route validates both saved evidence files, the 60-case dataset hash, common pinned model revision, adapter SHA, exact response coverage and recomputed metrics, then reruns native AuditKIT **on CPU without model inference**. It never overwrites prior evidence or claims to have independently verified original GPU execution. See [paired recovery instructions](docs/RECOVERABLE_PAIRED_EVALUATION.md). These paths are CPU-simulated regression tests only; real GPU training and final evaluation are still pending.

## New — real anonymous Hugging Face publication verification

The benchmark-publisher script already supports **explicit owner-approved** dataset release, but local status flags cannot prove a judge has access. The new [anonymous Hugging Face verification tool](scripts/verify_hf_collections.py) now rejects self-certified flags and checks the **actual** public collection and individual repository access with `token=False`, reads required dataset files at pinned commit revisions, validates their published SHA256SUMS hashes, checks the selected public license, and confirms the actual 48-base / 72-unscored research scope. It creates a timestamped, manifest-checksum-bound verification receipt only on an **explicit** remote call. No publication or credentials are involved:

```bash
python -m scripts.verify_hf_collections   # completely offline, read-only
# After real URLs exist and the owner approves anonymous checking:
python -m scripts.verify_hf_collections --verify-network
```

`scripts.track2a_readiness` now ignores manually written `external_access_verified` booleans and requires a matching anonymous receipt no older than 72 hours. The public Judge verification page shows this status. **The project currently has no verified public HF collection links, and this work does not publish any artifacts or resolve the benchmark-only model-collection ambiguity.** See [publication check runbook](docs/HF_ANONYMOUS_PUBLICATION_CHECKS.md).

**Incremental release:** If the benchmark dataset is publicly published before the model-collection question is resolved, `python -m scripts.verify_hf_collections --dataset-only --verify-network` checks that dataset and its dataset collection independently while deliberately preserving **NOT READY** for the official submission. This mode never fabricates or requires a model artifact; it writes a separate partial receipt. [Details](docs/HF_ANONYMOUS_PUBLICATION_CHECKS.md).
