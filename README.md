## November 10 — matched attack-family comparison for eventual 72-case GPU results

We added a new **source-pinned CPU analysis** that compares original 72-case Tiny Aya outputs against our unchanged deterministic inference-time guard. It reports paired structural and reference-string metrics, English/Norwegian/Arabic slices, and **12 matched control/attack task pairs per attack family**. It does not use gold answers inside the guard, rerun the model, train weights or fabricate 72 outputs. A frozen original GPU report must exist first; the current 72 challenge remains **unscored** until that happens. [Exact protocol and command](docs/TRACK2A_72_EVALUATION_PROTOCOL.md).

---

## November 9, 2026 — real public Track 2a dataset and read-only recovery

The owner-authorized Hugging Face upload on November 9 **created a real public, ungated synthetic dataset** and a separate public dataset collection. The original upload workflow failed **after uploading** because our anonymous-verification JSON reader incorrectly rejected normal Hugging Face snapshot-cache symlinks. That verification bug is fixed, and the repository now has a separate pinned **anonymous, read-only SHA256 verification workflow** that needs no HF token or new upload.

- [Published synthetic benchmark: 48 historical saved base-model outputs + 72 UNSCORED adversarial challenges](https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark)
- [Public dataset collection](https://huggingface.co/collections/shalnark01/policyproof-ai-multilingual-safety-datasets)
- [Automatic anonymous integrity check](.github/workflows/verify-public-track2a.yml) · [Read-only recovery instructions](docs/TRACK2A_V2_PUBLISHING.md)

**Do not rerun the publisher for this already-existing release.** The dataset and its collection are real, but the separate relevant public **model collection** and the actual **Lexsi organizer submission** remain outstanding. The synthetic examples and deterministic safety guard do not establish a trained model, independent semantic correctness or real-world financial safety.

---

## November 5 — read-only native training crash recovery audit

The native training launcher now writes its `in_progress` receipt **before** calling AlignTune's optimizer. A hard Colab interruption therefore cannot leave a nonexistent receipt that is later mistaken for completion. A final local receipt write failure after successful ZIP publication is recorded separately, not as failed model training. The new [CPU-only read-only reconciliation tool](scripts/reconcile_native_training_evidence.py) and **Stage 11 Step 7** compare original local evidence with safe ZIP content without retraining or mutating files. Even a perfectly matched archive remains only internally consistent self-reported research evidence, not independently attested GPU execution. [Protocol](docs/NATIVE_ALIGNTUNE_40STEP_T4_2026-11-05.md).

## November 5 — atomic native training artifact release

PolicyProof's native AlignTune path now checks **both** the TRL optimizer state and the independent step count returned by AlignTune, and packages an adapter ZIP in a private temporary file. After exact-three-member and CRC checks it publishes the ZIP atomically without overwriting an existing download, then marks its local receipt completed. Failed publication keeps a `failed` local receipt and does **not** present an incomplete ZIP. This prevents a training-export failure being mistaken for completed model evidence. The native run remains **unexecuted on a real T4** here. [Training evidence details](docs/NATIVE_ALIGNTUNE_40STEP_T4_2026-11-05.md).

## November 5 — prevent redundant AlignTune training-data reload and bloated artifacts

The 10/40-step native training path now hands the **original prechecked 78-row dataset object** directly to AlignTune's TRL dataset cache. The pinned 0.1.28 trainer would otherwise run `setup_data()` a second time inside `train()`. Its real installed-package behavior is covered by our separate SDK CI gate. The portable ZIP now contains only the safe PEFT adapter weights, adapter config and training receipt—**not** intermediate full checkpoints or local logs. [Technical details](docs/NATIVE_ALIGNTUNE_40STEP_T4_2026-11-05.md). These fixes are not evidence that native GPU training has actually completed.

The native 10/40-step training guard also now explicitly disables *load best model at end* whenever there is no evaluation dataset, closing an incompatibility found in the actual pinned AlignTune source. The final constructed TRL settings must remain `eval_strategy=no` and `load_best_model_at_end=False` before optimizer execution.

## November 5 — real pinned AlignTune/TRL source contract checks (no GPU)

GitHub CI now installs the **actual AlignTune 0.1.28 and TRL 1.7.1 wheels** in a separate lightweight CPU job and inspects the real SDK configuration-builder, factory, late-argument merge and SFTConfig source. A version/source/API drift fails the new job. The [published-source verification protocol](docs/REAL_ALIGN_SDK_SOURCE_CONTRACT_2026-11-05.md) explains exact SHA receipts and limitations. **This is source compatibility, not a runtime import, SFTConfig construction or executed native training.** The 40-step [T4 notebook](notebooks/11_native_aligntune_40step_t4.ipynb) still requires a real owner-triggered GPU run.

## November 5 — new native AlignTune 40-step T4 trial (unexecuted)

**Official 40-step training is now implemented as a separate explicit-opt-in route**, distinct from both the 10-step AlignTune smoke and prior 40-step *direct-PyTorch* fallback: [open the standalone T4 notebook](notebooks/11_native_aligntune_40step_t4.ipynb) · [detailed GPU evidence protocol](docs/NATIVE_ALIGNTUNE_40STEP_T4_2026-11-05.md). The procedure verifies original training-only data SHA, exact AlignTune/TRL optimizer budget, disabled mixed precision on T4, unmodified frozen 4-bit weights and separate PEFT archive. Its receipt records whether the loaded Tiny Aya SHA was actually observed; a Hub-model SHA alone is not sufficient for a 40-step Track 1 candidate. **This code has not been run on a live GPU through this change; no factual improvements, model publication or submission are claimed.**

## November 5 — cite an actual, unique policy block (not merely a header)

The deterministic [output guard](scripts/inference_guard.py) now requires **exactly one nonempty policy text block** for every cited `POL-...` ID, before the first untrusted-note boundary. A duplicated or empty policy header causes canonical abstention with a distinct `ambiguous_policy_source` diagnostic. The [numeric evidence](scripts/numeric_evidence_audit.py) and [multilingual topic-relevance](scripts/source_topic_audit.py) audits share the same source parser and send ambiguous citations to **human review**, not automatic factual approval. Two interactive judge-demo examples exercise both paths. This fixes disagreement between safety layers; it does **not** establish document authenticity, source entailment, native AlignTune training or model semantic accuracy. [Reproduction and limitations](docs/CITATION_SOURCE_BOUNDARY_2026-11-05.md).

## November 5 — immutable 60-case human-review files

Final candidate gold-label review packet exports and reconciliation are now **private, create-once and non-overwriting**. The [60-case review guide](docs/FINAL_REVIEW_WORKFLOW.md) explains how to preserve genuine reviewer CSVs and their hashes. The CLI refuses paths outside `reviews/`, symlinked inputs, malformed CSV records and duplicate reviewer paths. **No review was performed or signoff granted by this software change.**

## November 5 — real human disagreement resolution, still blinded

The [dev48 human review protocol](docs/DEV48_BLINDED_SEMANTIC_REVIEW.md) now includes **dispute-only third human adjudication**, fixing the previous dead end when two genuine reviewers disagreed. [CPU Colab: prepare anonymized dispute packet](notebooks/09_dev48_blind_dispute_export_cpu.ipynb) → [CPU Colab: verify a genuinely completed third review](notebooks/10_dev48_blind_dispute_resolve_cpu.ipynb). The private original reviewer CSVs and model-mapping key are SHA-pinned and never overwritten, and **only contested judgment fields** can be reconciled by a real independent adjudicator. No human votes, factual model improvement, or official submission acceptance are generated automatically.

## November 5 — no oracle labels in blinded human-review packets

The [development A/B human-review workflow](docs/DEV48_BLINDED_SEMANTIC_REVIEW.md) now exports **v2 masked packets**: opaque random case identifiers and the same neutral `review_case` category for every response. This fixes leakage of `UNKNOWN`/`INJECT` case IDs and unsupported-answer labels to blinded reviewers. The owner-only sealed key stores the true IDs/categories, and model order is balanced 8/8 per language. [One-ZIP CPU review notebook](notebooks/07_dev48_blind_human_review_cpu.ipynb) checks these protections before download. No reviewer judgments are fabricated or collected by the change; human review remains outstanding.

## November 5 — wrong-product source-citation checks (no GPU)

The [conservative multilingual cited-topic audit](docs/CITED_POLICY_TOPIC_AUDIT_2026-11-05.md) now flags clear cases where a model gives an overdraft-rate answer using a **real but unrelated savings withdrawal fee**: a number can match the cited policy and still concern the wrong product. The public guard demo and private SHA-pinned base-vs-adapter triage expose topic discrepancies without training, new inference or using gold labels for topic classification. **A topic match is not evidence entailment, a flag is not a factual verdict, and neither model quality nor submission status is proved.**

## November 5 — CuratorKIT ZIP security and early training-data validation

The training-data [CuratorKIT export notebook](notebooks/02_curatorkit_training_data.ipynb) now validates the exact pinned 78-row original export and scans the ZIP **before download**. The shared [archive verifier](scripts/verify_curator_archive.py) limits total decompressed size, member count, compression bombs, symlinks, path anomalies, and ambiguous signed JSON. The [Colab T4 restorer](scripts/restore_colab_session.py) additionally pins the source ZIP checksum **across validation and private staging** so an archive swapped mid-import cannot silently populate training files. [Full safeguard and reproduction guide](docs/CURATORKIT_ZIP_SAFETY_2026-11-05.md). CPU security tests verify these controls; **no new real AlignTune training, independent human semantic review, or final model evaluation occurred.**

## November 5 — verified, non-destructive Colab training data restoration

The [original AlignTune/T4 notebook](notebooks/03_aligntune_qlora_t4.ipynb) now restores the exact SHA-pinned 78-row CuratorKIT data through private staging and **never overwrites** an existing nonidentical training file. It also checks the requested Tiny Aya model revision against the *loaded* model config when available; missing loaded revision metadata remains explicitly unverified. [Training recovery protection and reproducibility guide](docs/COLAB_CURATOR_RESTORE_SAFETY_2026-11-05.md). **This change is CPU-tested only; no new AlignTune training or semantic model improvement was performed.**

## November 5 — reproducible native AuditKIT rerun from saved GPU output ZIP

A new [CPU-only Colab notebook](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/08_native_dev48_auditkit_cpu.ipynb) and [strict replay script](scripts/replay_dev48_native_auditkit.py) let the project owner verify the original SHA-pinned 48-case results ZIP, then actually run official **AuditKIT 1.1.3** on the 48 fresh-base and 48 adapted-model *saved* responses. The resulting receipt is private and **is not included as completed empirical evidence until the owner actually runs it**. It cannot prove GPU execution, factual improvement, or blind final performance. See [replay protocol](docs/DEV48_NATIVE_AUDITKIT_REPLAY.md). **Protect the base/tuned report from blinded reviewers.**

## November 5 — strict release evidence gates (Track 1 and Track 2a)

The [source-linked submission gate](docs/RELEASE_EVIDENCE_GATES_2026-11-05.md) now rejects fake human-review toggles, stand-alone paired-AuditKIT status JSON, and self-attested Hugging Face publication flags. The **Judge verification** page shows live local decision ledgers for both tracks. The original 48-case paired results remain synthetic development evidence; trained model semantic improvement, final reviewed 60-case evaluation, publication, and submission are **not verified**. Neither successful CI nor a working dashboard substitutes for those prerequisites.

# PolicyProof AI — Lexsi Labs Hackathon 2026

> **Latest research status (November 5, 2026):** A user-supplied 40-optimizer-step LoRA checkpoint ZIP passed structural and SHA-256 checks. A fresh same-requested-revision 48-case base-vs-adapter saved GPU comparison passed independent internal score/hash recomputation. JSON compliance improved 14/48 → 48/48 and correct source IDs improved 1/39 → 39/39; **answer-level semantic correctness is unverified**, supported-answer exact-string matches remain 0/39, and serious unsupported financial-policy answers remain. Original CUDA execution and loaded training-time model revision were not independently rerun/verified. The 60-case candidate final evaluation still lacks independent human gold review; Hugging Face publication and official submission are not verified. **Do not present this as a production-ready banking system or verified real-world safety.**

**[November 5 paired GPU development results](docs/DEV48_PAIRED_GPU_RESULTS_2026-11-05.md)** · **[Private, model-blind human review instructions](docs/DEV48_BLINDED_SEMANTIC_REVIEW.md)** · **[Review implementation](scripts/paired_semantic_review.py)**

**New evidence-boundary diagnostic:** [Citation-local numeric risk audit](docs/NUMERIC_EVIDENCE_AUDIT.md) flags amounts and other numeric literals that occur in a generated answer but not the *specifically cited* policy block. This runs on the saved 48-case baseline and is integrated into the public output-guard experiment and playground. Matching a number is **not** factual verification; invented overdraft claims may reuse a real savings fee and still pass the literal check. No model tuning or GPU was performed by this feature.

**Fastest next action, CPU only:** [Open Colab review-packet notebook](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/07_dev48_blind_human_review_cpu.ipynb), upload the original audited `policyproof_dev48_paired_gpu_results.zip`, and save the blank A/B reviewer CSV **plus** its private mapping key. The SHA-pinned [ZIP verifier and exporter](scripts/export_review_from_bundle.py) checks archive and raw output integrity without rerunning Tiny Aya. Real reviewers must still independently judge correctness and evidence; no notebook or test automatically approves the model.

**Historical note:** Some older milestones and GPU-quota troubleshooting sections below describe the November 3 development state; the November 5 evidence summary above supersedes those historical statements where they conflict.


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

[Read the November 3 QA report](docs/QA_VERIFICATION_2026-11-03.md) for historical baseline and CuratorKIT consistency checks. For the later saved adapter and paired development experiment, see the [November 5 evidence report](docs/DEV48_PAIRED_GPU_RESULTS_2026-11-05.md); model-level semantic improvement is still unverified.

## Current project milestones (updated November 3, 2026)

| Workstream | Actual status |
| --- | --- |
| Baseline inference | **Completed:** 48 genuine Tiny Aya outputs on Colab T4; honest limitations documented |
| CuratorKIT | **Completed:** 78/78 synthetic examples, three languages and audited archive checksums |
| AlignTune training | **Native trainer remains unproven:** original BF16 GradScaler failures; later saved 40-step manual Torch fallback is structurally verified, not a verified native AlignTune run |
| Direct PyTorch fallback | **Checkpoint archive inspected (Nov 5):** original 40-step adapter ZIP and LoRA tensor/hash structure verified; GPU training execution remains self-reported |
| AuditKIT | **Executed:** 48 saved Tiny Aya base outputs successfully replayed with real AuditKIT 1.1.3; fingerprint ae297dd84f990566 |
| Track 2a safety benchmark | **Actual CuratorKIT → AuditKIT chain completed on 48 recorded Tiny Aya outputs**; additional **72 curated synthetic attack cases are UNSCORED**. [Research report](docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md) · [alternate abstract](docs/TRACK2A_SUBMISSION_ABSTRACT.md); official HF collections and submission pending |
| Candidate final holdout | **Draft frozen in GitHub:** 60 independent multilingual cases; reviewer approval pending |
| Submission | **Not yet submitted:** 48-case development paired comparison now internally verified; independent semantic review, valid final 60-case comparison, public HF model/data collections and actual Lexsi submission remain outstanding |

**Fresh evaluation:** [60-case candidate](data/final_holdout_v1.jsonl) · [integrity checker](scripts/validate_final_holdout.py). Do not train on or tune against this set.  
**AuditKIT:** [48 actual saved baseline predictions](artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json) were replayed successfully through real AuditKIT 1.1.3 on GitHub Actions, with no newly generated inference. [Guarded output results](docs/INFERENCE_GUARD_RESULTS_2026-11-03.md) are a separate deterministic *inference-time* experiment, not a trained adapter.  
**Submission tracker:** [Track 1 status](docs/SUBMISSION_READINESS.md) · [Track 2a separate gate](scripts/track2a_readiness.py). Neither track is officially submitted. The Track 2a gate does NOT require a fictitious fine-tuned model.

## Historical November 3 GPU quota issue and Kaggle alternative (superseded)

Google Colab currently returned **"Cannot connect to GPU backend"** for the user. The original **78-row CuratorKIT ZIP restored successfully** on CPU, but no new Tiny Aya model has been loaded or trained since the Colab quota restriction. Google does **not** publish a guaranteed GPU quota reset time.

**Kaggle route:** [open the Kaggle GPU fallback notebook](notebooks/04_kaggle_gpu_fallback.ipynb). In Kaggle Settings, enable Internet and use GPU T4 x2 if offered (alternatively P100); add the original CuratorKIT training ZIP as a private input; configure an approved read-only HF token in Kaggle Secrets; then run the notebook in order. It prepares the quantized model and tokenizes all 78 examples, with *actual training disabled* until explicitly opted in. Hardware availability and project-specific success on Kaggle have **not been verified**.

**No GPU needed:** `python -m scripts.prepare`, `python -m scripts.validate_final_holdout`, and `python -m pytest -q` perform fixture generation and structural tests on CPU. They do **not** prove model improvement.

**Do not retry the old `recover_train_save` cell.** In the user's previous run, AlignTune/TRL still encountered BF16/FP16 precision conflicts. The documented direct PyTorch fallback was built to avoid those training-loop issues, but has not yet produced a successfully verified checkpoint.

## Historical T4 fallback development instructions (see latest November 5 checkpoint summary above)

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

Status: active fictional-policy research prototype; structural inspection supports a saved 40-step adapter and paired development-evaluation outputs, but independent GPU execution, human semantic improvements, final evaluation and final hackathon submission are not verified. All policies and evaluation examples are fictional, never actual bank information.

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

## New: reproducible, unpublished Track 2a research handoff

Every passing **real CuratorKIT → AuditKIT CPU** GitHub Actions job now assembles an **unpublished, locally verifiable Track 2a ZIP** from its actual toolkit outputs and the repository's saved Tiny Aya baseline report. This prevents our handoff from depending exclusively on an older, separately pinned 20-file archive.

The bundle includes the original **48 evaluated** synthetic development cases, the **72 separately unscored** synthetic adversarial cases, native CuratorKIT and AuditKIT receipts, source/curator exports, a research-only card, a strict "not published / no license / no trained model" status and full file checksums. It uses deterministic file ordering and archive metadata, validates real original-model scores, and refuses unsafe filenames or evidence claims.

To inspect without extracting or publishing:

```bash
python -m scripts.build_track2a_release_candidate \
  --inspect track2a_release_candidate_v2_UNPUBLISHED.zip
```

Get the ZIP from the `policyproof-track2a-real-curatorkit-auditkit-20261103` artifact of a **passing main-branch CI run**, or reproduce it locally with the documented CPU toolkit commands. The older hash-pinned owner publication script **intentionally does not accept** this new release candidate until its contents and licensing are independently approved. This improvement is **not** a Hugging Face upload, model fine-tune, new model inference, or official submission. [Read the reproducibility and release protocol](docs/TRACK2A_REPRODUCIBLE_HANDOFF.md).

## New: owner-approved v2 dataset publication is prepared (no upload yet)

The fresh **15-member** research ZIP produced by our current CuratorKIT/AuditKIT CPU CI job now has a **separate, strictly consent-gated Hugging Face dataset publisher**. The original historical 20-file pinned publisher and digest remain unchanged.

`scripts.publish_track2a_candidate` checks the **exact 64-character SHA256 supplied from a trusted CI run**, validates the complete bundle, recomputes original model metrics, checks honest 48-evaluated / 72-unscored status, and **defaults to an offline, license-free inspection**. It can prepare a temporary public-file preview after the **owner chooses a permitted license**. An actual network upload requires both `--confirm-rights` and `--confirm-publication`, a private `HF_TOKEN`, and the owner's dataset repository/collection choice. It verifies the staged file hashes **before any network call**, then independently checks the uploaded repo, collection, declared license, saved 48/72 records and SHA256SUMS with anonymous access before recording success.

This tool **has not published anything**, does not decide the owner's licensing rights, and cannot create a trained model or meet Lexsi's still-unresolved model-collection requirement. [Owner-facing publication runbook](docs/TRACK2A_V2_PUBLISHING.md).

### Full-file verification for future Hugging Face publication

The anonymous HF verifier now checks **every published research file**, not just the required five: full `SHA256SUMS.txt` coverage, safe filenames, no undeclared uploads, fixed v2 layout, a pinned dataset Git revision and source ZIP SHA. The v2 publisher requires the public revision to equal the actual upload commit and the public source claim to equal the originally approved archive hash. A separate CI job checks method/attribute compatibility against the *actual installed Hugging Face Hub SDK* without making HTTP requests. All publishing remains **opt-in**, and no published dataset or model is claimed by this check. See [full verification protocol](docs/HF_ANONYMOUS_PUBLICATION_CHECKS.md).


### Human review must be backed by original reviewer files

The 60-case candidate cannot enter final base-vs-adapter GPU evaluation merely because a JSON signoff says `review_complete: true`. `python -m scripts.verify_review_provenance` and the paired runner now require genuine 2–4 reviewer CSV files under **private, Git-ignored** `reviews/`, full per-file SHA256 values, exact dataset/gold/source preservation, all 180 positive judgments per reviewer, an independently recomputed reconciliation, and individual approval declarations. **This verifies file consistency only; no human identity, proficiency, independence or semantic correctness is independently authenticated.** The signoff remains explicitly incomplete and no human reviews have been fabricated. [Review-evidence protocol](docs/REVIEW_SOURCE_PROVENANCE.md).


For human-reviewed final benchmark signoff, `scripts.prepare_review_signoff` can generate a **private unsigned draft only** from completed, unanimously reconciled CSVs, leaving every approval field false. Nothing in CI performs actual human review or creates a completed signoff. [Instructions](docs/REVIEW_SOURCE_PROVENANCE.md).

## Next GPU milestone: real 40-step Tiny Aya QLoRA training

The model **has not yet completed a real GPU training run**. Alongside the 10-step integration smoke, the repository now offers an explicit **40-step direct-PyTorch QLoRA fallback** on an existing AlignTune-loaded quantized Tiny Aya model. It uses 66 of the 78 original CuratorKIT training-only samples for optimizer steps and 12 for internal validation-loss diagnostics. It never trains on the 48/60/72 evaluation sets. There is a separate checkpoint archive `policyproof_t4_torch_extended.zip`, with loss logs, exact base revision, and file checksum-verifiable adapter metadata. **No automatic training or GPU budget allocation occurs in CI.** A real T4/P100 Colab or Kaggle runtime is required; model improvement must be measured separately.

[Open the GPU execution instructions](docs/GPU_TRAINING_NEXT_STEPS.md) · [Colab training notebook](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/03_aligntune_qlora_t4.ipynb) · [Kaggle fallback notebook](notebooks/04_kaggle_gpu_fallback.ipynb).

## New: genuinely paired GPU evaluation of the 40-step checkpoint (not yet run)

The uploaded `policyproof_t4_torch_extended.zip` is a structurally valid 40-step `manual_torch_extended_fallback` adapter. **There are no real base-versus-adapter paired predictions yet.** We now provide a separate, default-OFF [Colab T4 evaluation notebook](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/06_dev48_pair_eval_t4.ipynb) and `scripts.dev48_gpu_pair` for an *actual* fresh 48-case development comparison. The old saved Tiny Aya baseline has **no pinned revision**, so this runner explicitly generates a **new** untouched base-model pass at the *same requested model revision* as the adapted pass; it does not recycle the historical outputs or inspect the unreviewed 60-case final candidate. Both real GPU output files are saved write-once, and a native AuditKIT CPU comparison and downloadable result ZIP can be produced afterward. Training does **not** run during this evaluation.

The 48-case set is previously inspected synthetic development material, **not blind evaluation or independent semantic proof**. The original 40-step checkpoint's training receipt does not independently verify the revision of weights loaded during training; the run uses its recorded HF revision as a fresh, matched inference request, with that limitation disclosed. [Exact step-by-step instructions](docs/EVALUATE_REAL_40_STEP_ADAPTER.md).
