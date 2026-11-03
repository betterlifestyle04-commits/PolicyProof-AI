# PolicyProof AI — Lexsi Track 1 submission draft (working document)

> **Do not submit as a completed improvement study yet.** This is a write-up skeleton with verifiable claims only. The genuine tuned-model results, public Hugging Face artifact URLs and final adjudicated semantic metrics are still missing.

## Project overview

PolicyProof AI explores whether an openly available multilingual language model can be made more reliable at answering questions from *provided policy text*—rather than answering from general knowledge. The model must return a structured JSON object with an answer, a supporting policy identifier and an explicit abstention decision when evidence is insufficient.

The study focuses on three languages: English, Norwegian and Arabic. Its policy documents are fictional and designed for reproducible technical testing. The prototype is **not** an actual financial advisory system.

## Why this problem?

Small multilingual models are practical for privacy-sensitive, self-hosted document workflows, but accurate wording alone does not establish reliability. A useful grounded answer should cite a source, respect missing evidence, follow the required output schema and ignore commands embedded inside untrusted documents. PolicyProof examines these independently rather than hiding regressions behind one overall grade.

## Toolchain and training choices

- **Cohere Labs Tiny Aya Global:** the eligible dense open-weights multilingual base model (approximately 3.35 billion parameters), gated noncommercial license accepted by the project owner
- **CuratorKIT:** completed local schema checking, exact deduplication, cleaning and Alpaca/ShareGPT export for 78 fictional training examples (26 per language)
- **AlignTune:** v0.1.28 trainer factory and 4-bit LoRA model loading verified on a Colab Tesla T4; initial optimizer runs **failed** at BF16 gradient unscaling. This is a known technical blocker, not a successful fine-tune
- **AuditKIT:** actual official AuditKIT 1.1.3 replay executed successfully in CI against all 48 saved user-run Tiny Aya baseline responses; fingerprint `ae297dd84f990566`. A tuned-model comparison remains pending
- **Possible direct PyTorch fallback:** under investigation. If successful, label separately as a PyTorch training run, not an AlignTune trainer success

## Methodology

The initial dataset contains 78 synthetic training examples using 12 invented policy records. The historical development benchmark contains 48 cases drawn from six different policy records, including unsupported questions and document-injection probes. Those 48 cases were inspected during debugging and therefore cannot serve as an unbiased final test.

A separate 60-case candidate benchmark was created using eight new fictional policy records, with 20 samples per language. It must receive independent bilingual/evidence review and be frozen before the final paired comparison. It must never enter training or prompt/hyperparameter selection.

For the actual comparison, use the same base checkpoint revision, deterministic decoding, policy prompts, schema validation and evaluation questions for the base and candidate models. Measure exact-match score, JSON-schema compliance, correct cited policy ID on answerable cases, unsupported-question abstention, unsupported-claim rate and language/injection slices. Report both deterministic and human-reviewed semantic results.

## Verified baseline — unchanged Tiny Aya Global, 48 fictional development cases

| Measurement | Verified baseline | Interpretation |
| --- | --- | --- |
| Strict end-to-end exact match | **8/48 = 16.67%** | Fails correct paraphrases when text differs |
| Complete 3-field JSON schema | **33/48 = 68.75%** | All 48 were parseable JSON objects |
| Raw expected source ID, answerable questions | **38/39 = 97.44%** | Source-ID match alone does not establish answer correctness |
| Correct unsupported-question abstention | **8/9 = 88.89%** | Small denominator |
| Substantive unsupported claim | **1/9 = 11.11%** | Corrected conservative evaluator |
| Human-reviewed semantic accuracy | **Not measured** | Blank review sheet prepared, no judgments yet |

Reference: [Baseline report](BASELINE_2026-11-02.md). These are baseline diagnostics, not improvements.

## Separately measured inference-time safeguard (NOT model fine-tuning)

A deterministic fail-closed output validator processed the same 48 saved unchanged Tiny Aya responses without new GPU inference. It rejects incomplete JSON schemas and contradictory abstentions, preserving valid answer-shaped outputs. On the **inspected development split**, the following deterministic results were recorded:

| Metric | Raw saved Tiny Aya | After deterministic guard |
| --- | ---: | ---: |
| Complete required JSON schema | 68.75% | **100%** |
| Correct unsupported-case abstention | 88.89% | **100%** |
| Unsupported-case substantive claim | 11.11% | **0%** |
| Strict exact end-to-end match | 16.67% | **18.75%** |
| Exact supported answer match | 0/39 | **0/39** |

This is a real *inference-time validator* improvement, but it is NOT AlignTune training, does not change the model's weights, and does not demonstrate semantic correctness. It is an exploratory safety-repair fallback only. [Full original 48 predictions](../artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json) · [evaluation limitations](INFERENCE_GUARD_RESULTS_2026-11-03.md).

## Tuned model results — REQUIRED BEFORE SUBMISSION

**Status: NOT AVAILABLE.** Do not fabricate any of the following. Fill them only from the first complete, saved checkpoint and paired evaluation:

- Exact adapter checkpoint URL and SHA256: **PENDING**
- Actual training backend, verified optimizer steps, seed and hardware: **PENDING**
- Frozen dataset checksum and reviewed gold set: **PENDING**
- Candidate strict accuracy and schema compliance: **PENDING**
- Candidate citation validity, abstention and unsupported-claim rates: **PENDING**
- Independent semantic review and inter-reviewer agreement: **PENDING**
- Paired deltas, uncertainty intervals and language regressions: **PENDING**
- AuditKIT evaluation run fingerprint and export: **PENDING**

## Limitations and lessons

This study starts with a very small, repetitive, template-generated, fictional corpus. Three-language performance partly tests translation from English source documents. Exact string comparison can penalize semantically correct answers, while citing the expected ID does not ensure factual grounding. The benchmark is not a representative bank policy corpus or evidence of real-world financial safety.

The T4 environment also exposed a reproducibility problem between AlignTune's extra precision arguments, TRL v1.7.1's automatic BF16 QLoRA adapter conversion and PyTorch's FP16 gradient scaler. Disclose these negative results and the final actual backend rather than claiming the requested integration always worked.

## Links — complete only with real verified URLs

| Resource | URL / status |
| --- | --- |
| Source code | https://github.com/betterlifestyle04-commits/PolicyProof-AI |
| Baseline research report | https://github.com/betterlifestyle04-commits/PolicyProof-AI/blob/main/docs/BASELINE_2026-11-02.md |
| CuratorKIT report | https://github.com/betterlifestyle04-commits/PolicyProof-AI/blob/main/docs/CURATION_2026-11-03.md |
| HF adapter | **PENDING** |
| HF curated dataset | **PENDING** |
| HF Article / report | **DRAFT:** [research article](HF_ARTICLE_DRAFT.md), publication pending |
| Public live demo | **Optional:** public GitHub repository available; hosted Streamlit demo not verified |
| Submission receipt | **PENDING** |

## Suggested 3-minute demo storyboard

1. **0:00–0:25 — Problem.** Show an English/Norwegian/Arabic question, relevant policy evidence and output schema.
2. **0:25–0:55 — Dataset.** CuratorKIT curation results, 78 examples, synthetic-data limitations and source-policy splits.
3. **0:55–1:20 — Model.** Tiny Aya Global and T4 setup; show real adapter and logs **only if training actually succeeded**.
4. **1:20–2:15 — Evaluation.** Use frozen identical questions, cite real before/after metrics, language slices, unsupported claims and reviewer findings. Never substitute mock metrics.
5. **2:15–2:45 — Failure and risks.** Show prompt-injection handling, regression cases and the real toolchain limitations.
6. **2:45–3:00 — Reproducibility.** Link GitHub, HF artifacts, exact hashes, and invite judges to reproduce.

## Gate for public submission

**Do not mark ready** until (a) the required toolchain has been executed and accurately described, (b) a legitimate trained checkpoint is available, (c) model comparison is real and reproducible, (d) HF artifacts and license are compliant, (e) the public demo/repository works without privileged access, and (f) the submission receipt is confirmed before November 14, 2026.

Official requirements: https://hackathon.lexsi.ai
