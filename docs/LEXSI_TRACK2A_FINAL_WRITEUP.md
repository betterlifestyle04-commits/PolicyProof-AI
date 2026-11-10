# PolicyProof AI — Multilingual evidence-grounding and safety failure analysis

**Lexsi Labs AI Safety and Alignment Hackathon 2026 — Track 2a, adversarial and multilingual safety benchmark (research write-up).**

**Submission status: Draft, NOT submitted.** This article is reproducibility documentation, not a claim of acceptance. The public-facing model collection, adapter, and private research outputs require separate release-rights approval and successful anonymous verification.

## Executive summary

PolicyProof AI investigates how well a small multilingual open-weights model follows fictional policy evidence under structured-response and adversarial conditions. We tested English, Norwegian and Arabic question answering, evidence citation and abstention for unsupported questions. In a separate 40-step LoRA tuning trial, complete JSON formatting rose from 14/48 to 48/48 on an already-inspected development set, but unsupported-question abstentions fell from 4/9 to 2/9. **Our central finding is that schema compliance does not imply grounded correctness or safety.**

The project contributes a synthetic multilingual benchmark, source-hashed research reports, reusable evaluation scripts, adversarial variants, offline review workflow and an experimental adapter with explicitly documented limitations.

## Track and toolchain

The Track 2 safety pipeline uses **CuratorKIT** for validated synthetic training/data curation and **AuditKIT** for evaluation, satisfying the two-tool track minimum as described on the [official track page](https://hackathon.lexsi.ai/). We additionally used **AlignTune** for a controlled native 40-step LoRA trial. The project **does not claim to have used SafeTune or CircuitKIT**. Its own deterministic inference-time JSON/abstention guard is not a substitute for these official toolkits.

The open-weights source model is [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), with a requested model revision of `af89d219b53ed9b13b8a4645f9c8028973510324`. This source has CC BY-NC terms and gated file access; publishing the base-model weights is outside this project.

## Data and experimental design

- **78 curated training cases:** synthetic fictional-policy examples curated with CuratorKIT, original SHA256 `4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c`. These are the native AlignTune training cases.
- **48 development cases:** previously inspected synthetic items (16 per language), including nine unsupported questions. Both unchanged base and the native adapter were compared using the same prompts and deterministic decoding. The dev set is not an independent final benchmark.
- **72 adversarial challenge cases:** source-pinned synthetic questions and prompt-injection variants. Original unchanged-base outputs were saved and strict-scored; these remain separate from training. Because attacked variants reuse underlying questions, cases are correlated rather than 72 independent policy tasks.
- **60-case private final review candidate:** reserved for independent human semantic review, which has **not been completed**. No final blind semantic-accuracy claim is made.

Dataset source files and documentation are in the public [PolicyProof dataset](https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark) and [GitHub repository](https://github.com/betterlifestyle04-commits/PolicyProof-AI).

## Experiment A — unchanged base on 72 adversarial cases

The original owner-run experiment saved 72 Tiny Aya model responses and an AuditKIT 1.1.3 local replay receipt. The private archive has SHA256 `f86d1002a162ceac44270cdb25edf2769ee471d469faa158f0ad1719efc65d57`. Its saved original source counts were independently checked for hash consistency and against the strict scorer.

| Descriptive strict-reference metric | Unchanged base |
| --- | ---: |
| Exact reference matches | 11/72 (15.3%) |
| Valid required JSON schema | 58/72 (80.6%) |
| Correct unsupported-question abstentions | 6/12 (50.0%) |
| Correct supported-case citation identifiers | 28/60 (46.7%) |

The strict exact-match numbers do **not** establish semantic factual accuracy. An equivalent paraphrase may fail the string rubric. Attack-marker echo also does not measure full prompt-injection success. Original GPU execution metadata are self-reported, not externally attested.

An independently applied deterministic output guard increased JSON-valid responses from 58/72 to 72/72 and correct unsupported abstentions from 6/12 to 11/12. Supported-answer exact match remained 5/60 both before and after, and accepted answer coverage decreased. One unsupported claim remained after guarding. This is **postprocessing, not weight-space safety repair**.

See [72-case original-source research report](TRACK2A_72_OBSERVED_PRIVATE_SOURCE_RESULTS_2026-11-10.md).

## Experiment B — native AlignTune 40-step trial and matched 48-case development evaluation

The owner saved a separate PEFT LoRA adapter after a self-reported native AlignTune 40-step T4 session. The saved artifact has 288 BF16 LoRA tensors totaling 3,833,856 nonzero elements, with independently checked file hashes. The originally loaded base revision during training was not independently observable. The adapter ZIP is private until licensing/release review.

The fresh, matched development comparison saved an original model response report and an adapted model response report at the same inference-time model revision and decoding configuration.

| Metric | Base | Native AlignTune adapter |
| --- | ---: | ---: |
| Complete JSON schema | 14/48 | **48/48** |
| Expected raw citation ID, 39 supported cases | 31/39 | **39/39** |
| Strict exact supported-answer text | 0/39 | 0/39 |
| Correct unsupported abstentions | **4/9** | 2/9 |
| Unsupported-question claims | **5/9** | 7/9 |
| All-case strict exact matches | **4/48** | 2/48 |

The drop in unsupported abstention was concentrated in two Arabic examples (2/3 → 0/3), a very small, nonrepresentative subgroup. Matching the source ID **does not demonstrate that the answer follows from the cited passage**. The training trial is best described as increasing syntactic response reliability while worsening abstention under this strict development rubric.

See [paired native AlignTune research report](TRACK2A_NATIVE40_DEV48_OBSERVED_PRIVATE_RESULTS_2026-11-10.md).

## What the experiments show, and what they do not

**Supported:** syntactic structured-output improvement in one synthetic 48-case development sample; failures of strict supported-answer matching and abstention; reproducible frozen dataset manifests, test harnesses and private original-source hashes.

**Not established:** improved semantic accuracy, robust safety across languages, attack success rates, causal generalization, human review, verified train-time model revision, or independently audited GPU execution.

We deliberately preserve negative evidence. Optimizing exclusively for parsable JSON can create *more convincing unsupported answers*. Future work should jointly measure factual entailment, calibrated refusal, abstention rates and user-visible answer coverage using **new disjoint development sources**. Human bilingual reviewers should examine paraphrases and unsupported questions before making accuracy or safety claims.

## Reproducibility and public artifact map

- **Code and notebooks:** [PolicyProof-AI](https://github.com/betterlifestyle04-commits/PolicyProof-AI), including CuratorKIT source replay, native AlignTune Stage 11, Stage 12 72-case baseline, and Stage 13 paired dev48 notebook.
- **Dataset:** [policyproof-track2a-benchmark](https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark).
- **Dataset collection:** [multilingual safety datasets](https://huggingface.co/collections/shalnark01/policyproof-ai-multilingual-safety-datasets).
- **Model/adapter collection:** [Public model collection](https://huggingface.co/collections/shalnark01/policyproof-ai-lexsi-track-2-model-and-adapter-research-6af39caac578d29e1c66c6) now contains the [public, ungated adapter-only repository](https://huggingface.co/shalnark01/policyproof-ai-native40-align-2026). The public adapter LFS SHA256 matches the owner's original. Gated upstream base weights were NOT published. The original Colab publication command returned a false-negative exit after remote creation; independent anonymous full-content verification has its own read-only action.
- **Research logs:** Aggregate, source-hashed scores above. Original raw private model responses and trainer logs have not been publicly released; rights for redistribution remain under owner review.

The competition form must include the owner's **registered email**, a truthful abstract of fewer than 500 words, HF public model/dataset collections, public code and this write-up. The competition's first-stage deadline is **15 November 2026, 11:59 PM AoE**. The organizer confirmation must be saved separately.

## Limitations and ethics

All policies and many attack prompts in the challenge are fictional and synthetic. The sample is small and correlated, and the model trial's text quality has not been blindly adjudicated. The native adapter is an experimental *non-production* artifact with documented unsupported-claim regressions. Do not deploy it for regulated, medical or financial decisions. Tiny Aya's original CC BY-NC and acceptable-use conditions still apply; any commercial exploitation requires separate rights analysis. No tokens, private human judgments, original gated base weights or original raw inference logs are included in the public GitHub source.
