# Original native AlignTune 40-step paired development results — 10 November 2026

**Evidence state:** Owner-uploaded private Stage 13 paired-result ZIP received and structurally checked on 2026-11-10. Exactly 48 matched base and native-AlignTune adapter predictions per arm; original model outputs and private weights **NOT** published. This is a previously inspected **synthetic development** comparison, not a blind final test. **No semantic answer quality, overall safety benefit, original GPU execution, or training-time model revision has been independently attested.**

## Private evidence provenance — do NOT upload raw responses publicly

| Original evidence file | SHA256 |
| --- | --- |
| Owner ZIP, four exact members | `e0319c83fbb87ef8570ca0c42caf68bea4a4fef1f8ae320bbe11e9a7c1761c68` |
| 48 original base predictions | `1910ae570da788b758eab64f3a9e265bdbc4295d094ff36c1af85b84deea4f0a` |
| 48 native40 adapter predictions | `91130d03047c7ec6b9967bfe61f22f30d84797ece7ba991da1e35d61705eecc4` |
| Matched AuditKIT report | `74d8402c85a6d090885e3f8814449087a762c39f9ab7a995b4004b293c9e3311` |
| Original private native40 adapter ZIP | `dcefc5b0babf708aa1b48ac0349ddb3bc4745405c72e9636c5860d7719d8f0e9` |

**Source binding as recorded in the reports:** 48-case development dataset SHA256 `d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c`; base-model inference revision `af89d219b53ed9b13b8a4645f9c8028973510324`; native AlignTune LoRA tensor SHA256 `ba985c1966050764a5220e87ea581a9bd482ba87b055fee4a1e363643f1cc92e`. Both inference passes self-report deterministic generation (`do_sample=false`, `num_beams=1`, `max_new_tokens=160`) on a Tesla T4 using an authenticated local commit-pinned Hugging Face snapshot. This is a **recorded inference-time source check**, not proof of the loaded model's training-time revision or external GPU attestation.

The four ZIP entries have valid CRCs, match the embedded file SHA256 manifest, and contain the exact same 48 unique case IDs in both prediction arms (16 English, 16 Norwegian, 16 Arabic). The saved per-case binary score totals match the archived paired comparison. **A separate raw-prediction-to-gold semantic or strict reference recomputation has NOT been done in this audit**; the following are source-report-consistent diagnostics, not independently certified correctness.

## Measured paired development diagnostics

| Synthetic reference metric | Unchanged base | Native AlignTune 40-step adapter | Observed change |
| --- | ---: | ---: | --- |
| Required JSON schema valid | **14/48 (29.17%)** | **48/48 (100%)** | +34 |
| Strict exact-reference matches | **4/48 (8.33%)** | 2/48 (4.17%) | -2 |
| Exact supported-answer text (39 answerable cases) | 0/39 | 0/39 | 0 |
| Raw expected source ID among answerable (including malformed schema) | 31/39 | **39/39** | +8 |
| Schema-valid correct source ID among answerable | 1/39 | **39/39** | +38 |
| Correct canonical abstention (9 unsupported) | **4/9 (44.44%)** | 2/9 (22.22%) | -2 |
| Unsupported-question substantive claims | 5/9 (55.56%) | **7/9 (77.78%)** | +2 |
| Strict-schema accepted non-abstaining answers | 4/48 | **45/48** | +41 |
| Missing required JSON fields | 33/48 | **0/48** | -33 |

The two strict-match regressions **were both previously correct unsupported abstentions**, not supported-answer knowledge losses. The adapter made no improvements to exact supported-answer text (0/39 on both arms). More valid JSON and expected citations can coincide with more unsupported, potentially misleading claims. Thus **do not claim that this fine-tuning improved factual correctness, safe abstention, or overall AI safety**. The change in structured outputs is real at the level of the saved strict scorer, but citation-ID presence is not factual entailment.

### Language slices (only 16 cases and 3 unsupported per language)

| Language | Strict matches before → after | Valid schema before → after | Correct unsupported abstentions before → after | Unsupported claims before → after |
| --- | --- | --- | --- | --- |
| English | 2/16 → 2/16 | 5/16 → 16/16 | 2/3 → 2/3 | 1/3 → 1/3 |
| Norwegian | 0/16 → 0/16 | 4/16 → 16/16 | 0/3 → 0/3 | 3/3 → 3/3 |
| Arabic | 2/16 → 0/16 | 5/16 → 16/16 | 2/3 → 0/3 | 1/3 → 3/3 |

The Arabic unsupported subset accounts for both lost correct abstentions; **3 cases are too few** for a general claim about Arabic model reliability. Under strict scoring, the tuned model still made zero exact supported-answer matches in every language. Paraphrases may express correct facts despite exact-text mismatch, so this is **not** a semantic failure rate.

## Research and publication decision

**No-go for a claimed model-safety improvement.** Preserve the private original adapter and both inference reports intact, without retraining on the 48, 60 or 72 evaluated cases. Before a model-quality claim, perform blind multilingual *semantic* review of a bounded, representative case sample using fictional policy context and source entailment, including unsupported questions and formatting/paraphrase false negatives. Compare against the unchanged baseline, and develop any abstention or entailment changes on separate training/dev examples—not these frozen cases.

For a Track 2a safety-benchmark submission, these measured findings and the failure mode are defensible **as limited negative research results** when described truthfully. The [public benchmark dataset](https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark) remains its previously published source version, and the public dataset collection remains unchanged. The original raw Stage 13 predictions and LoRA weights remain private. No public model collection or official Lexsi submission is claimed by this report.

### Source-integrity limits

An owner-provided archive can establish internally consistent files and timestamps but cannot independently prove that the GPU executed the recorded runs. The training-time loaded base-model SHA was not independently observed. The 48-case development set was previously inspected; it is **not** a blind benchmark. The paired exploratory statistics are descriptive, with correlated examples and multiple comparisons. No outside reviewer has verified correctness of individual response meanings.
