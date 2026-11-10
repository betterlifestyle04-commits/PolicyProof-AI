# Stage 12 — observed 72-case private research results (10 November 2026)

**Evidence state:** User-supplied private original-response ZIP inspected; original file hashes and all 72 saved predictions verified; deterministic source scores independently recomputed. **GPU execution not independently attested. Human semantic review not complete.**

This report contains **only aggregate metrics**. It does not upload or publish the original raw responses, checkpoints, tokens, model weights or private GPU logs. The existing [public dataset](https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark) still contains the original **unscored** 72-case challenge; this private follow-up result has not been published there.

## Provenance and checksums

| Original source | SHA256 |
| --- | --- |
| User-supplied six-member Stage 12 ZIP | f86d1002a162ceac44270cdb25edf2769ee471d469faa158f0ad1719efc65d57 |
| Frozen 72-case JSONL | cdd5cf4e54caa6c79ce1f0cd5bd0f05344f0dc22039a2ad56a3c1c0ee2d32cd1 |
| Original 72 model outputs | ea50e13376183538b1ec25f8c8265f54633602573f1c3b19ca15ced5ccea0a69 |
| Native AuditKIT receipt file | 256b260188acd1669d147c6ee1612dec40cb8d4e46b82d56a62bc83524c8ec10 |
| Measured 72-case score report | c79231e81438c49d545cf4b65c5d6ab9ad0456bd026be0bbf5723d97c0a41fd8 |

The six-file ZIP matches its original embedded completion receipt hashes. Exactly 72 distinct model-output IDs match all 72 frozen source rows. The model report self-records CohereLabs/tiny-aya-global, revision af89d219b53ed9b13b8a4645f9c8028973510324, Tesla T4 GPU, unchanged base model, deterministic decoding, and source code commit 8df6c021f91fd313e028b6991557f5aba89857b9.

A separate CPU implementation recomputed all **overall, language, attack-family and matched-control/attack strict score totals** from the original saved outputs. The counts exactly match the measured report. This secondary environment did not run a fresh native AuditKIT 1.1.3 evaluation because that distribution was not installed; the archive's native AuditKIT receipt was hash-checked. Original execution is still **self-reported runtime evidence**, not external GPU attestation.

## Original unchanged-base results

| Strict synthetic-reference metric | Original result |
| --- | ---: |
| Exact reference match across all examples | **11/72 (15.28%)** |
| Schema-valid output | **58/72 (80.56%)** |
| Correct supported-case citation (strict JSON) | **28/60 (46.67%)** |
| Raw expected source ID, irrespective of schema | **45/60 (75.00%)** |
| Exact supported-answer string | **5/60 (8.33%)** |
| Canonical abstention on unsupported questions | **6/12 (50%)** |
| Non-abstaining content on unsupported questions | **6/12 (50%)** |

**Strict exact match by language:** English 9/24 (37.5%), Norwegian 2/24 (8.33%), Arabic 0/24 (0%). Unsupported canonical abstentions: English 4/4, Norwegian 2/4, Arabic 0/4.

**Do not confuse exact-match with semantic correctness.** Numerically plausible answers such as "14 calendar days" may fail against "Within 14 calendar days" while conveying the same fact. Arabic zero exact matches in this small sample is **not** evidence of zero Arabic factual accuracy. Independent bilingual human grading is required.

All four adversarial families have just 1/12 strict matches, but **clean controls also have 1/12**, so those numbers do not prove that the attacks succeeded. Literal override-marker echoes were zero; attacks can still affect structure or meaning without echoing those markers. The 48 attacked variants reuse only **12 underlying answerable task stems from four fictional policy documents**.

## Paired deterministic output-guard comparison

Same saved raw outputs, same frozen dataset; **no model re-inference, training, change to model weights or use of reference labels by the guard**. The gold labels are consulted only after guarded predictions are produced.

| Metric | Original | With guard |
| --- | ---: | ---: |
| Schema-valid outputs | 58/72 | **72/72** |
| Strict exact reference matches | 11/72 | **16/72** |
| Correct unsupported abstentions | 6/12 | **11/12** |
| Unsupported-case non-abstaining content | 6/12 | **1/12** |
| Correct supported-case citations | 28/60 | **28/60** |
| Exact supported-case answer strings | 5/60 | **5/60** |
| Accepted non-abstaining responses | 37/72 | **29/72** |

All five additional exact matches were **unsupported-case abstentions**. The guard did not improve supported-answer semantic or exact correctness and reduced accepted answer coverage by eight responses. It still let **one unsupported claim** through, demonstrating that a valid source-ID shape does not establish factual entailment. The improvements in schema and abstention are **postprocessing outcomes, not model training or a certified prompt-injection defense**.

**By-language exact matches (before → after):** English 9→9, Norwegian 2→3, Arabic 0→4. The gains in Arabic are from unsupported-case abstentions, **not supported-answer performance**.

## Scientific follow-up priorities

1. Preserve the original private ZIP, T4 runtime logs and checkpoint records without modification. A locally consistent model report cannot prove real accelerator execution on its own.
2. Blindly examine a controlled multilingual sample of near-correct numeric answers, invalid or missing JSON fields, contradictory abstention flags and unsupported claims. Use bilingual reviewers; do not relabel exact matches as verified semantics.
3. Investigate the **one remaining unsupported claim that passed the guard**. Source-ID existence is not proof that a passage entails the response; assess this separately from formatting.
4. Do not retrain on the frozen 72 test cases or retroactively modify their gold answers to improve the reported score. Develop improvements on disjoint training/dev data, then run a separately designated evaluation.
5. The historical submission abstract still states that the 72 set was unscored. **That historical draft must be revised before submission** to reflect these private source-checked findings; the repo cannot infer that a ZIP uploaded in chat has been released publicly.

**No native AlignTune training, independent semantic correctness, HF model collection, or official Lexsi submission is established by this source report.**
