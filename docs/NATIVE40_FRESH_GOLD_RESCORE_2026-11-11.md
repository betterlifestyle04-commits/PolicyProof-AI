# Independent strict gold rescoring of the original native AlignTune development experiment

**Audit performed:** 11 November 2026, entirely offline on the owner's original private four-file Stage 13 results ZIP and frozen published 48-case gold dataset.

**What is new:** Prior [source-consistency report](TRACK2A_NATIVE40_DEV48_OBSERVED_PRIVATE_RESULTS_2026-11-10.md) only compared saved per-case score fields with archived aggregates. This new audit *re-executed the published deterministic scoring rubric* directly on **all 96 original raw responses** (48 base + 48 adapter) against the immutable 48-case gold dataset. **Every saved per-case score field matched the fresh calculation**, including required-field validity, citation identifiers, strict exact match, unsupported-abstention and raw unsupported-claim detection.

This verifies deterministic reference-score reproducibility. **It does not constitute independent human semantic correctness, model execution attestation or a blind final evaluation.** The 48 cases were previously inspected and synthetic.

## Original source hashes and reproducibility

| Evidence | Original SHA256 |
| --- | --- |
| Owner private Stage 13 ZIP | `e0319c83fbb87ef8570ca0c42caf68bea4a4fef1f8ae320bbe11e9a7c1761c68` |
| Frozen 48-case gold dataset | `d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c` |
| Original base predictions report | `1910ae570da788b758eab64f3a9e265bdbc4295d094ff36c1af85b84deea4f0a` |
| Native AlignTune adapter predictions report | `91130d03047c7ec6b9967bfe61f22f30d84797ece7ba991da1e35d61705eecc4` |
| Native LoRA tensor file | `ba985c1966050764a5220e87ea581a9bd482ba87b055fee4a1e363643f1cc92e` |

On CPU, check out the exact repository source and supply the original *private* Stage 13 ZIP:

```bash
python -m scripts.rescore_original_native40_dev48 \
  --bundle /path/to/policyproof_native40_dev48_paired_gpu_results.zip \
  --expected-sha256 e0319c83fbb87ef8570ca0c42caf68bea4a4fef1f8ae320bbe11e9a7c1761c68 \
  --output artifacts/native40_fresh_gold_rescore_aggregate.json
```

The command rejects modified ZIPs, mismatched gold datasets, unpaired raw outputs and altered saved per-case scores. Output is **aggregate-only**, write-once, and never contains the original private raw predictions, personal secrets or model weights. [Stage 15 Colab](../notebooks/15_native40_fresh_gold_rescore_and_blind_human_review.ipynb) provides the same owner workflow without a GPU.

## Fresh paired outcomes (48 matched cases)

| Metric | Original base | Native40 adapter | Beneficial / adverse case switches | Exploratory exact McNemar p |
| --- | ---: | ---: | ---: | ---: |
| Complete JSON schema | 14/48 | **48/48** | 34 / 0 | ≈1.16×10⁻¹⁰ |
| Strict exact match | **4/48** | 2/48 | 0 / 2 | 0.5 |
| Exact supported-answer text | 0/39 | 0/39 | 0 / 0 | 1.0 |
| Expected source ID in raw supported answer | 31/39 | **39/39** | 8 / 0 | 0.0078125 |
| Valid-schema source ID in supported answer | 1/39 | **39/39** | 38 / 0 | ≈7.28×10⁻¹² |
| Correct abstention on unsupported questions | **4/9** | 2/9 | 0 / 2 | 0.5 |
| Unsupported-question claim (lower is safer) | **5/9** | 7/9 | 0 / 2 | 0.5 |

P values are **exploratory**, conditional on changed matched cases, uncorrected for multiple endpoints, and not independent safety certifications. The strong evidence here concerns **JSON formatting**, not better factual answers. The unsupported-abstention regression is a small-sample descriptive risk, not a statistically established population-level effect.

### By language (16 cases each, only three unsupported per language)

| Language | Valid JSON: base → adapter | Correct unsupported abstentions: base → adapter |
| --- | --- | --- |
| English | 5/16 → 16/16 | 2/3 → 2/3 |
| Norwegian | 4/16 → 16/16 | 0/3 → 0/3 |
| Arabic | 5/16 → 16/16 | 2/3 → 0/3 |

**Critical distinction:** strict supported-answer text remained 0/39 in both arms. The model may have returned semantically correct paraphrases that exact-string matching rejects; that possibility **has not** been judged by an independent human. Conversely, the right citation ID is not proof of source entailment. Do not claim that 0/39 is a human-measured factual error rate or that 39/39 source IDs mean 100% grounded accuracy.

## Next research milestone — genuine model-blinded human review

The project includes [private blinded paired semantic review tooling](NATIVE40_DEV48_HUMAN_REVIEW_HANDOFF.md) and a new owner-facing [Stage 15 notebook](../notebooks/15_native40_fresh_gold_rescore_and_blind_human_review.ipynb). It can export 96 responses with balanced randomized A/B slots, without original model labels, gold answers or source case IDs. The owner-only mapping is stored separately, never on public GitHub/HF.

**Human review has NOT been completed.** The owner needs at least two genuinely independent qualified reviewers to judge semantic correctness, evidence support and unsafe/injection-following behavior, with disagreements adjudicated by a separate human. No synthetic or AI-produced reviewer votes should be presented as human verification.

The 60-case private final benchmark and 72-case adversarial challenge remain frozen and excluded from training. Additional fine-tuning on these already-inspected cases would contaminate the experiment; no new GPU training is needed for the current Track 2a submission.
