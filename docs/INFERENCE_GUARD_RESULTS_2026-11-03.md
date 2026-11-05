# Inference-time output guard — measured development-set result (2026-11-03)

**Experiment:** deterministic, fail-closed output-shape and citation-identity guard over **48 previously saved responses** from unchanged CohereLabs Tiny Aya Global. **No new inference, no adapter, no weight changes, no semantic oracle, no fine-tuning.**

**Evidence:** [GitHub Actions CPU execution](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41640840381) (and later reruns). Download the \`policyproof-real-baseline-cpu-evidence\` artifact for:
- \`verified_real_baseline.json\` — exact dataset hash, 48 unique IDs, deterministic metric recomputation.
- \`auditkit_actual_baseline_replay.json\` — official AuditKIT 1.1.3 native precomputed evaluation of 48 saved model responses; fingerprint \`ae297dd84f990566\`.
- \`guarded_development_evaluation.json\` — real before/after deterministic scores and all 48 transformed responses.
- \`submission_gate.json\` — **NOT_READY** for a fine-tuning submission.

Dataset SHA256: \`d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c\`  
Original model revision: **not recorded** in the source report. This prevents exact original-weight reproduction.

## What changed

The deterministic guard accepts ONLY well-formed JSON with keys \`answer\`, \`source_id\`, \`abstain\`. If an output is incomplete, contradictory or cites a source absent from the supplied policy context, the guard returns the canonical \`INSUFFICIENT_EVIDENCE\` abstention. It does **not** consult gold answers, correct source labels or the dataset category to decide whether to accept a response. It cannot determine semantic support.

In the original 48 responses, the guard made the following decisions:

| Outcome | Cases |
| --- | ---: |
| Originally well-formed answer-shaped output retained | 20 |
| Originally valid abstention retained | 8 |
| Incomplete/malformed required schema → canonical abstention | 15 |
| Contradictory abstention fields → canonical abstention | 5 |
| **Total** | **48** |

## Paired deterministic development results

| Metric | Original saved model output | After output guard | What it means |
| --- | ---: | ---: | --- |
| Strict exact end-to-end correctness | 8/48 = **16.67%** | 9/48 = **18.75%** | Improvement of 1 previously inconsistent unsupported response; NOT proof of more factual supported answers |
| Complete three-key schema | 33/48 = **68.75%** | 48/48 = **100%** | By construction, since the guard emits canonical JSON |
| Required-field omissions | 15/48 = 31.25% | 0/48 = 0% | Malformed outputs are not accepted as answers |
| Contradictory abstention fields | 5/48 = 10.42% | 0/48 = 0% | Contradictions normalized to abstention |
| Correct unsupported-case abstention | 8/9 = **88.89%** | 9/9 = **100%** | Denominator is 9 unsupported questions |
| Unsupported-case substantive claim emission | 1/9 = **11.11%** | 0/9 = **0%** | Syntax/contradiction guard fixes this development-set example |
| Accepted cited-source-ID match on answerable cases | 20/39 = **51.28%** | 20/39 = **51.28%** | No increase in well-formed, evidence-attributed supported outputs |
| Exact supported answer content match | 0/39 = **0%** | 0/39 = **0%** | **No demonstrated improvement to content correctness** |
| Accepted answer-shaped responses | 20/48 | 20/48 | 28 responses after the guard are canonical abstentions |

A raw expected source-ID match prior to guarding was 38/39 = 97.44%; after guarding only 20/39 = 51.28% remain in accepted supported outputs. Raw source-ID match on noncompliant outputs is NOT reliable citations. Show both rather than choosing the larger number.

## Source-boundary hardening (November 5, 2026)

The output guard previously authorized a **repeated** policy source ID (or a policy header without any body text) because it counted header strings rather than validating the underlying source blocks. The numeric evidence audit separately excluded duplicate blocks. These two decision procedures could disagree about whether a citation was trustworthy. They now use one parser: only one nonempty pre-[UNTRUSTED-NOTE] policy block per ID counts. Duplicated or empty cited blocks produce `ambiguous_policy_source` and a canonical abstention, while the numeric and topic audits retain the claimed ID as a **human-review risk flag**. A duplicate appearing only *after* the untrusted-note boundary is ignored.

See [dedicated source-boundary security and regression guide](CITATION_SOURCE_BOUNDARY_2026-11-05.md). All 48 baseline responses are replayed on the same original saved data (no new GPU inference); matching a unique source remains **insufficient proof of factual support**.

## Limitations and how to interpret

1. This evaluation was on a **development set repeatedly inspected during debugging**, not a blind holdout. Do not generalize confidence intervals or claim human-level truth calibration.
2. Output-shape compliance of 100% is **tautological** for a canonical JSON validator. It measures integration robustness, not model understanding.
3. Abstaining can conceal some incorrect supported answers as well as sacrifice valid paraphrases. All nine unsupported abstentions pass here, but there are only nine such cases.
4. The 20 retained supported answers were **not** proven factually correct, and none matched the exact reference answer in the strict scorer.
5. This is a separate **inference-time guard experiment**, not a fine-tuned Tiny Aya checkpoint and not an AlignTune training result. It should not be substituted for the Lexsi Track 1 requirement to show a real trained-model improvement.
6. The official rules prefer a chained pipeline of at least two Track 1 tools and a genuine before/after model result; organizer confirmation would be needed before treating a guard-only project as an eligible alternative track submission.

## Reproduce on CPU

\`\`\`bash
python -m pip install "auditkit>=1.0,<2"
python -m scripts.prepare
python -m scripts.verify_evidence \
  --baseline artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json \
  --dataset data/unseen_policy_test.jsonl
python -m scripts.auditkit_replay \
  --report artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json \
  --dataset data/unseen_policy_test.jsonl
python -m scripts.inference_guard \
  --baseline artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json \
  --dataset data/unseen_policy_test.jsonl
\`\`\`

These commands need no CUDA, no Hugging Face token, and no banking data.
