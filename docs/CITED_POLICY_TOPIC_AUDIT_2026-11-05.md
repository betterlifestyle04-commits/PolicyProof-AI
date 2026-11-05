# Cited-policy topic mismatch diagnostics — source relevance, not factual grading

**November 5, 2026.** The original 48 synthetic development cases were previously inspected and are **not** a blind final evaluation. This diagnostic uses no new GPU model inference, training or independent human semantic review.

## The real gap in the previous numeric audit

A question asks "What is the overdraft interest rate?" or "Hva er overtrekksrenten?" or "ما فائدة السحب على المكشوف؟". A model asserts "The overdraft rate is NOK 25 daily" and cites [POL-U003], which actually says: "Flex Savings allows three free withdrawals per calendar month; later withdrawals cost NOK 25 each."

NOK 25 is literally present. The existing number-literal checker therefore cannot distinguish this wrong-product substitution. The new source_topic_audit.py separately flags **a question/cited-policy topic mismatch requiring human review**, not a proven hallucination.

## Exact decision procedure

1. Run the conservative JSON and source-ID syntax guard; invalid outputs and proper abstentions receive no topic verdict.
2. Extract only the specifically cited policy block before the first [UNTRUSTED-NOTE]; duplicated source IDs remain ambiguous.
3. Run a small, published multilingual English/Norwegian/Arabic rulebook on the QUESTION and specifically cited policy text independently. It recognizes overdraft, mortgage, investment returns, foreign ATM fees, savings withdrawals, virtual cards, mailing address, contactless, scheduled payments, and new international beneficiaries.
4. Flag only if exactly one topic is recognized for both, and they are different.
5. Matching topics are explicitly classified as **semantics unverified**; missing/multiple topic matches are **incomplete or ambiguous** without automatic factual approval or rejection.

This is intentionally lexical and narrow, with no calibration across real financial documents. False negatives and false positives are both possible. Relevant synonyms, conditions, units, negations, dates and cross-product comparisons may not be properly handled. The tool must NEVER be used to automatically authorize, reject or approve an actual banking answer.

## Reproduce on original saved untuned development outputs (CPU)

Run these commands from the repository root:

    python -m scripts.prepare
    python -m scripts.source_topic_audit

The GitHub CI and public demo build create artifacts/dev48_topic_relevance_audit.json alongside the separate numeric review report. The hosted PolicyProof AI observatory shows both under **Output guard experiment**; **Try the output guard** accepts editable synthetic questions and policy text.

The report's saved-score provenance is checked against the original research data, but topic detection does not use gold expected answers or expected source IDs. Successful review of the code cannot authenticate the original GPU inference execution.

## Owner-only private paired output screening

    python -m scripts.prepare
    python -m scripts.pair_numeric_triage --bundle /path/to/policyproof_dev48_paired_gpu_results.zip

The original ZIP SHA-256 is pinned by default. The git-ignored owner-only report reviews both numeric literals and question/source topic mismatches for the 48 saved base and 48 saved adapted outputs, including newly and no-longer flagged case IDs. The real original archive is NOT stored publicly and has NOT been replayed through this new checker in this change. Tests use synthetic paired ZIP fixtures.

**Do not share the owner-only report or model A/B mapping with human reviewers before independent blind votes are locked.** Fewer flags cannot establish superior factual model quality.

## Claim boundary and related protocol

The checker does not prove source entailment, numeric correctness, model improvement, AlignTune training, human reviewer identity or official submission. The complete separate human semantic A/B review and independently reviewed candidate-final 60-case evaluation remain required.

Run CPU tests with:

    python -m pytest -q tests/test_source_topic_audit.py tests/test_pair_numeric_triage.py tests/test_dashboard_smoke.py

Related: [numeric evidence audit](NUMERIC_EVIDENCE_AUDIT.md) and [independent blinded human review](DEV48_BLINDED_SEMANTIC_REVIEW.md).
