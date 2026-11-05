# Canonical cited-policy boundary audit — November 5, 2026

**Scope:** Synthetic banking-policy research and deterministic output validation. This is **not** an authenticated financial-document retrieval service and does not prove a banking response is correct. No model training, GPU inference, independent human review or external verification is performed here.

## The failure we closed

Previously the inference-time guard accepted a source if its identifier matched any bracketed `[POL-...]` header before `[UNTRUSTED-NOTE]`. Another component—the citation-local numeric auditor—rejected duplicate policy IDs. That discrepancy allowed an output to be called `valid_supported_shape` by the main guard even if two contradictory policies shared the same ID. A header lacking a policy body also passed source-ID membership.

An ambiguous source ID is not reliable evidence. A correctly formatted answer can still be unsupported even if its source ID is recognized.

## New unified trust boundary

`scripts/inference_guard.py` owns a single **non-model** policy-block parser. `policy_source_ids`, the guard, and the numeric and multilingual topic auditors now rely on the same source-block inventory.

The parser:
- recognizes policy headers only at the beginning of lines, with horizontal whitespace immediately after `] `, and only before the first `[UNTRUSTED-NOTE]` marker;
- returns only source IDs that occur **once** and have non-whitespace policy text before the next source heading or untrusted boundary;
- explicitly identifies duplicate or empty-block IDs as **ambiguous**, including an ID duplicated with identical text;
- rejects a citation to an ambiguous ID with a canonical `INSUFFICIENT_EVIDENCE` abstention and action `ambiguous_policy_source`;
- retains legitimate unique/nonempty source IDs even when another unrelated ID is ambiguous;
- lets the numeric and question/topic triage tools flag ambiguous citations for human investigation while exposing the original claimed ID, not an approved source;
- ignores any apparent `[POL-...]` headers after an untrusted-note marker, even if they repeat a legitimate ID.

All of these are **syntactic** conditions. An attacker who can edit text *before* the trusted/untrusted boundary can still forge legitimate-looking policy text. Real banking applications must retrieve provenance-authenticated documents from a separate secured store and verify access permissions. The parser cannot prove meaning, amounts, units, conditions, product relevance, or financial correctness. The verdict `valid_supported_shape` specifically means the JSON is valid and the ID refers to one nonempty synthetic policy block—not that the answer follows from it.

## Try it in the judge demo

In <https://policyproof-ai-observatory.onrender.com>, select **Try the output guard**, then select:

1. **Duplicate cited policy IDs** — two policy headings with the same `POL-DEMO-1` ID, one conflicting.
2. **Empty cited policy text** — the policy ID appears but its content is blank.

Click **Validate response**. Both should show an explicit human source-review risk and return canonical JSON abstention.

A clean single-policy response should remain `valid_supported_shape`; the same source header repeated only *after* the untrusted-note boundary should not invalidate the real pre-boundary source.

## Reproduction without GPU

Run from the repository root:

```bash
python -m scripts.prepare
python -m pytest -q tests/test_inference_guard.py tests/test_numeric_evidence_audit.py tests/test_source_topic_audit.py tests/test_dashboard_smoke.py
python -m scripts.inference_guard --baseline artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json
python -m scripts.numeric_evidence_audit
python -m scripts.source_topic_audit
```

All recorded 48-case model-output experiments are deterministic **replays** of previously saved original-model outputs. These tests include duplicate matching and contradictory source text, empty blocks, malicious untrusted-note content, malformed headers, mixed valid/invalid policies, and cross-auditor agreement.

## Strict claim boundaries

The 48-case results have been repeatedly inspected; they are not a blinded final benchmark. Genuine human semantic assessment, native AlignTune training evidence and the independently reviewed 60-case final evaluation are all outstanding. Passing the software tests does not replace those steps.
