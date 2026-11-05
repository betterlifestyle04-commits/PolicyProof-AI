# Citation-local numeric evidence audit — exploratory, CPU only

**Status:** Implemented for PolicyProof's fictional policy research. It is a **review prioritization diagnostic**, not an independent accuracy grader or new model training.

## The remaining substantive problem

The paired 40-step adapter development results have strong schema and source-ID scores but do **not** establish answer correctness. For example, a response can cite POL-U003 (a savings withdrawal fee) and still invent an overdraft interest fee using its NOK 25 value. Citation-ID correctness alone cannot establish that a rule answers the user's specific question.

The new audit works only on **saved responses** and the exact fictional source context. It first applies the existing structural output guard. For syntactically accepted, non-abstaining answers, it extracts numeric literals from the answer and the single *cited* [POL-...] evidence block, rather than searching all policy blocks. Source text after [UNTRUSTED-NOTE] is excluded. It recognizes Arabic-Indic numerals, normalized thousands separators, times and small English number words in source text.

- A number absent from its cited block is a **review flag**.
- A number present in its cited block is **not proof** of the correct financial product, time unit, condition, or answer.
- An answer with no numeric literals is **not cleared for factual safety**; the rule cannot check nonnumeric facts.
- Neither the question gold answer nor the expected source ID is passed to the numeric checker. Case IDs and languages are retained only for reviewer triage.

## Reproduce the saved untuned baseline diagnostic

From a repo clone with Python and the pinned source files:

    python -m scripts.prepare
    python -m scripts.numeric_evidence_audit \
      --report artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json \
      --dataset data/unseen_policy_test.jsonl \
      --output artifacts/dev48_numeric_risk_audit.json

Or execute the existing CPU build:

    python -m scripts.build_public_demo

Both commands run without a GPU, a Hugging Face token, model downloads, new inference, new training, or external LLM judgment. The archived baseline report was already generated previously. The audit verifies the dataset fingerprint and all 48 prediction IDs before creating a report.

In the public **Output guard experiment**, a separate section displays the saved-model numeric-risk count and flagged case IDs. In **Try the output guard**, select "Invented numeric fee (needs review)", then click **Validate response**. The structural guard will accept the valid JSON and known citation, but the numeric evidence diagnostic flags the invented NOK 100 amount because it is missing from the cited source. This demonstrates why syntax success must not be misrepresented as semantic safety.

## Optional: private base-versus-trained numerical triage from the original paired ZIP

The public dashboard contains **only the saved original baseline** because the 40-step adapter's raw GPU output archive has not been committed to GitHub. To audit **both** models from your original paired-result ZIP (without publishing it), use:

    python -m scripts.prepare
    python -m scripts.pair_numeric_triage \
      --bundle /path/to/policyproof_dev48_paired_gpu_results.zip

This requires the audited source ZIP checksum (pinned by default), verifies the four-member archive and independently recomputes all 48 raw response score rows for both model reports before writing any private triage. The output is **reviews/dev48_numeric_triage_owner_only.json**, a Git-ignored and exclusive-create file. It contains the two sets of numeric review flags, case-level deltas and explicit limitations. Neither original ZIP nor adapted model is modified. No GPU work occurs.

**Critical blinding safeguard:** Keep this owner-only report confidential until two independent reviewers have completed their 96-row A/B semantic judgments. Showing base/tuned names or relative flag counts during review would compromise model blinding. This report is only a triage aid and cannot substitute for full independent review.

## Where the heuristic cannot help

A model response claiming **"overdraft interest is NOK 25 daily"** while citing POL-U003 will probably pass this numeric *literal* match, because the source does mention NOK 25—in a completely different sense: savings withdrawal fees. That result is intentionally still labeled **semantics unverified**, never **correct**. Unit and subject entailment require actual independent human judgment or a separately validated semantic system.

Similarly, spelled-out Arabic dual forms, relative paraphrases, calculated equivalents, comparisons, conversions and negations are beyond this basic number matching. Literal mismatches may be legitimate paraphrases. False negatives and false positives are both possible. The algorithm has **no calibrated precision, recall or external test validation**. Do not use it as a financial decision authority, production banking security guarantee or evidence of model generalization.

The next required experiment remains the separate blinded A/B human review of 48 saved development cases, followed by the independently reviewed 60-case candidate final comparison. Never tune on that final set or report 48-case diagnostics as final-blind performance.
