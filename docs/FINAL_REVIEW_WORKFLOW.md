# Final holdout: independent multilingual gold-label review

**Status (November 4, 2026): NOT REVIEWED.** This is a reproducible, CPU-only workflow for assembling and reconciling reviewer feedback on the 60-case final *candidate* benchmark. It is not evidence of model inference, fine-tuning, validated translations, or two independent human reviewers.

## Scope and trust model

- 60 fictional-policy cases: 20 English, 20 Norwegian, and 20 Arabic.
- 48 supported answers, 9 unsupported prompts, 3 injection probes. All labels are **proposed**.
- Reviewers see the question, complete fictional context, proposed answer/source, and abstention proposal. They **do not see model predictions**.
- Three verdicts for **each case**: `evidence_support_yes_no`, `translation_faithful_yes_no`, `question_unambiguous_yes_no`. Allowed values: lowercase `yes`, `no`, or blank until judged.
- At least two **different actual humans** should independently review every case, with suitable Norwegian, Arabic and evidence interpretation competence. The validator only checks distinct *codes*, not real identity or language proficiency.
- The public final candidate is **not genuinely blind**; it remains a versioned, held-out development protocol, not proof of generalization to unseen real financial policy.

## Generate reviewer-specific packs (CPU only)

Each reviewer should download a separate packet from the Streamlit **Final benchmark review** page, or run:

```bash
python -m scripts.review_holdout --export reviewer_A \
  --output reviews/reviewer_A.csv
python -m scripts.review_holdout --export reviewer_B \
  --output reviews/reviewer_B.csv
```

These files contain all 60 fictional examples, full proposed labels, source evidence, reviewer pseudonym and repeated SHA-256 of the untouched candidate dataset. **Every verdict is blank.** The script does not generate reviews, model outputs or inference.

Review in a spreadsheet editor or CSV tool. Do not change case IDs, question/context, gold labels, source IDs, dataset SHA, or reviewer code. Write just `yes` or `no` into the three verdict columns, and optional `review_notes` for a proposed correction. For unsupported items, check all supplied policy records to confirm the answer really is missing. For injection items, do not treat attacker notes as real sources. Check translations, negations, units and paraphrases. Reviewers should not view each other's filled packets before completing their own.

## Consolidate the files, without inventing sign-off

```bash
python -m scripts.review_holdout \
  --summarize reviews/reviewer_A.csv reviews/reviewer_B.csv \
  --output artifacts/human_review_reconciliation.json
```

This command checks all 60 cases and the exact original content; rejects changed question text, changed source evidence, changed proposed gold, wrong dataset SHA, missing/duplicate case IDs, unknown verdict values, and repeated reviewer codes; then reports:

- `REVIEW_INCOMPLETE`: some verdicts blank.
- `REQUIRES_CORRECTIONS_OR_ADJUDICATION`: at least one negative verdict or disagreement.
- `READY_FOR_MANUAL_ADJUDICATION_AND_SIGNOFF`: all verdicts are present and unanimous `yes`, but reviewers must still independently sign off.

The JSON includes every affected case/criterion, unanimous agreement percentage, per-reviewer coverage, and file SHA-256s. It always returns `review_complete: false`, `independent_human_identity_verified: false`, and `requires_manual_signoff: true`. Agreement can be perfect even when both humans are wrong.

## What remains manual

Reviewers must resolve disagreements, check proficiency and independence, and explicitly sign their conclusions. If source material, questions or gold answers require any correction, **create benchmark v2 with a new hash before model evaluation**; do not quietly edit the frozen v1 file. Keep the old files as provenance and produce new review packets for the revised version.

Only after human review and adjudication should the separate `docs/final_benchmark_review_signoff.json` be updated with real reviewer information and the exact dataset hash. The script intentionally does **not** update this signoff file or assert that any person reviewed the set. Existing submission checks must not treat an unsigned reconciliation as a completed review.

No model on this final candidate should be prompted, trained, tuned or scored during the above review process. Do not select hyperparameters or stop training based on these labels. The original 48-case saved baseline remains the development diagnostic.

## Practical privacy and evidence hygiene

Use fictional policy records only. Reviewers should use pseudonyms, not emails or full names, in public files. The Streamlit server serves blank CSVs but does not accept or store completed human reviews. Keep filled files private until the reviewers authorize publication. A human review only validates the suitability of a synthetic benchmark; it cannot establish real-world financial advice reliability or model safety.

See [final benchmark protocol](FINAL_BENCHMARK_PROTOCOL.md) and [submission readiness](SUBMISSION_READINESS.md).

## New: original reviewer files are now mandatory for GPU review signoff

The guarded paired evaluator no longer accepts `review_complete: true` and two pseudonyms alone. The reviewer must retain **private, SHA-256-linked copies** of both full 60-case annotated CSVs and an unmodified reconciliation report under `reviews/`. Before accepting review completion, `python -m scripts.verify_review_provenance` now rechecks the full CSVs against the original frozen benchmark bytes and requires all 180 judgments per reviewer to be affirmative, no disagreements or negatives, exactly matching reconciliation content, and explicit individual manual approval declarations with declared Norwegian/Arabic/English competence across the review team. `reviews/` is gitignored.

**This remains consistency validation, not proof of human independence, language expertise or correct semantics**. Never generate positive votes automatically or mark signoff complete merely to pass CI. All existing actual human work must be retained privately; the public repository remains **NOT REVIEWED**. For the exact v1 review-evidence schema and instructions, see [source-linked review provenance](REVIEW_SOURCE_PROVENANCE.md).
