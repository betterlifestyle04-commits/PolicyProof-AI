# Final 60-case paired uncertainty diagnostics

**Status, November 6, 2026: prepared, NOT executed.** No independently validated final 60-case base-vs-tuned experiment exists in the public repository. This module does not train a model, run GPU inference or verify semantic improvement.

## Why matched cases matter

The same 60 fictional-policy examples must be scored for unchanged and fine-tuned models. An overall score gain can hide Norwegian or Arabic regressions. Supported citations must use supported-only denominators; abstention and unsupported claims must use unsupported-only denominators.

The CPU-only `scripts.final_pair_diagnostics` module re-scores preserved raw predictions. For each metric and each language/category slice, it records eligible case count, baseline/tuned success counts and rates, beneficial percentage-point difference, improved/regressed/unchanged matched cases and exact two-sided McNemar/binomial p-value for *discordant* cases.

Metrics are strict exact match, JSON schema validity, supported exact answer, supported citation-ID match, unsupported correct abstention and unsupported claim (where lower is better).

The optional Wilson 95% interval is only for the **fraction of favorable changes among discordant cases**, NOT for the full accuracy difference. A p-value of 1 with zero discordance is **not proof of equivalence**. These exploratory p-values are **not corrected for multiple comparisons**, and small language/category groups have little statistical power. Every report explicitly records `statistical_superiority_established: false`, `multiple_testing_adjusted: false`, and `independent_human_semantic_accuracy_verified: false`.

## Evidence verification and dashboard

On a future genuine paired run, AuditKIT embeds `case_matched_exploratory_diagnostics` in `artifacts/paired_auditkit.json`. The Track 1 source-linked release gate then recomputes this entire block from all 120 preserved raw answers, the unchanged 60-row frozen dataset and the deterministic scorer. A forged subgroup score, contradictory p-value or missing summary fails the release gate even if the aggregate counts remain intact.

The public dashboard **Judge verification** tab displays final matched-case statistics only if the full source-linked paired evaluation validates. Until genuine native AlignTune training, qualified blind gold-label review and paired model outputs exist, the page says that **final results are unavailable**. Saved 48-case development outputs must never be substituted for final model evidence.

Exact strings, citation IDs and JSON formatting are diagnostics, not proof of semantic correctness. Public synthetic cases are not a hidden real-world sample, and these statistics do not establish safety, causality or real-world model improvement. Human blind review of model answers remains necessary. Do not claim GPU attestation, independent reviewers, Hugging Face publication or organizer submission based on these outputs alone.
