# Native AuditKIT 1.1.3 replay of the original saved dev48 pair

**State: software implemented; owner has not yet rerun the original private ZIP in this repository.** This optional CPU step addresses the documented gap that the previous 48-case paired GPU ZIP was checked internally but the official AuditKIT evaluator was not independently re-executed in that inspection. The same raw model outputs are used; **no additional GPU generation, training or human semantic grading occurs**.

## Why it matters

An archived report can contain plausible counts and a purported AuditKIT fingerprint. Running `auditkit==1.1.3` again against the **unmodified 96 saved raw responses** and identical 48 questions provides a reproducible software-toolkit check, while the SHA-256 of the original ZIP binds input provenance.

The replay validates, in sequence:

1. The source ZIP equals SHA-256 `e98b06482985a9c2f5dfa62100dccda7421830de8292bae013250ccf4ca1099c` (unless a separately independently authenticated SHA is supplied).
2. ZIP inventory, per-member checksums and original nonfinal development declarations match the expected four-file bundle.
3. Both raw reports have the same 40-character requested Tiny Aya revision, the exact 48-case dataset hash, identical deterministic generation settings, matching adapter SHA-256, complete case IDs and recomputed saved score rows.
4. The original paired comparison's *per-metric baseline and candidate counts* reproduce from the raw outputs.
5. **Only after all input checks:** official AuditKIT 1.1.3 executes two CPU `precomputed` scoring runs, one for each model, then all returned paired counts are checked again.
6. A new report is created **exactly once**, only at `reviews/dev48_native_auditkit_cpu_owner_only.json`, which is Git-ignored.

The output includes the actually executed AuditKIT version, two toolkit fingerprints and the computed counts, plus explicit negations of GPU execution, semantic correctness, final-blind status and independent human review. **The original ZIP is not copied to a public repository or modified.** The private report reveals the base versus adapted model and **must not be shown to blinded human reviewers before their votes are locked**.

## The easiest route: Colab CPU notebook

[Open the native AuditKIT replay notebook](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/08_native_dev48_auditkit_cpu.ipynb)

Select the default CPU runtime; run all cells. Upload the original `policyproof_dev48_paired_gpu_results.zip`, not the training checkpoint ZIP. The notebook first verifies without running AuditKIT, then explicitly opts in to run the evaluator, and offers the private JSON report for download. Keep the downloaded report with the original archive, never with the blinded reviewer CSVs.

## Local shell equivalent

From a fresh repository clone, install only the official CPU evaluator and generate the same synthetic development dataset:

```bash
python -m pip install 'auditkit==1.1.3'
python -m scripts.prepare
# Read-only preflight — proves ZIP and frozen pair consistency, no toolkit execution
python -m scripts.replay_dev48_native_auditkit --bundle /path/to/policyproof_dev48_paired_gpu_results.zip
# Execute the real toolkit on SAVED predictions, save an owner-only report
python -m scripts.replay_dev48_native_auditkit --bundle /path/to/policyproof_dev48_paired_gpu_results.zip --execute
```

An existing report is never silently overwritten. Review the original first; use a distinct owner-only report filename if repeating an experiment. A changed ZIP must not be accepted just by changing a hash until the new data has actually been authenticated.

## Limitations and remaining decision

Running the native toolkits confirms only that the official evaluator reproduces the **deterministic structural scores** on a **previously inspected synthetic development set**. It does not authenticate the training-time GPU steps or exact loaded model weights. It does not establish answer-grounding, real-world reliability, or performance on the separate 60-case candidate final dataset.

A valid source ID might cite the wrong banking rule, and an exact-text mismatch might reject a correct paraphrase. Perform the separate independent blinded 96-response semantic review and 60-case gold-label review before comparing final model quality or changing training decisions.

Official Lexsi initial submissions are due **November 14, 2026 AoE**. An owner-only native AuditKIT report is useful supplementary research evidence, not submission confirmation or a substitute for hosted Hugging Face collections.
