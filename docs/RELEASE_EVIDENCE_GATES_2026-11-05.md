# PolicyProof AI — evidence-bound submission decision gates

**Updated November 5, 2026.** This is a live code-level protocol, **not** a submission receipt or a claim of successful model generalization.

**Tracks:** (1) Tiny Aya multilingual fine-tuning/data curation, and (2a) safety/evidence challenge curation and evaluation. Their eligibility and scientific claims are deliberately kept separate.

## The danger this change addresses

Previously, the Track 1 local submission gate accepted a hand-edited HF publication manifest with booleans such as external_access_verified: true and public_items_ungated_verified: true plus plausible URLs. It also accepted a stand-alone paired AuditKIT JSON with the expected status, count and adapter hash; the underlying saved model outputs could be missing. A final-review boolean plus two pseudonyms could similarly mark a human-review advisory as passed.

Those were **false-positive readiness paths**. The new gate refuses these shortcuts. The legitimate CuratorKIT, AuditKIT, 40-step PyTorch fallback experiments and 48-case paired development evidence remain unchanged.

## Track 1: evidence needed for each release claim

| Check | What the local validator actually requires | What it cannot independently prove |
| --- | --- | --- |
| Untuned model baseline | All 48 saved model predictions, exact input SHA-256 and recomputed score rows | Which GPU/session originally generated the saved outputs |
| AuditKIT baseline replay (supporting) | Same 48-case dataset hash, model identity/revision, replay metadata and negative claims on inference/training | AuditKIT execution in an independent runtime; separately observed CI is stronger evidence |
| Trained adapter | Structurally valid PEFT safetensors plus validated receipt, pinned revision and eligible trainer claim | That optimizer steps truly ran or eligible AlignTune training was the source |
| 60-case independent gold-label review | Exact frozen benchmark, source reviewer CSVs, full declared yes/no vote matrix, recomputed reconciliation, manual declarations and language coverage | Reviewer identity, competence, independence or actual human actions |
| Final 60-case paired comparison | Preserved untuned + adapted 60-case raw reports; same benchmark SHA, base revision and generation settings; matching validated adapter SHA; recomputed per-case scores and every paired result delta | Human semantic improvement or authenticity of self-declared GPU execution metadata |
| Public Hugging Face artifacts and collections | Fresh manifest-SHA-linked anonymous HF Hub API receipt, accessible model/dataset collection entries, adapter repo and logs repo membership | License rights, model relevance, immutable third-party attestation or organizer acceptance |
| Repository/write-up | README and Track 1 submission draft present | That the submission has been filed or read by judges |

### Saved baseline and AuditKIT receipt safety

The CPU baseline verifier now reads the 48-case source evidence through a bounded, non-symlink JSON parser that rejects duplicate field names and non-finite numbers (including deeply nested fields). The original 48-case dataset must also be a bounded regular file; score rows, prediction records, and numeric metric types are checked before accepting a replay. The submission gate applies the same strict parsing to the saved baseline AuditKIT receipt, preventing contradictory duplicate status keys from appearing complete. These checks **do not authenticate the original GPU inference session**, establish answer correctness, or create any new model outputs. Tests intentionally use synthetic source JSON; no real metrics are newly measured.

The **human-review evidence** now participates in the Track 1 mandatory gate. This is intentionally stricter than whether independent review is a literal eligibility rule: PolicyProof cannot ethically claim validated final benchmark results while treating proposed labels as adjudicated truth.

A local gate result of **READY_FOR_MANUAL_REVIEW** means it is ready for human release review, not submitted, proven safe, or a winner. An attacker who can modify all repository files can still forge a consistent set of saved receipts. The code cannot turn self-declarations into independently authenticated human identities or GPU provenance.

## Track 2a remains a separate route

The Track 2a checker requires successful CuratorKIT/AuditKIT work on **saved 48-case base responses**, the separately curated **unscored 72-case challenge**, public research write-up and independently checked public HF collection access. It does **not** claim that the 72 challenges have been answered, that a model has been fine-tuned with AlignTune, or that a model collection is relevant to a benchmark-only entry. Organizers should be asked which model collection is acceptable for Track 2a.

The same release dashboard shows both tracks side by side, but it does not combine them into a fictional success.

## Reproduce locally (CPU only)

From the repo root:

    python -m scripts.prepare
    python -m scripts.build_public_demo
    python -m scripts.submission_gate --output artifacts/submission_gate.json
    python -m scripts.track2a_readiness --output artifacts/track2a_readiness.json

The Track 1 `--output` option now accepts only a **new, direct `artifacts/<name>.json` filename**. It creates the report exclusively and refuses existing files, symlink targets, traversal, public docs, benchmark sources and human-review records. For a new run, choose another filename (for example `artifacts/submission_gate_second_check.json`); do not erase old research evidence to re-run a gate. The public-demo builder detects the earlier CI-generated `artifacts/submission_gate.json`, recomputes the entire readiness report, and **reuses it read-only only when every field matches**. A stale, malformed, or altered report blocks the build instead of being silently overwritten. Add `--strict` to receive a nonzero exit code while incomplete. Both default commands are nonpublishing, no GPU, no new Tiny Aya inference and no model training.

On the [public observatory](https://policyproof-ai-observatory.onrender.com), open **Judge verification → Release decision ledger** to see the exact conditions that pass or block each track. The downloadable JSON report is a snapshot of local structural checks only, not external verification.

## New official native 40-step training pathway (November 5)

[Standalone Colab Stage 11](../notebooks/11_native_aligntune_40step_t4.ipynb) and the [native T4 protocol](NATIVE_ALIGNTUNE_40STEP_T4_2026-11-05.md) now implement a distinct 40-step **AlignTune wrapper.train()** route with the original pinned 78 training examples. This is **not executed** by CPU CI, has not produced a real saved 40-step AlignTune adapter, and is not yet an eligible Track 1 artifact. The code refuses the stale 10-step max_steps override, FP16/BF16 GradScaler, changed training source, partial optimizer steps, overwritten output and unobserved loaded model revisions in local readiness candidate labeling. The older 40-step direct-PyTorch fallback remains separate and cannot be renamed as native AlignTune.

## Training export correctness gate (November 5)

Native AlignTune candidate archives now require matching reported TRL trainer-state steps **and** AlignTune's returned `TrainOutput.steps`, a matching exact-three-member adapter-only ZIP layout, and explicit atomic archive publication metadata. A private temporary ZIP is validated for member set and CRC before it is linked without replacing any preexisting file; local training status is `packaging` until publication completes. CPU tests simulate disk failures, ZIP corruption and a concurrent writer, and ensure a failed export never becomes a published successful ZIP. These safeguards do not themselves establish that optimizer steps truly executed on a GPU or that the resulting answers are semantically correct.

## CPU-only native training interruption gate (November 5)

A new read-only `scripts.reconcile_native_training_evidence` audit distinguishes interrupted Colab sessions, failed or incomplete packaging, fully consistent local and archived receipts, and ZIP publication followed by failed local receipt synchronization. It compares real ZIP safetensors/hashes and the original local adapter, never changes artifacts, and refuses to equate a self-reported local receipt or published ZIP with independently proven T4 optimizer execution. The Stage 11 notebook includes a diagnostic cell that runs without invoking any training. Owner review is mandatory for any discrepancy.

## Highest-impact remaining work

1. **Track 1:** Obtain a checkpoint produced by an eligible, demonstrated AlignTune optimizer run (the manual PyTorch 40-step fallback cannot be relabeled AlignTune). Then independently review the frozen 60 cases before any final model scoring.
2. **48-case development evidence:** Use the separate private blind A/B packet and at least two real competent reviewers. This cannot replace the frozen candidate-final evaluation.
3. **Track 2a:** Publish only the audited permitted research dataset and documentation under a verified license. Do not upload a trained model that does not exist; resolve the collection requirement with organizers.
4. **Both tracks:** Verify anonymous HF collection access, license rights, judge usability, and capture the official submission confirmation. A hosted UI is not itself a submission.

The release gate is designed to **fail closed** on missing model output, unverifiable access, incomplete reviewer evidence and tampered paired score counts. Continuous integration tests use synthetic fixtures to validate the checks, not to manufacture research evidence.

## Related controls

- [Human source review verification](REVIEW_SOURCE_PROVENANCE.md)
- [Frozen 60-case paired evaluation protocol](FROZEN_PAIRED_INFERENCE.md)
- [Separate Track 2a release status](TRACK2A_REPRODUCIBLE_HANDOFF.md)
- [Pinned 48-case paired development results](DEV48_PAIRED_GPU_RESULTS_2026-11-05.md)
- [Full paired human review workflow](DEV48_BLINDED_SEMANTIC_REVIEW.md)
