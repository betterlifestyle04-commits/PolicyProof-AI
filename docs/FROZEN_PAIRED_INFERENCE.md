# Frozen 60-case paired evaluation: guarded GPU runner

**Current status (November 4, 2026): BLOCKED.** This is executable code for a *future* evaluation on an approved GPU, **not** an experiment that has been run. The currently public 60-case final candidate has not been independently human-reviewed, and no full-size, verified AlignTune adapter exists. The 48 saved Tiny Aya baseline outputs are from a different, inspected development set and **cannot be used as the 60-case base comparator**.

## CPU preflight (works without credentials, model weights, or a GPU)

From a clean Python 3.11+ checkout:

```bash
python -m scripts.prepare
python -m scripts.frozen_pair_inference
```

The default command performs **no model download, model inference, checkpoint mutation, GPU work or publishing**. It checks:

- Frozen 60-case integrity, multilingual balance, source-policy isolation and SHA-256.
- An explicitly completed human-review declaration tied to the exact dataset SHA, **supported by all 60 rows from each of at least two distinct reviewer CSVs, exact file hashes, independently recomputed reconciliation, no negative/incomplete votes, and each reviewer's declared approval**. It still **cannot** independently verify actual human identity, action, independence, language competence or semantic correctness.
- A full-size LoRA adapter with the verified Tiny Aya tensor geometry, SHA-256 receipts, 78-row CuratorKIT training hash, declared completed AlignTune training backend and a pinned full 40-character base-model revision. This remains **artifact consistency and self-reported training**, not independently observed GPU work.

Today it correctly prints `"status": "BLOCKED"`. DO NOT change or forge signoff/adapter files to make it pass. See [human reviewer process](FINAL_REVIEW_WORKFLOW.md) and [adapter verification](TRAINING_ARTIFACT_VERIFICATION.md).

## Actual paired GPU inference: only after BOTH prerequisites genuinely exist

The trained artifact must have been independently reviewed, and the frozen evaluation must have been adjudicated by qualified reviewers *before* testing. Put the real adapter under `artifacts/align_qlora_smoke/adapter` with its untouched `run_info.json` in the parent directory. Preserve original run logs.

On a CUDA-equipped environment with gated Tiny Aya access and dependencies installed:

```bash
python -m pip install torch transformers peft bitsandbytes accelerate "auditkit==1.1.3"
python -m scripts.frozen_pair_inference \
  --adapter-directory artifacts/align_qlora_smoke \
  --execute --confirm-review --confirm-gpu-run \
  --base-output artifacts/final_base_inference.json \
  --tuned-output artifacts/final_tuned_inference.json \
  --paired-output artifacts/paired_auditkit.json
```

**Operator confirmation flags are necessary but never sufficient.** The program refuses execution unless the dataset, review signoff and artifact checks pass. It then executes two separate inference runs with the *same* pinned `CohereLabs/tiny-aya-global` revision, the *same* 60 prompts, and deterministic `do_sample=False`, `num_beams=1`, `max_new_tokens=160`.

- **Base pass:** original 4-bit quantized model without an adapter.
- **Tuned pass:** independently loads the same base revision, then the validated local PEFT adapter. No training code is executed.
- **Evaluation pass:** scores both saved raw-output sets with the actual AuditKIT precomputed interface and the deterministic PolicyProof scorer. It refuses incomplete or mismatched ID coverage and does not write the paired report unless the replay succeeds.
- Each report preserves original raw outputs, model SHA, adapter hash where applicable, dataset SHA, language/case IDs, generation settings, actual GPU run metadata, and explicit limitations.

The process writes new JSON output files using exclusive creation rather than silently overwriting previous evidence. It does **not** automatically modify the human-review signoff or claim improvement. This script **has not yet been run with a genuine checkpoint on CUDA**, so the true runtime memory footprint, API compatibility and model behavior remain unverified.

## Scientific limitations and next stages

Success in this pipeline proves only that a real model-prediction collection and deterministic comparison executed in the runtime. It does not prove safety, factual grounding, statistical significance, reviewer competence, generalization or a gain due to AlignTune.

Do not tune on the 60-case final candidate after seeing results. The original test cases and proposed gold labels are public, so this cannot be presented as a secret blind holdout. Exact-string matching can miss semantically correct paraphrases. Have independent reviewers judge model answers without seeing model identity, report disagreement and matched-case uncertainty, and release the negative cases too.

No current GitHub Actions runner invokes real CUDA inference. CPU CI verifies that the preflight refuses the currently missing prerequisites. **Never alter production metadata to make a test pass or pass off synthetic unit fixtures as trained model outputs.**

## Relevant files

- [Frozen inference source](../scripts/frozen_pair_inference.py)
- [CPU safety tests](../tests/test_frozen_pair_inference.py)
- [Paired AuditKIT evaluator](../scripts/auditkit_compare.py)
- [Final benchmark protocol](FINAL_BENCHMARK_PROTOCOL.md)
- [Submission readiness](SUBMISSION_READINESS.md)


## Interruption-safe base pass and explicit recovery (added November 4, 2026)

The runner now saves the **complete 60-case original-base report as soon as original-base inference succeeds**, rather than waiting for adapted-model loading and the AuditKIT comparison. If the tuned pass fails, preserve this baseline and investigate the failure. After correcting the GPU/runtime issue, an operator may retry with the same arguments plus `--resume-base` (and still `--execute --confirm-review --confirm-gpu-run`). The saved report must pass strict dataset hash, model revision, generation configuration, full case coverage, raw-output score recomputation, and declared GPU provenance checks before a model is loaded. It is never automatically accepted; no file is overwritten. See [the complete interruption/recovery runbook](RECOVERABLE_PAIRED_EVALUATION.md). This feature has **only synthetic CPU safety tests**, and no real GPU experiment has been run.


## CPU-only recovery after a failed AuditKIT comparison

The first and second **complete** 60-case model passes are now persisted separately, before the AuditKIT comparison. If the model passes finish but AuditKIT fails, **do not reload Tiny Aya**: run `python -m scripts.finish_saved_pair` to inspect the current read-only preflight, then (only after checking the original GPU logs and artifacts) opt in with `python -m scripts.finish_saved_pair --finalize --confirm-saved-inference`. This loads and rigorously validates saved full raw-prediction files, the frozen dataset digest, original revision, tuned adapter SHA and deterministic scorers, before performing native AuditKIT precomputed scoring on CPU. It never executes inference and will not overwrite existing outputs. See [the recovery runbook](RECOVERABLE_PAIRED_EVALUATION.md). **No real paired GPU model comparison has yet occurred in this project.**


### Source review proof is mandatory, not optional

For the exact private reviewer-file checks, reconciliation record and signoff schema, see [source-linked review provenance](REVIEW_SOURCE_PROVENANCE.md). A bare signoff JSON with `review_complete: true` and two arbitrary codes is rejected. The default repository correctly remains blocked; no review forms have been filled on the reviewers' behalf.
