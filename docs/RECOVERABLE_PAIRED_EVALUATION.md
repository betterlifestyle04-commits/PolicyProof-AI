# Interruption-safe paired GPU evaluation and verified resume

**Status (November 4, 2026): Implemented and CPU-tested; NO REAL GPU EVALUATION EXECUTED.** The final 60-case benchmark is still an unreviewed public synthetic candidate, and the project has no verified completed AlignTune adapter. This guide describes how to avoid losing the first 60-case base inference pass when genuine prerequisites become available.

## The problem this solves

The original paired GPU runner waited for base inference, adapted-model inference, and a successful AuditKIT comparison *before saving any of the model's outputs*. A CUDA out-of-memory or adapter-loading failure in the second pass could discard the entire successful base pass. Rerunning unnecessarily consumes scarce accelerator time and risks changing experimental conditions.

The revised runner now writes the **complete 60-case base output as soon as it finishes** using an exclusive-create, fsync-backed JSON writer. It does not save incomplete sets of predictions. Subsequent errors leave the verified base stage on disk and do **not** produce a misleading successful paired evaluation artifact. If AuditKIT scoring fails, the base file remains while no paired result is claimed.

## Default preflight remains free and read-only

```bash
python -m scripts.frozen_pair_inference
```

This checks the frozen synthetic benchmark, actual human-review signoff declaration, and validated full-size adapter metadata. It downloads no weights and runs no inference. **The current project correctly reports BLOCKED.**

## First real GPU attempt (only when prerequisites legitimately exist)

After independent qualified human review, the frozen SHA-256 signoff, genuine saved and validated adapter, full pinned Tiny Aya model revision, and authorized access to weights:

```bash
python -m scripts.frozen_pair_inference \
  --adapter-directory artifacts/align_qlora_smoke \
  --execute --confirm-review --confirm-gpu-run \
  --base-output artifacts/final_base_inference.json \
  --tuned-output artifacts/final_tuned_inference.json \
  --paired-output artifacts/paired_auditkit.json
```

The runner creates **base** JSON immediately after all 60 base predictions complete; then runs the adapted model and AuditKIT comparison. Each output path uses **exclusive creation**, never overwriting an existing artifact. Stored base and candidate reports are generated from actual GPU model outputs *only in a genuine execution*.

## Recovery if the adapted-model pass fails

Inspect the exception, hardware/runtime and output file contents. If, and **only if**, the original base JSON exists and the tuned and paired outputs do not, explicitly authorize a new run using the same paths with:

```bash
python -m scripts.frozen_pair_inference \
  --adapter-directory artifacts/align_qlora_smoke \
  --execute --confirm-review --confirm-gpu-run --resume-base \
  --base-output artifacts/final_base_inference.json \
  --tuned-output artifacts/final_tuned_inference.json \
  --paired-output artifacts/paired_auditkit.json
```

Before any GPU call, `--resume-base` rejects:

- Missing, symlinked, oversized, corrupted, ambiguous or duplicate-key JSON.
- Incomplete case coverage, extra/duplicated IDs, null/raw non-string outputs and changed response text with mismatched saved scores.
- Model identity or pinned revision differences; changed candidate dataset SHA or generation settings; fine-tuned outputs being presented as the base.
- Missing self-declared GPU provenance or timestamp; unsupported human semantic-review claims.
- Changed scores/metrics, since these are recomputed from raw responses and checked **exactly**.
- Existing tuned or paired outputs (which must be inspected, not silently replaced), or attempts to overwrite frozen data, review signoff or the adapter files.

This **does not prove** the underlying GPU run occurred: metadata can be forged by an adversary capable of constructing files. Inspect the original notebook logs and runtime history separately. The resumed run always rechecks the same final-holdout preflight and valid full-size adapter before continuing.

**The resume option can save the first 60-case base pass, not individual unfinished predictions.** A crash halfway through the original base pass still requires a new original-base evaluation. Never resume with different prompts/model revision, never change the benchmark after seeing predictions, and never claim a saved base means completed adapted inference.

## Evidence integrity and limitations

Only the base JSON is persisted on a tuned-model failure. A completed paired result is written only after both complete reports and a successful native AuditKIT replay. If process termination occurs between writing tuned and paired files, the runner will refuse to overwrite them; inspect the artifacts and use the separate offline paired AuditKIT replay rather than generating new predictions unnecessarily.

The software tests use **synthetic stubbed responses in temporary test files**, not a real Tiny Aya model. They validate write-once semantics, dataset/provenance consistency, attempted resume after a simulated second-stage failure, and rejection of altered data; they must never be counted as research observations.

The core project remains **not submission-ready**: genuine GPU training, independent benchmark validation, paired model inference, and public Hugging Face collection requirements are incomplete. See [full paired inference instructions](FROZEN_PAIRED_INFERENCE.md) and [submission readiness](SUBMISSION_READINESS.md).
