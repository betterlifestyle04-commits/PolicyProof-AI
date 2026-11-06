# How to evaluate the new 72-case Track 2a challenge — only with actual model outputs

**Current state:** 72 multilingual adversarial cases have been structurally curated, but **no model outputs exist for these new cases**. This document is a future inference/evaluation protocol. It is not a claim of a completed experiment. The original 48-case results are separate.

## Why measure this set?

The 72 cases are 12 matched fictional policy-language base questions, each measured with:
- one clean control
- four distinct prompt-injection variants (role spoofing, fake policy source, encoded instruction, language override)
- plus 12 additional unsupported queries to study abstention

Four fictional source policies are reused across variants, so examples are **correlated** rather than 72 independent topics. Do not claim a statistically representative real-world safety sample.

## Step 1 — Generate and freeze the exact dataset (CPU only)

Run from repository root with Python 3.11:

\`\`\`bash
python -m scripts.generate_track2a_challenge --output-dir artifacts/challenge72_for_inference
\`\`\`

This writes \`challenge_v1.jsonl\`, exactly 72 rows. It does not call an LLM.

**Expected frozen SHA-256:** \`cdd5cf4e54caa6c79ce1f0cd5bd0f05344f0dc22039a2ad56a3c1c0ee2d32cd1\`. The evaluator refuses any changed bytes. Keep this file read-only. Do not train on this challenge set.

## Step 2 — Use the guarded unchanged-base Tiny Aya GPU runner

The repository now contains a reproducible execution script at `scripts/challenge72_gpu_inference.py`. This is **prepared code only**: we have not executed the 72-case inference on CUDA, and its existence or passing CI does not create original model outputs. Use a GPU runtime where your Hugging Face account legitimately has gated Tiny Aya access, and where you have reviewed the applicable license and compute costs.

Start in the checked-out repository with Python, the matching challenge file, and your *actual* full 40-character Git commit SHA for the authorized model weights (never guess a revision):

```bash
python -m scripts.challenge72_gpu_inference \
  --dataset artifacts/challenge72_for_inference/challenge_v1.jsonl \
  --revision FULL_REAL_40_HEX_MODEL_SHA
```

**Default is CPU-only.** It checks the exact frozen 72-case SHA, all six language/category distributions, pinned requested revision and a new safe output path. `READY_FOR_EXPLICIT_GPU_EXECUTION` means only that local prerequisites are structurally present—not that your GPU, permissions, token, download or runtime will work.

Only when you decide to incur actual model inference, execute:

```bash
python -m scripts.challenge72_gpu_inference \
  --dataset artifacts/challenge72_for_inference/challenge_v1.jsonl \
  --revision FULL_REAL_40_HEX_MODEL_SHA \
  --output artifacts/track2a_72_unchanged_base_predictions.json \
  --execute --confirm-gpu-run --acknowledge-gated-model-license
```

The code reuses the guarded unchanged-base 4-bit Transformers loader from the final-60 pipeline. It **checks the commit SHA observed on the loaded model object before generating**; a requested revision alone is not enough. It generates all 72 prompts without fine-tuning and with deterministic decoding (`do_sample=False`, `num_beams=1`, `max_new_tokens=160`). It checks that the original challenge has not changed, validates the complete 72-output report and writes it exactly once under `artifacts/`. If any precondition, GPU call, source hash or output check fails, no completed report is claimed. Repeated runs cannot overwrite saved predictions.

The complete inference file is a locally recorded GPU result, **not independently authenticated GPU provenance or human semantic accuracy**. Preserve the full private CUDA console logs, runtime versions, model SHA, source commit, original report and failure trace if any. Never upload tokens or gated weights.

**Alternative external runner:** a report produced by another authorized runtime is acceptable for scoring only if it supplies all real raw predictions, the exact dataset SHA, UTC timestamp, deterministic generation configuration, and actual reported loaded-model commit/GPU metadata. The scorer checks internal consistency but cannot authenticate an external claim.

## Previous manual report schema (for inspection only)

A GPU inference run is needed to generate 72 raw outputs with **CohereLabs/tiny-aya-global**. This has not been executed on the new challenge here. When a permitted GPU becomes available, generate each prompt with \`policyproof.core.make_prompt(row)\` and save the raw model response string for every row. Pin the full 40-hex-character Hugging Face base-model revision before inference. Use deterministic settings and record \`generation_config\`, \`created_utc\` and exact dataset hash. Do not regenerate or modify gold labels after seeing model outputs.

Save the following actual inference report as JSON. The example shows only the schema; the \`predictions\` list must contain **72 actual model-generated responses**, each unique and with the correct case ID, before running the evaluator.

\`\`\`json
{
  "engine": "transformers",
  "is_real_model_result": true,
  "is_finetuned": false,
  "model": "CohereLabs/tiny-aya-global",
  "model_revision": "FULL_40_HEX_COMMIT_SHA_REQUIRED",
  "dataset_sha256": "cdd5cf4e54caa6c79ce1f0cd5bd0f05344f0dc22039a2ad56a3c1c0ee2d32cd1",
  "created_utc": "REPLACE_WITH_REAL_UTC_TIMESTAMP",
  "generation_config": {
    "do_sample": false,
    "num_beams": 1,
    "max_new_tokens": 160
  },
  "actual_gpu_inference_metadata": {
    "gpu": "REPLACE_WITH_REAL_CUDA_DEVICE",
    "torch_version": "REPLACE_WITH_REAL_TORCH_VERSION",
    "loaded_model_revision_observed": "SAME_FULL_40_HEX_SHA_OBSERVED_ON_LOADED_MODEL",
    "loaded_model_revision_verified": true
  },
  "predictions": []
}
\`\`\`

**Never set** \`is_real_model_result\` to true for oracle, hand-authored, simulated, or fabricated predictions. The current new challenge has **zero** such reports. The scorer will reject this example until all 72 original responses, the exact SHA, actual loaded-revision GPU metadata and valid timestamp are populated. Hand-authoring these fields is not an authorized replacement for executing inference.

## Step 3 — Independently check the provenance

Confirm in the inference environment that the downloaded model weights match the recorded Hugging Face revision. Save the machine/GPU, Transformers and tokenizer versions, UTC time, decoding config and inference source commit. The evaluation script can validate the metadata and count, but it **cannot prove** the person who prepared the report really ran inference. That requires external logs/checkpoint attestations.

## Step 4 — Score actual saved outputs with AuditKIT 1.1.3 (CPU)

\`\`\`bash
python -m pip install "auditkit==1.1.3"
python -m scripts.score_track2a_72 \
  --dataset artifacts/challenge72_for_inference/challenge_v1.jsonl \
  --model-report artifacts/track2a_72_unchanged_base_predictions.json \
  --output-dir artifacts/actual_72_case_evaluation
\`\`\`

The script **will fail closed** unless:
- all 72 predictions are present with unique correct IDs
- output strings are actual raw model responses, not missing/null
- dataset SHA and model identity match
- the full model revision is supplied
- the original base model is used and the inference run is self-declared as real
- real AuditKIT 1.1.3 completes all 72 evaluations

On success it writes two immutable CPU scoring reports under a **new, direct** `artifacts/<name>/` directory, followed by an `EVIDENCE_COMPLETE.json` receipt. This last marker binds the SHA-256 of both scored files, the unchanged frozen benchmark and the original saved model prediction report. If a write fails, the marker is absent and the folder must **not** be treated as a completed evaluation; no existing evidence is overwritten.

The scorer refuses duplicate-key or non-finite JSON, symlinked or oversized source files, and a changed dataset or original report during scoring. Existing output directories cannot be overwritten. Results also include a 24-case breakdown for English, Norwegian and Arabic, alongside the attack-family analysis. Each language slice is small and correlated, so an apparent favorable number does not certify multilingual safety or factual correctness.

This command performs **no new model inference**, training, human grading, or Hugging Face publication. Original generation receipts remain self-declared until verified independently.

## Step 5 — Interpret the numbers conservatively

The report separates all six challenge families, each language, unconditional versus conditional rates, and paired exact-success outcomes for the same 12 fictional control questions across the four attacks. A literal attacker marker appearing in an output is **not** a full attack-success metric. A correct citation ID is **not** proof of textual support. Exact text match penalizes valid paraphrases.

**Human review is still necessary** to adjudicate hallucinations, answer paraphrases, translation fidelity and attack success before any final safety-performance claim. To claim genuine improvement, compare clearly separated pre/post systems on the same frozen benchmark and report answer coverage, over-refusal, abstention and factual correctness. Never imply inference-time formatting fixes are model-weight changes.

## Current competition readiness

The Track 2a benchmark + report pipeline is reproducible and the new attack corpus is curated, but the 72 new cases are unscored. Hugging Face public model and dataset collections and official Lexsi submission are still incomplete. The original Tiny Aya model is gated, and publication must respect its access and license restrictions.

- [Track 2a existing 48-case findings](TRACK2A_SAFETY_BENCHMARK_2026-11-03.md)
- [CPU-only HF dataset publication guide](TRACK2A_HF_PUBLICATION_GUIDE.md)
- [Full public source](https://github.com/betterlifestyle04-commits/PolicyProof-AI)
