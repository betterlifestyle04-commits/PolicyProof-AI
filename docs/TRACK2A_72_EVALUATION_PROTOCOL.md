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

## Step 2 — Obtain actual model responses elsewhere

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
    "max_new_tokens": 128
  },
  "predictions": []
}
\`\`\`

**Never set** \`is_real_model_result\` to true for oracle, hand-authored, simulated, or fabricated predictions. The current new challenge has **zero** such reports. The scorer will reject this example until all 72 responses and the exact revision are populated.

## Step 3 — Independently check the provenance

Confirm in the inference environment that the downloaded model weights match the recorded Hugging Face revision. Save the machine/GPU, Transformers and tokenizer versions, UTC time, decoding config and inference source commit. The evaluation script can validate the metadata and count, but it **cannot prove** the person who prepared the report really ran inference. That requires external logs/checkpoint attestations.

## Step 4 — Score actual saved outputs with AuditKIT 1.1.3 (CPU)

\`\`\`bash
python -m pip install "auditkit==1.1.3"
python -m scripts.score_track2a_72 \
  --dataset artifacts/challenge72_for_inference/challenge_v1.jsonl \
  --model-report /path/to/real_tiny_aya_challenge72_predictions.json \
  --output-dir artifacts/actual_72_case_evaluation
\`\`\`

The script **will fail closed** unless:
- all 72 predictions are present with unique correct IDs
- output strings are actual raw model responses, not missing/null
- dataset SHA and model identity match
- the full model revision is supplied
- the original base model is used and the inference run is self-declared as real
- real AuditKIT 1.1.3 completes all 72 evaluations

On success it creates a saved deterministic metric report and an actual AuditKIT receipt; it performs **no new inference**.

## Step 5 — Interpret the numbers conservatively

The report separates all six challenge families, each language, unconditional versus conditional rates, and paired exact-success outcomes for the same 12 fictional control questions across the four attacks. A literal attacker marker appearing in an output is **not** a full attack-success metric. A correct citation ID is **not** proof of textual support. Exact text match penalizes valid paraphrases.

**Human review is still necessary** to adjudicate hallucinations, answer paraphrases, translation fidelity and attack success before any final safety-performance claim. To claim genuine improvement, compare clearly separated pre/post systems on the same frozen benchmark and report answer coverage, over-refusal, abstention and factual correctness. Never imply inference-time formatting fixes are model-weight changes.

## Current competition readiness

The Track 2a benchmark + report pipeline is reproducible and the new attack corpus is curated, but the 72 new cases are unscored. Hugging Face public model and dataset collections and official Lexsi submission are still incomplete. The original Tiny Aya model is gated, and publication must respect its access and license restrictions.

- [Track 2a existing 48-case findings](TRACK2A_SAFETY_BENCHMARK_2026-11-03.md)
- [CPU-only HF dataset publication guide](TRACK2A_HF_PUBLICATION_GUIDE.md)
- [Full public source](https://github.com/betterlifestyle04-commits/PolicyProof-AI)
