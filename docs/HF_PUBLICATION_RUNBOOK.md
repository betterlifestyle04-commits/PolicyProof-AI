# Hugging Face publication runbook — PolicyProof AI (2026-11-03)

**Status:** Nothing here proves that an HF repository has been created or is judge-accessible. The dataset, logs and model card templates are ready in GitHub; the **actual trained adapter does not yet exist**. Do not publish an empty model repo or claim a successful fine-tuning result.

The official Lexsi rules require public Hugging Face artifacts for datasets, adapters/models and logs, plus an HF Article. Publication requires the owner's own approved Hugging Face account and credentials. Never paste an HF token into chat, a notebook cell, README, or GitHub Issues.

## 1. Prepare a private authenticated environment

Install the current Hugging Face CLI on a computer/notebook where you own the account:

\`\`\`bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf auth whoami
\`\`\`

Use a **fine-grained write token** with only the required repository permissions. Keep it in the account's secure credential store. The gated base model must already be approved for the same HF user who registered for Lexsi.

Substitute \`YOUR_HF_USERNAME\` with your real HF namespace in the examples. Do not publish to someone else's namespace.

## 2. Dataset repository — can be completed without GPU

Only publish user-owned, permissioned fictional data; confirm the exact dataset license before publishing.

\`\`\`bash
hf repos create YOUR_HF_USERNAME/policyproof-curatorkit-v1 --repo-type dataset --exist-ok
hf upload YOUR_HF_USERNAME/policyproof-curatorkit-v1 templates/huggingface/dataset_README.md README.md --repo-type dataset
hf upload YOUR_HF_USERNAME/policyproof-curatorkit-v1 policyproof_curatorkit_train_v1.zip data/policyproof_curatorkit_train_v1.zip --repo-type dataset
\`\`\`

The ZIP must be **the original user-provided CuratorKIT file**, not a new/generated lookalike. Its SHA256 must equal \`c6340969501965bf0bbfe8fa9247f41150b59e999b4881e78a27f16a233b8948\`. The Alpaca training JSONL inside must match \`4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c\`.

**Important:** The dataset card is a draft: select/verify a license you are authorized to grant before making the dataset public. Do not automatically assert the Tiny Aya model license applies to independent original synthetic dataset text.

## 3. Logs and original-model baseline — can be completed without GPU

Get \`policyproof-real-baseline-cpu-evidence.zip\` from the latest passing GitHub Actions run. This contains:
- \`verified_real_baseline.json\` (recomputed original saved response metrics)
- \`auditkit_actual_baseline_replay.json\` (genuine official AuditKIT 1.1.3 replay)
- \`guarded_development_evaluation.json\` (separate output-validator development experiment)
- \`submission_gate.json\` (NOT_READY)

\`\`\`bash
hf upload YOUR_HF_USERNAME/policyproof-curatorkit-v1 artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json logs/original_tiny_aya_baseline.json --repo-type dataset
hf upload YOUR_HF_USERNAME/policyproof-curatorkit-v1 policyproof-real-baseline-cpu-evidence.zip logs/policyproof-real-baseline-cpu-evidence.zip --repo-type dataset
\`\`\`

These are saved-model inference records and CPU evaluations, **not** a trained-model evaluation. The original model-revision SHA is missing; disclose it.

## 4. Adapter repository — DO NOT EXECUTE BEFORE TRAINING SUCCEEDS

Only after a controlled GPU run completes, you have **adapter_config.json**, nonempty **adapter_model.safetensors**, a run receipt showing at least 10 completed optimizer steps, all SHA256 hashes and a truthful description of the actual backend:

\`\`\`bash
hf repos create YOUR_HF_USERNAME/policyproof-tiny-aya-global-lora --exist-ok
hf upload YOUR_HF_USERNAME/policyproof-tiny-aya-global-lora LOCAL_ADAPTER_DIRECTORY . 
hf upload YOUR_HF_USERNAME/policyproof-tiny-aya-global-lora templates/huggingface/adapter_README.md README.md
\`\`\`

**Before uploading** replace the explicit "NOT A TRAINED MODEL" disclaimer with real evidence, without inventing results. Keep noncommercial base-model license and acceptable-use obligations; check that the base model permits your intended adapter distribution. If the adapter was trained using the direct PyTorch fallback, say so; do **not** claim the AlignTune trainer completed.

## 5. HF Article and final URLs

Use [HF research article draft](HF_ARTICLE_DRAFT.md) as the basis for a Hub Article. It contains verified baseline and output-guard results and explicitly lists missing tuned-model work. It is **not** a successful fine-tuning report. Complete the tuned-model section only when metrics and screenshots are verified.

Create/fill \`docs/hf_publication_manifest.json\` with the actual public URLs and set \`external_access_verified\` true only after opening them logged out or in a fresh browser. For the final public entry, link:
1. Public GitHub code and exact commit
2. Public HF dataset and logs, plus adapter if trained
3. Article and, if applicable, an optional HF Space
4. Reproducibility commands and limitations
5. Genuine original vs tuned-model metrics and their denominators

Official references: https://hackathon.lexsi.ai/ and https://huggingface.co/docs/huggingface_hub/guides/cli

**Do not enter any false URLs to satisfy the readiness gate.**
