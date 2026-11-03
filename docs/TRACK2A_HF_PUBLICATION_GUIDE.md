# Publish PolicyProof Track 2a dataset safely (CPU-only; owner approval required)

The official Lexsi submit page requires **public Hugging Face model and dataset collections**, all included repositories public and ungated, plus an abstract (<500 words) and a GitHub write-up. It is not enough to link a GitHub ZIP. Track 2a allows an adversarial/safety benchmark plus report, but the generic rules also request before/after model-improvement metrics. We therefore cannot guarantee benchmark-only submission acceptance without organizer confirmation.

## Artifact already prepared

File: PolicyProof_Track2a_HF_Dataset_Upload_Ready_NOT_PUBLISHED.zip (20 items, original SHA256SUMS in provenance/).

This archive contains the 48 **already evaluated** development cases, 72 **new, UNSCORED** synthetic challenges, saved Tiny Aya base outputs, real native CuratorKIT/AuditKIT receipts, hash metadata, a draft HF README and a blank review sheet. It contains **no trained adapter, no original gated model weights and no private HF token**.

Before publication inspect the original ZIP, confirm legal rights, and select your license. The owner must choose from CC BY 4.0 (attribution), CC0 1.0 (public domain dedication, as applicable), or CC BY-SA 4.0 (share alike). This is **not a substitute for legal review**. Do not assume the dataset can be licensed without considering third-party input and model permissions.

## Easiest: CPU-only Google Colab publishing notebook

1. Open [the dedicated HF publishing notebook](../notebooks/05_publish_track2a_hf_cpu_only.ipynb) in Google Colab (File → Open notebook → GitHub; select this project's notebook).
2. Select **CPU runtime**; GPU is not needed.
3. Upload exactly the verified ZIP when prompted, and confirm it passes the strict hash check.
4. Inspect the data, choose and enter a compatible dataset license, then run the offline dry-run.
5. To publish: register or log in to your personal Hugging Face account, generate a **write-scoped** token with access to a new dataset repository, and store it **privately** in Colab **Secrets** as HF_TOKEN. Do not paste a token in code, this chat, a screenshot, email or a public repository.
6. Set HF_DATASET_REPO_ID to the real namespace/name, set the two explicit publication confirmation flags to True, and rerun the final cell.
7. The notebook uploads the dataset, creates the dataset collection and checks that an **unauthenticated** client can read the repo, all staged files, the README and the collection membership. A local JSON receipt is written on success.
8. Open the dataset and collection links in a signed-out browser. Do not set external_access_verified in the GitHub manifest solely because an authenticated upload succeeded.

On a repeated run after a partial failure the publisher **refuses to overwrite an existing dataset repo by default**. Inspect the actual published contents before deciding whether the existing-repo override is appropriate.

## Alternative: trusted local Python terminal

From the repository root, after installing huggingface_hub and obtaining the ZIP:

    python -m scripts.publish_track2a_dataset --archive path/to/PolicyProof_Track2a_HF_Dataset_Upload_Ready_NOT_PUBLISHED.zip --license cc-by-4.0

This is **offline**. To publish, after confirming your rights and privacy, set HF_TOKEN as a private environment variable and use the full command below; replace the namespace/repo ID with your own. The illustrative license is **not preapproved**:

    python -m scripts.publish_track2a_dataset --archive path/to/PolicyProof_Track2a_HF_Dataset_Upload_Ready_NOT_PUBLISHED.zip --license cc-by-4.0 --repo-id YOUR_NAMESPACE/policyproof-track2a-safety-2026 --publish --confirm-rights --confirm-publication --create-collection

The script does not prompt for or print a token. It refuses to publish unless all explicit opt-ins are provided. Only the data repo and data collection are created; **no model repo/collection or Lexsi form submission** is performed.

## Why this does not finish the entire competition submission

The official form also requires a **public, ungated Hugging Face model collection**. PolicyProof has **no verified trained model or adapter**. The referenced Tiny Aya upstream has gated access; we cannot present it as an ungated newly created model. Organizer clarification is needed for a benchmark-only Track 2a project. A drafted question to Lexsi exists in Gmail but is **not sent**.

The 72 extra challenge questions have **never been answered by the model**, and the 48 already measured development cases have no blinded independent review. Do not call the combined 120 cases an evaluation sample.

## Verification links

- Official rules and submit section: https://hackathon.lexsi.ai/
- GitHub project: https://github.com/betterlifestyle04-commits/PolicyProof-AI
- Research report: https://github.com/betterlifestyle04-commits/PolicyProof-AI/blob/main/docs/TRACK2A_SAFETY_BENCHMARK_2026-11-03.md
- Hugging Face upload API: https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
- Hugging Face collections: https://huggingface.co/docs/huggingface_hub/package_reference/collections

**Last day to submit: November 14, 2026 (AoE).** No official Lexsi submission has yet been confirmed.
