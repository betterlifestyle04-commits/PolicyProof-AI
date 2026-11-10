# URGENT NOVEMBER 10 UPDATE — Public release ALREADY completed: DO NOT republish

**The original Stage 14 Section 6 returned a `CalledProcessError` only after its upload and collection-creation operations had successfully created the remote resources.** The anonymous Hugging Face API now shows:

- Public ungated model adapter: https://huggingface.co/shalnark01/policyproof-ai-native40-align-2026
- Actual public model collection: https://huggingface.co/collections/shalnark01/policyproof-ai-lexsi-track-2-model-and-adapter-research-6af39caac578d29e1c66c6
- Public dataset collection: https://huggingface.co/collections/shalnark01/policyproof-ai-multilingual-safety-datasets-6af24cc5b3c20ac76f6ceef5
- Exact model weight SHA256 as published in the public LFS metadata: `ba985c1966050764a5220e87ea581a9bd482ba87b055fee4a1e363643f1cc92e`, identical to the source ZIP.

The already-published model has five intended source-derived files and no gated original Tiny Aya base weights. **Do not rerun the old publication command, do not delete or replace this model repo, and do not upload the source ZIP again.**

**Recovery:** Open the latest [Stage 14](../notebooks/14_owner_native40_hf_release.ipynb), run Section 1 to refresh repo source in a fresh session, then go straight to **Section 7**, which runs `verify_existing_release()` anonymously on the live public model. No Hugging Face login, rights flags, GPU or ZIP upload is necessary for read-only verification. Alternatively check GitHub's [anonymous model verifier workflow](../.github/workflows/verify-public-native40.yml) after merge. Confirm all 5 file SHA256s. Then proceed directly to the official [Lexsi form](https://hackathon.lexsi.ai/submission/) with the real model and dataset collection URLs.

**The earlier Step B instructions below applied only before publication and are SUPERSEDED. They are retained as historical provenance and must not be followed again.** The owner's license/release-rights responsibility remains independent of technical access. No official Lexsi organizer submission is verified.

---

# Lexsi 2026 owner publication — one private adapter, one public model collection

**Status of this file: instruction sheet; publication has not occurred.** Deadline for first-stage Lexsi submission: **15 November 2026, 11:59 PM Anywhere on Earth**. Read [official organizer rules](https://hackathon.lexsi.ai/) and our [research write-up](LEXSI_TRACK2A_FINAL_WRITEUP.md).

## What is already complete

The owner has verified original native AlignTune training artifacts and a paired 48-case development evaluation. The public source repository is [PolicyProof-AI](https://github.com/betterlifestyle04-commits/PolicyProof-AI). The previously verified public dataset collection is [PolicyProof multilingual safety datasets](https://huggingface.co/collections/shalnark01/policyproof-ai-multilingual-safety-datasets), containing the [source-pinned benchmark dataset](https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark). Do not mistake historical anonymous dataset verification for new live publication verification. The native adapter and private raw results have **not** been released.

## Step A — Owner reviews release rights before ANY public action

1. Confirm the exact original ZIP: `policyproof_align_native40.zip`, SHA256 `dcefc5b0babf708aa1b48ac0349ddb3bc4745405c72e9636c5860d7719d8f0e9`.
2. Read [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), its **CC BY-NC 4.0** terms and acceptable-use policy. Confirm your noncommercial research-adapter release is permitted. CC BY-NC is **not** a commercial release grant.
3. Verify you own or are authorized to redistribute the *adapter-only* trained tensors and the aggregate synthetic research measurements. The native adapter **worsened** abstention on unsupported examples (4/9 → 2/9).
4. Examine [the prepared model card](HF_NATIVE40_ADAPTER_MODEL_CARD.md). It specifically prohibits a false safety-improvement claim and lists the gated base-model dependency. **Do not upload the original Tiny Aya base weights**. Do not upload private raw predictions, reviewer judgments, tokens or original Colab runtime credentials.
5. If any condition is uncertain, leave publication OFF and request written clarification from the rights owner or organizer; do not bypass the gate. These are owner decisions the code cannot take for you.

## Step B — Run the owner-only release notebook in a CPU Colab

Open [Stage 14 owner HF release](../notebooks/14_owner_native40_hf_release.ipynb) directly using [this Colab link](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/14_owner_native40_hf_release.ipynb).

**No T4 GPU, model inference, or training needed.**

1. Run Section 1, which fetches current public source code.
2. Run Section 2 and upload **only the original** `policyproof_align_native40.zip`. The original source SHA must match before any further step.
3. Run Section 3; download the **`...HF-PREVIEW-NOT_PUBLISHED.zip`**. It has exactly five model-repository files: PEFT adapter tensors, PEFT config, truthful README model card, aggregate-only `EVALUATION_SUMMARY.json`, and release-checksum manifest. This command has **no network write or authentication**.
4. After Step A and inspecting the preview, personally change the *three* approval flags in Section 4 to `True`. If you cannot approve all three, do **not** proceed to upload.
5. Run Section 5 and sign in through the official `huggingface_hub.notebook_login()` widget to the actual HF dataset-owner account. It needs a **write** token to create a repo and collection. Never paste a token into a code cell, ChatGPT or GitHub.
6. Run Section 6 only after independently approving, and explicitly change `RUN_PUBLIC_HF_RELEASE=True`. The CLI cross-checks your HF username and exact original hash; it creates a new **public, ungated adapter-only model repository**, uploads **only the five allowlisted files**, creates the matching **public model collection**, and checks that all model files and the collection are readable **anonymously** with exact hashes.
7. Save the **actual model repository URL** and **actual model collection URL** printed by the successful command. **Do not invent either URL**. If an upload fails midway, the code will not claim completion; the HF owner must inspect any partial remote repository before retrying.

The model repo slug is intended to be `<verified-owner>/policyproof-ai-native40-align-2026`. The collection URL is assigned by Hugging Face, and **must be taken from the live successful result**, never predicted.

## Step C — Final anonymous checks and official form

- While signed out, verify the model collection opens and contains the public, ungated adapter model repository. The source Tiny Aya base model **itself is gated**, so demonstrate that the *adapter-only repository* is ungated and document the upstream gating honestly.
- Confirm the public dataset repository and **separate dataset collection** open anonymously. The project already has source-hashed dataset evidence, but recheck publication status before submitting.
- Open the public [final write-up](LEXSI_TRACK2A_FINAL_WRITEUP.md), updated current [abstract](TRACK2A_CURRENT_SUBMISSION_ABSTRACT_2026-11-10.md), [GitHub repo](https://github.com/betterlifestyle04-commits/PolicyProof-AI), and optional [public demo](https://policyproof-ai-observatory.onrender.com).
- Complete [Lexsi's official form](https://hackathon.lexsi.ai/submission/) using **your actual registration email**, title **PolicyProof AI**, **Track 2a (adversarial/multilingual benchmark)**, official CuratorKIT and AuditKIT toolchain, supplemental AlignTune trial, and the two *actual* public HF collection URLs. The latest submission before deadline counts.
- Save a screenshot or PDF of the **organizer's actual confirmation**. HF publication, GitHub CI and owner notebook completion **are not evidence of official submission**.

## Release status / STOP conditions

| Requirement | Status at code preparation |
| --- | --- |
| Native adapter original SHA | Known, owner privately uploaded |
| Aggregate research and model card | Prepared; metrics expose safety regression |
| Public model repo created via this script | **NO** |
| Public model collection created | **NO** |
| HF owner signed in/rights approved | **NOT CONFIRMED HERE** |
| Registered Lexsi team email | **OWNER ONLY** |
| Official form submitted | **NO** |

**No additional model training is a prerequisite to delivery.** For Track 2a, the measured research can be honestly presented as an adversarial safety benchmark with documented failure modes; a model-safety-improvement claim is unsupported. The original data and model are subject to their separate license limits.
