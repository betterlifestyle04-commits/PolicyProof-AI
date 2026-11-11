## Solo-builder route (updated 11 November): no external reviewer required

The owner is working alone with an AI assistant and has no independent reviewers. **That is permitted: Lexsi teams can have one member, and Track 2a problem statement #2 accepts a benchmark dataset and report.** Our original strict 96-response score replay, source-pinned public HF collections, and [AI-assisted explicit-currency case analysis](NATIVE40_AI_ASSISTED_EXPLORATORY_CASE_ANALYSIS_2026-11-11.md) have all been prepared without fabricated human votes.

**Do not spend the remaining days searching for reviewers or retraining on frozen examples.** The optional private reviewer packet is **NOT** required for a truthful first-stage Track 2a submission. Never claim independent human safety verification.

Only remaining owner-only tasks: (1) confirm the noncommercial release rights and that the public URLs open under your account; (2) perform a brief view of the [judge overview](JUDGES_START_HERE.md) and [public demo](https://policyproof-ai-observatory.onrender.com/); (3) enter the exact registered email and checked real HF collection URLs in the [official Lexsi form](https://hackathon.lexsi.ai/submission/); (4) save the real organizer confirmation. **Do not submit until you choose to do so. Deadline 15 Nov 2026 11:59 PM AoE.**

---

## 11 NOVEMBER — Scientific audit upgraded; optional blinded review ready

A fresh **raw-prediction-to-frozen-gold** recalculation independently reproduced all 96 archived strict scores: [research audit](NATIVE40_FRESH_GOLD_RESCORE_2026-11-11.md). This does **not** establish semantic answer correctness; original model outputs can be correct paraphrases that exact matching rejects. A [private blinded human-review packet](NATIVE40_DEV48_HUMAN_REVIEW_HANDOFF.md) and [Stage 15 Colab](../notebooks/15_native40_fresh_gold_rescore_and_blind_human_review.ipynb) are available if you can recruit **genuine qualified independent reviewers** before the deadline. Do not fake review or retrain on the frozen sets.

**This optional review is NOT needed to submit an honest Track 2a benchmark with documented failure modes.** Public model/dataset collections are technically verified; the official Lexsi form and owner's publication-rights responsibility remain. Do not delay a first valid submission merely to claim uncompleted reviews.

---

## NOVEMBER 10 — RELEASE STATUS CORRECTED AFTER COLAB ERROR

The first owner publication command returned a `CalledProcessError` **after** both public Hugging Face resources were created. Read-only HF API checks confirmed the public ungated original native adapter model and **the corresponding public model collection with the correct model**. The adapter's public LFS SHA256 equals the original owner ZIP's adapter tensor SHA. The [publication manifest](hf_publication_manifest.json) is updated with the exact URLs. Do **not** rerun the old Stage 14 publication.

**Your remaining personal steps:** (1) optionally inspect the tokenless [read-only verification](../notebooks/14_owner_native40_hf_release.ipynb) Section 7 or GitHub Action; (2) verify publication terms/release rights; (3) complete the [official Lexsi form](https://hackathon.lexsi.ai/submission/) with the registered email, [real public model collection](https://huggingface.co/collections/shalnark01/policyproof-ai-lexsi-track-2-model-and-adapter-research-6af39caac578d29e1c66c6), [public dataset collection](https://huggingface.co/collections/shalnark01/policyproof-ai-multilingual-safety-datasets-6af24cc5b3c20ac76f6ceef5), [final write-up](LEXSI_TRACK2A_FINAL_WRITEUP.md) and [under-500-word abstract](LEXSI_TRACK2A_ABSTRACT_AFTER_PUBLIC_MODEL_RELEASE.md); (4) retain the actual organizer confirmation. **No more training, dataset upload or model publication is required.**

**The pre-publication instructions below are historical and superseded by this section.** Official form delivery has NOT been confirmed.

---

# Lexsi Track 2a — Final owner checklist and form fields

**As of 10 November 2026: this document is a delivery checklist, not a submission receipt.** Official deadline: **15 November 2026, 11:59 PM Anywhere on Earth (AoE)**. See [Lexsi official rules](https://hackathon.lexsi.ai/). The official initial submission does **not** require a video or slides; these are requested of shortlisted teams after 16 November. The latest valid submission before the deadline counts.

## What is genuinely complete

- [x] Public GitHub repository: https://github.com/betterlifestyle04-commits/PolicyProof-AI.
- [x] Source-checked 78-example CuratorKIT train evidence and 40-step **self-reported** native AlignTune adapter with independently checked saved tensor structure.
- [x] Original 72-case source-consistent challenge evaluation and paired native40 48-case development scores, documented without invented semantic accuracy.
- [x] CuratorKIT and AuditKIT both used for Track 2, plus supplemental native AlignTune experiment. Do **not** claim actual SafeTune or CircuitKIT execution.
- [x] Public benchmark dataset (previously verified): https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark.
- [x] Previously public dataset collection: https://huggingface.co/collections/shalnark01/policyproof-ai-multilingual-safety-datasets.
- [x] Public research write-up: [LEXSI_TRACK2A_FINAL_WRITEUP.md](LEXSI_TRACK2A_FINAL_WRITEUP.md), the GitHub README, and aggregate source-hashed experiment reports.
- [x] Adapter-only public model card: [HF_NATIVE40_ADAPTER_MODEL_CARD.md](HF_NATIVE40_ADAPTER_MODEL_CARD.md), with CC BY-NC 4.0/AUP caveat and **abstention regression**.
- [x] Owner-only [no-GPU publication notebook](../notebooks/14_owner_native40_hf_release.ipynb) prepared OFF by default.
- [ ] **Owner rights/terms approval** for PUBLIC, ungated publication of a noncommercial research adapter and aggregate results.
- [ ] **Public HF model repository and model collection** created, with all model-repository files and SHA256s anonymously verified.
- [ ] **Fresh live public dataset/model collection checks**, after the owner provides the real model-collection URL.
- [ ] **Official Lexsi form submitted**, organizer confirmation saved.

## Personally complete these owner-only steps (in order)

1. **Review release rights.** Read [Tiny Aya Global's upstream terms](https://huggingface.co/CohereLabs/tiny-aya-global), CC BY-NC and Cohere's AUP. Confirm you may release **this adapter-only** trained research result publicly and that you own the generated aggregate results. If unsure, do not publish and ask the rights holder/organizer. Model *base weights* must never be uploaded.
2. **Open [Stage 14 in Colab](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/14_owner_native40_hf_release.ipynb)**. This is a **CPU-only** release, not training. Run Sections 1–3 and upload the original `policyproof_align_native40.zip`, SHA256 `dcefc5b0babf708aa1b48ac0349ddb3bc4745405c72e9636c5860d7719d8f0e9`. Download and inspect the private preview.
3. **Only with your consent**, set the three rights/public-approval flags `True`, log in using Hugging Face's official secure widget with a write-capable token for the **same dataset-owning HF account**, and enable `RUN_PUBLIC_HF_RELEASE=True` in Section 6. The script checks the exact hash, model card, login account, public visibility, public file hashes and collection membership. Do not paste the token in chat.
4. **Copy the ACTUAL public HF model collection URL** from the script's successful output and send that link here. I can then update the repo's publication manifest and run anonymous readback verification; do not guess a collection URL from its title.
5. **Check anonymous access** to model collection, model repo, dataset collection, dataset repo, GitHub README and final write-up. Source Tiny Aya may remain gated; the *adapter repo* must be public and ungated. Verify the original training adapter is **not** misrepresented as having safer abstention.
6. **Open the [official submission form](https://hackathon.lexsi.ai/submission/)** and enter the registered team's own email, actual public HF model/dataset collection links, current under-500-word abstract, research report URL, GitHub source URL, tools, track and problem statement. Verify all links and text before clicking Submit. The owner must confirm the identity and final legal-release decision.
7. **Save a screenshot of the organizer's submission confirmation**. Upload it here (with any private email hidden) if you want a final delivery audit. GitHub CI success and HF repo upload are not an official confirmation.

## Form field handoff (fill only genuine facts)

| Lexsi form question | Enter |
| --- | --- |
| Project | **PolicyProof AI** |
| Track | **Track 2 — AI Red Teaming, Safety and Alignment; Track 2a — Adversarial and multilingual safety evaluation** |
| Problem statement | **2a #2 Adversarial and safety evaluation sets**; supplementary **#4 multilingual safety** |
| Registered email | **YOU MUST ENTER THE EXACT REGISTERED EMAIL** — never guess it |
| Core track toolkits | **CuratorKIT** (synthetic curation) + **AuditKIT** (benchmark evaluation) |
| Supplemental toolkit | **AlignTune** (original 40-step LoRA experiment) |
| Base model | **CohereLabs/tiny-aya-global**, source revision requested `af89d219b53ed9b13b8a4645f9c8028973510324` |
| GitHub repo | https://github.com/betterlifestyle04-commits/PolicyProof-AI |
| HF dataset collection | https://huggingface.co/collections/shalnark01/policyproof-ai-multilingual-safety-datasets |
| HF model collection | **PASTE THE ACTUAL URL PRODUCED BY VERIFIED RELEASE; NOT YET AVAILABLE** |
| HF dataset repo | https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark |
| Write-up/article | https://github.com/betterlifestyle04-commits/PolicyProof-AI/blob/main/docs/LEXSI_TRACK2A_FINAL_WRITEUP.md |
| Live demo (if requested) | https://policyproof-ai-observatory.onrender.com — verify it's accessible before including |
| Short abstract | [After-publication draft](LEXSI_TRACK2A_ABSTRACT_AFTER_PUBLIC_MODEL_RELEASE.md), **only after actual model release**; [private/current draft](TRACK2A_CURRENT_SUBMISSION_ABSTRACT_2026-11-10.md) otherwise |
| Claims | Honest format improvement but **unsupported-question abstention regression**; no independent semantic safety improvement demonstrated |

## What not to claim or upload

**Do not** claim a safe production-ready model, pass an exact-string score as human fact-checking, claim SafeTune/CircuitKIT was used, imply independent GPU attestation, or include private original model outputs, reviewers' notes, passwords, HF tokens or base Tiny Aya weights. The 48 development cases are already inspected, the 72 challenge cases are correlated, and the 60 private candidate has no completed independent human review.

**Do not submit a fake model-collection URL.** If public adapter release is legally blocked, raise a Lexsi Discord issue requesting confirmation that the dataset-and-report Track 2a alternative can satisfy the model-collection field. Preserve the deadline and submit only a truthful entry.
