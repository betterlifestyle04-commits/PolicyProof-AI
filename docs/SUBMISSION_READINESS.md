# Lexsi 2026 submission readiness — archival notes updated Nov 5

> **November 5 update (supersedes stale Nov 3 progress details below):** A user-provided 40-step PyTorch LoRA fallback and saved 48-case GPU paired development results now exist; this is **not** demonstrated successful native AlignTune training and **not** human-verified factual improvement. The [new strict release-evidence protocol](RELEASE_EVIDENCE_GATES_2026-11-05.md) requires actual source-linked 60-case review, preserved paired base/tuned reports with recomputed scores, and a fresh manifest-matched anonymous Hugging Face receipt. No final 60-case evaluation, public HF collection verification, or official submission is currently verified. The previous Nov 3 narrative below is archival and must not be used to describe the latest training state.



## November 7 — stronger Track 2a source-linked release gate

The alternative Track 2a release checker now refuses to declare the 48-case CuratorKIT/AuditKIT chain or the new 72-case curated challenge complete merely because their mutable summary JSON files say `status=complete`. It reads and validates original source files and tool receipts instead.

- **48 original model responses:** require the exact pinned benchmark SHA-256, all original raw responses with matching IDs, real CuratorKIT input and export labels, checksums and native AuditKIT report, matching model identity and **recomputed category, language and overall scores**. All evidence paths must be bounded, regular files with no symlinks or duplicate JSON fields.
- **72 new unscored adversarial cases:** require the original frozen dataset SHA-256, actual CuratorKIT input/export records with complete case IDs and reference labels, an exact export digest, all six category counts and all three language counts. The manifest must still state **no new model predictions, no evaluation and no human review**.
- **CI and reproducibility:** the real-toolkit CI workflow recomputes both sets from the source data before it can call Track 2a research-ready. A status-only fake test fixture is explicitly **RESEARCH_INCOMPLETE** even when its JSON claims success.
- **Immutable readiness receipts:** `python -m scripts.track2a_readiness --output artifacts/track2a_readiness.json` creates only a new direct artifacts JSON report or reuses a byte-identical one. It cannot overwrite a benchmark, private review, prior experiment or existing differing receipt.

This is a **CPU-only local consistency gate**, not proof that the original GPU actually ran or that model answers are correct. Public Hugging Face collections remain unpublished/unverified; original model access restrictions and organizer eligibility still require human decisions. Track 1 native AlignTune training and independent final review remain blocked. The official submission is not confirmed.

**Track:** 1 — Fine-tuning and Data Curation  
**Project:** PolicyProof AI · multilingual evidence-grounded synthetic policy QA  
**Initial submission:** **November 14, 2026** (official site says end of day; organizer email previously specified 23:59 AoE).  
**Demo Day:** November 22, 2026.  
**Status:** **Not ready to submit as an experimentally successful fine-tuning result.** No validated tuned adapter or before/after improvement yet.

Official rules: https://hackathon.lexsi.ai

## Requirements and actual evidence

| Requirement | State | Evidence / next action |
| --- | --- | --- |
| Hackathon registration | User reported registered; email previously checked | Keep the confirmation email, match Hugging Face username |
| Cohere eligible open-weights model | **Verified** | User accepted gated Tiny Aya Global license; 3.35B model ran in Colab T4 |
| CuratorKIT data curation | **Verified** | v0.1.22, 78/78 synthetic examples, 26 each EN/NO/AR; [curation evidence](CURATION_2026-11-03.md) |
| AlignTune integration | **Partially verified** | v0.1.28 imported; TRL trainer factory validated; quantized PEFT model loaded with 252 Linear4bit modules and 3,833,856 trainable parameters |
| AlignTune real successful training | **Blocked** | Initial attempts ended before optimizer step at BF16 GradScaler. Last direct PyTorch fallback failed earlier during tokenizer-to-tensor conversion; fix committed but not T4 re-executed |
| AuditKIT | **Verified on real saved baseline** | Official AuditKIT 1.1.3 replayed all 48 saved Tiny Aya base responses in GitHub Actions; fingerprint `ae297dd84f990566`. This is precomputed evaluation, not tuned-model evaluation |
| Multi-repo Track 1 chain | **Not fully demonstrated** | Need an executed CuratorKIT→AlignTune→AuditKIT path with honest evidence; direct PyTorch fallback cannot be called completed AlignTune training |
| Base-model evaluation | **Verified** | 48/48 actual unchanged Tiny Aya inferences; 16.67% strict exact outcome, 68.75% JSON schema, 8/9 correct unsupported abstentions; see [baseline report](BASELINE_2026-11-02.md) |
| Held-out final test | **Candidate created; human review pending** | [60-case new fictional policy candidate](../data/final_holdout_v1.jsonl), 20 cases per EN/NO/AR; not used for training or scoring |
| Trained-model improvement | **Not verified** | Needs actual trained checkpoint and paired model outputs on a frozen reviewed set. A separate deterministic output guard improved schema/abstention on saved development responses, but this is **not** fine-tuning |
| HF artifacts | **Not published** | Create permitted Hugging Face model/adapters/data/log repositories and add verified hashes, license and model card |
| Public project/demo | **GitHub repo available** | Research dashboard code exists; public hosted site and real judge access not yet verified |
| Submission/write-up | **Draft in repository** | Fill real metrics only after evaluation. Submit through official process; verify receipt |

## Compute access update — November 3, 2026

- The owner's Google Colab session hit the **GPU usage limit**; the dialog explicitly says it cannot connect to a GPU backend. Google provides no fixed reset time.
- The original CuratorKIT ZIP **passed verification again (78 rows)**, but the model restoration stopped at the CUDA availability guard. **No fine-tuning checkpoint has been completed.**
- Prepared a separate [Kaggle GPU notebook](../notebooks/04_kaggle_gpu_fallback.ipynb) targeting offered GPU T4 x2 or P100 hardware; execution, compatible drivers, and trainer success **not yet verified**.
- While GPU access is absent, proceed with CPU-only structural validation, baseline provenance review, independent bilingual gold-label review and submission completeness. Do not replace live-model evidence with mock output.
- Free/paid GPU services have quotas, restrictions and availability variations. Prefer available free quota or the organizer's confirmed eligible credits before making a purchase.

## Official submission form fields verified November 3, 2026

The organizer's current [Submit](https://hackathon.lexsi.ai/) section requires:
- An abstract **under 500 words** ([247-word truthful draft](LEXSI_SUBMISSION_ABSTRACT.md) is ready)
- A separate **public, ungated Hugging Face model/adapter collection** — **MISSING**
- A separate **public, ungated Hugging Face dataset/log/benchmark collection** — **MISSING**
- An article or write-up; the official page accepts a GitHub README if HF Articles are unavailable
- Public GitHub repository, explicit toolchains, chosen tracks, and problem statement

A public website alone does not fulfill the two Hugging Face collection fields. The release gate now checks for actual collection URLs and explicit external/ungated verification. The field verification was done from the organizer's live page; the final form URL and completed submission are still unverified.

## New verified CPU baseline / output guard — November 3, 2026

- [Real saved-model baseline JSON](../artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json) is now committed. Its 48 responses and dataset SHA-256 were verified on CI.
- Actual AuditKIT 1.1.3 `precomputed` evaluation completed with 48 saved Tiny Aya responses; see [GitHub Actions run](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41639380727).
- A separately measured [inference-time guard](INFERENCE_GUARD_RESULTS_2026-11-03.md) improved **JSON schema validity 68.75%→100%** and **correct unsupported-case abstention 88.89%→100%** on the **inspected development set**, without training, without new inference, and with **0% exact supported-answer match both before and after**.
- CuratorKIT and AuditKIT now have verified separate artifacts, but **the end-to-end Track 1 fine-tuning pipeline is still incomplete**. Do not claim a trained model or a before/after *model* improvement.
- Run `python -m scripts.submission_gate --strict` for an intentionally fail-closed readiness check: it should return NOT_READY until trained adapter, paired tuned model data, and HF publication evidence exist.

## Risk register and mitigation

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Model never completes a real optimizer step | **Critical** | T4 tokenizer fix, offline preflight and safe one-time retry; fallback only as workaround, not proof of AlignTune training |
| Precision mismatch in official training | **Critical** | TRL 1.7.1 turns LoRA adapters BF16 during trainer initialization; must use compatible precision path, verify optimizer and save artifacts |
| Losing Colab checkpoint | **High** | Download ZIP and run info on genuine success before runtime disconnect; store hashes |
| Artificially inflated accuracy | **Critical** | Strict metric vs semantic review kept separate; show unsupported and injection failures |
| Very small synthetic training data | **High** | Expand with human validation and evidence-grounding; disclose 78 template-like examples as a limitation |
| Final test used during development | **High** | Freeze the fresh 60-case candidate after human review; do not train/tune on it; evaluate only once for final reporting |
| Full model or API key published accidentally | **High** | Only permitted adapter weights, approved model license and no tokens/secrets |
| AuditKIT chain not demonstrated | **High** | Run AuditKIT on saved baseline now without GPU; later on trained predictions, compare matched samples |
| Missed deadline | **Critical** | Complete pre-submission package earlier than Nov 14, verify actual Devpost/organizer receipt |

## Evidence standards

- A green `pip install` or import check is **not** model training.
- Successfully loading a model and LoRA layers is **not** an optimizer step.
- Training on fictional policies is **not** evidence of real banking reliability or safety.
- A direct PyTorch fallback checkpoint is a genuine checkpoint only after optimizer steps and adapter hash checks, but it must **not** be described as completed AlignTune trainer training.
- AuditKIT replay of previously generated predictions is genuine use of its evaluation API, but does **not** create new model results.
- Do not fabricate metrics, model URLs, training artifacts, human review judgments, judge permissions, or submission confirmations.

## Next execution order

1. Tokenizer-only preflight and controlled PyTorch fallback retry in the existing Colab T4 (after `git pull`). Preserve prior failure record.
2. Run `scripts.auditkit_replay` on preserved baseline predictions; no GPU needed.
3. Review candidate final holdout by at least two independent language/evidence reviewers, then freeze it and never use for tuning.
4. Complete either supported AlignTune training or transparently document its failure and determine eligibility implications.
5. Run same-prompt base vs tuned evaluation with evidence and human review, publish Hugging Face artifacts and final write-up.
6. Submit and verify acceptance prior to the official deadline.

*GitHub Actions CPU evidence has now been independently checked, including the actual saved baseline and AuditKIT. Live Colab GPU access is unavailable at the last screenshot; trained-model execution remains unverified.*

## Additional checkpoint evidence hardening — November 4, 2026

A completed-looking `run_info.json` and nonempty `adapter_model.safetensors` are no longer sufficient for `trained_adapter` readiness. `scripts.verify_training_archive` checks PEFT safetensors structure, LoRA A/B tensors, SHA-256 integrity against saved receipts, exact CuratorKIT training data hash, original model ID, source code revision, declared training backend and pinned base-model revision. It rejects corrupt/fabricated files and distinguishes manual PyTorch fallback from AlignTune's trainer. This is still **internal artifact integrity**, not proof of training execution, generalization or competition eligibility. See [training-artifact verification guide](TRAINING_ARTIFACT_VERIFICATION.md).
