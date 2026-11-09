# Judge quick start — PolicyProof AI (GPU-free verification)

**Public source:** https://github.com/betterlifestyle04-commits/PolicyProof-AI
**Live read-only judge demo:** https://policyproof-ai-observatory.onrender.com (Render free tier; startup may be slow after idle). The deployment is an observatory of saved outputs, not a live trained language model.  
**Official hackathon:** https://hackathon.lexsi.ai  
**Track status:** Track 1 fine-tuning attempt **unfinished**. Track 2a **actual CPU benchmark/toolkit chain completed**; official HF publication and submission pending.

## November 9: official Lexsi form-preparation handoff (NOT submitted)

Open the [verified official submission page](https://hackathon.lexsi.ai/submission/) to enter the form **yourself**. The organizer requires your registered email (never store it in a public artifact), a strictly under-500-word abstract, distinct public/ungated Hugging Face model and dataset collections, a reproducible write-up/README, source code/notebooks and the model/toolkit/track/problem statement. Deadline: **November 14, 2026, end of day AoE**.

The new CPU-only command `python -m scripts.lexsi_submission_handoff` prepares a bounded copy/paste **draft** with a word-count check and local source-linked Track 2a readiness. Verified HF collection URLs remain **null** until the separate anonymous remote-access receipt passes. The original 72-case historical abstract is **blocked** if newer saved original responses make its old unscored claim false; a revised abstract cannot claim new evaluation without source replay. The real registration email, actual license rights and organizer confirmation remain **manual tasks**. No website login, model call, HF publishing or official form submission occurs.

A successful native-toolkit main CI run retains a separate downloadable artifact named `policyproof-lexsi-track2a-owner-form-draft-UNSUBMITTED`. Open the live **Judge verification** tab for the human-readable abstract and missing-field checklist. See [owner form handoff](LEXSI_OWNER_FORM_HANDOFF.md). **Neither the dashboard nor the CI artifact is an official submission receipt.**

## November 9: owner-only offline form checklist (NOT submitted)

The **Judge verification** tab can now download an HTML checklist with readable,
read-only fields and the historical abstract, based on the same source-linked
validation as the existing JSON handoff. The original 72-case draft is withheld
if newer real saved responses make it stale. Unverified HF collection URLs remain
missing, and registration email never enters the download. The form page is
offline, static and cannot publish or submit. CI also retains this source-checked
HTML in the UNSUBMITTED owner-form artifact. See
[exact owner instructions](LEXSI_OWNER_FORM_HANDOFF.md).

## NEW: one verifiable source-evidence bundle for the judges (November 8)

A successful **main-branch** CuratorKIT/AuditKIT CI run produces `policyproof-lexsi-judge-evidence-UNSUBMITTED`, a downloadable artifact with the exact ZIP `policyproof_lexsi_judge_evidence_UNSUBMITTED.zip` and `judge_packet_verification.json`. The ZIP contains an **allowlisted** 48-original-response / 72-curated-but-unscored research handoff, SHA256SUMS, a manifest and a `START_HERE.txt` explaining exactly what was and was not done.

Use [Research scaffold CI → successful main run](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/workflows/ci.yml), find the `Real CuratorKIT then AuditKIT safety benchmark (CPU)` job, and download the **judge evidence UNSUBMITTED** artifact. Preserve the ZIP intact, and run:

```bash
python -m scripts.build_judge_evidence_packet --inspect \
  artifacts/policyproof_lexsi_judge_evidence_UNSUBMITTED.zip
```

The verifier reads every file, checks an exact inventory and byte-level checksums, reconstructs the original dataset and toolkit receipts into an isolated temporary folder and **recomputes all 48 original model-output scores and the 72 unscored challenge source labels**. A complete marker or owner-authored JSON flag by itself cannot pass. It refuses unknown or symlink-like ZIP members, oversized data, duplicates, changed score receipts, invalid public paired-development summaries, and apparent credentials. The checks run **without CUDA, Hugging Face login, network calls, independent human votes, or model generation**.

Only source artifacts and public research narrative are included; **the original uploaded 48-case paired raw ZIP is NOT distributed**, and the public paired summary cannot be mistaken for raw inference evidence. No PEFT adapter, gated model weights, token, reviewer CSV, final 60-case score, approved Hugging Face collection, or official Lexsi submission confirmation is included. The ZIP is for **judge reproduction and limitation disclosure**, not proof of semantic improvement.

From a private checkout, the owner can create an independent fresh copy after the native toolkit receipts are already generated:

```bash
python -m scripts.build_judge_evidence_packet
python -m scripts.build_judge_evidence_packet --build
```

The builder only creates a **new** direct `artifacts/<safe-name>.zip` and refuses overwrites. The live dashboard's **Judge verification** tab links to the CI artifact source. Do not submit a synthetic test fixture as real evidence.

## What works today

- A verified original 78-row CuratorKIT training-only archive, 26 rows per language (English/Norwegian/Arabic).
- 48 saved real-model-labelled Tiny Aya Global development responses; exact dataset checksum and deterministic metrics verified on GitHub Actions.
- Actual native **AuditKIT 1.1.3** replay against those saved baseline responses; fingerprint \`ae297dd84f990566\`.
- Fail-closed output schema/citation/abstention guard with real *before-and-after system-level* development metrics, no GPU.
- A nine-page Streamlit observer with an interactive **synthetic-only** guard playground and separate 48-scored versus 72-unscored safety benchmark explorer; each page is covered by AppTest.
- **116 passing standard CPU tests, 2 skipped**, plus 7 separate PyTorch CPU safety tests. The real two-toolkit Track 2a job passed independently.

- Real **CuratorKIT 0.1.22 → AuditKIT 1.1.3** chain on the 48 preserved synthetic base-model responses, plus 72 additional CuratorKIT-curated **UNSCORED** multilingual challenge cases (no independent semantic review, no model inference on these 72 cases). Read [Track 2a report](TRACK2A_SAFETY_BENCHMARK_2026-11-03.md) and [alternative abstract](TRACK2A_SUBMISSION_ABSTRACT.md).

**What does NOT work yet:** completed AlignTune finetuning, trained adapter, genuine base-vs-tuned model evaluation, HF-public artifacts and confirmed final submission. These remain critical for full Track 1 eligibility. The guard is not a trained model.

## Verify on a clean Python 3.11 machine (no HF token, no GPU)

\`\`\`bash
git clone https://github.com/betterlifestyle04-commits/PolicyProof-AI.git
cd PolicyProof-AI
python -m pip install -U pytest "streamlit>=1.40,<2" "auditkit>=1.0,<2"
python -m scripts.prepare
python -m scripts.validate_final_holdout
python -m scripts.verify_evidence \
  --baseline artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json \
  --dataset data/unseen_policy_test.jsonl
python -m scripts.auditkit_replay \
  --report artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json \
  --dataset data/unseen_policy_test.jsonl
python -m scripts.inference_guard \
  --baseline artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json \
  --dataset data/unseen_policy_test.jsonl
python -m pytest -q
python -m scripts.submission_gate
streamlit run research_dashboard.py
\`\`\`

**Expected results (not claims about human semantic accuracy)**

| Evidence | Expected |
| --- | --- |
| Original saved responses | 48 matching IDs and SHA-256 \`d61e5955…7a49d3a4c\` |
| Unmodified Tiny Aya strict exact | 16.67% (8/48) |
| Unmodified Tiny Aya complete JSON schema | 68.75% (33/48) |
| Guarded complete JSON schema | 100% (48/48) by construction |
| Guarded correct abstention on 9 unsupported examples | 100% (9/9) |
| Exact supported answer match | **0/39 before and after** |
| Local readiness verdict | **NOT_READY** |

These are the *already-inspected synthetic development* cases. The original model commit hash is missing from the saved report. No newly generated base-model inference is involved. Do not equate source IDs with content truth.

## Dashboard walkthrough

1. **Research overview:** see train/test counts, limits and manifest.
2. **Dataset explorer:** view fictional policy evidence and multilingual questions, never sensitive banking data.
3. **Evaluation runs:** select the preserved Tiny Aya baseline (nested in \`artifacts/real_baseline\`); inspect metrics and all 48 original raw outputs. Distinguish simulated test-harness artifacts from model outputs.
4. **Safety benchmark:** compare the 48 actually scored cases with the 72 separate, unscored new adversarial cases; inspect proposed labels without pretending they are human-reviewed.
5. **Output guard experiment:** raw vs deterministic guarded results and small-sample uncertainty; schema compliance is enforced by construction.
6. **Try the output guard:** use fictional sample policy text to test the validator, without model inference.
7. **Judge verification:** current Track 1 not-ready release gate and actual AuditKIT provenance.
8. **Protocol & next steps:** see remaining GPU fine-tuning and official HF collection requirements.

## Reproduce the Track 2a actual two-toolkit chain separately (CPU only)

Use a separate clean Python 3.11 environment to avoid changing the original GPU training dependencies:

- Install the pinned official CuratorKIT 0.1.22 and AuditKIT 1.1.3 packages.
- Run the project data-prepare module.
- Run the scripts.track2a_chain Python module with output directory artifacts/track2a_safety.
- Run scripts.generate_track2a_challenge with --curate and output directory artifacts/track2a_challenge_unscored.
- Run scripts.track2a_readiness to produce the separate Track 2a release gate receipt.

The expected output is 48 evaluated base-model saved responses, 72 synthetic **unscored** challenge cases, and a Track 2a status of RESEARCH_READY_HF_PUBLICATION_BLOCKED, not a completed official submission. The exact command sequence is in the [CI job](../.github/workflows/ci.yml) and the [Track 2a report](TRACK2A_SAFETY_BENCHMARK_2026-11-03.md).

## Where to read more

- [Baseline model evidence](BASELINE_2026-11-02.md)
- [Detailed guard results](INFERENCE_GUARD_RESULTS_2026-11-03.md)
- [Human final benchmark review protocol](FINAL_BENCHMARK_PROTOCOL.md)
- [GPU recovery notebook](../notebooks/04_kaggle_gpu_fallback.ipynb)
- [Current QA verification](QA_VERIFICATION_2026-11-03.md)
- [Hugging Face release checklist](HF_PUBLICATION_RUNBOOK.md)
- [Submission draft](LEXSI_SUBMISSION_DRAFT.md)

If any command fails, report the exact traceback and **do not convert the failure into a green "ready" status**. The benchmark and logs should remain immutable.


## New final-candidate human review workbench

The **Final benchmark review** page lets independent reviewers download **blank** per-reviewer CSVs containing all 60 fictional final candidate questions, evidence and proposed labels. It never runs inference or pretends that reviewers have evaluated them. The CLI `python -m scripts.review_holdout --summarize <reviewer_A.csv> <reviewer_B.csv>` detects altered cases, missing judgments, disagreements and negative labels while deliberately withholding sign-off. See [full review workflow](FINAL_REVIEW_WORKFLOW.md). These are workflow and data-integrity improvements, **not** completed human validation.

## Offline adapter evidence gate (no GPU)

For a future genuine AlignTune output ZIP, judges can inspect the structural hashes and provenance fields using `python -m scripts.verify_training_archive --archive /path/to/archive.zip --strict-track1`. This does not invoke Tiny Aya and **cannot independently prove training occurred**. A manual PyTorch fallback remains non-equivalent to finished AlignTune training. The submission validator also rejects arbitrary nonempty files presented as PEFT checkpoints. See [checkpoint verification guide](TRAINING_ARTIFACT_VERIFICATION.md). No public trained adapter exists at the time of this update.


## New paired-model evaluation preflight (November 4, 2026)

Run `python -m scripts.frozen_pair_inference` without any flags from a clean checkout to see why genuine 60-case paired GPU evaluation remains blocked. It checks the unchanged final-candidate hash, human-review signoff, and a PEFT adapter with an AlignTune-declared run, matching hashes and pinned base-model revision. **No model is loaded or called.** The public **Judge verification** page shows the same fail-closed prerequisite table. Once independent review and actual training exist, follow [the paired GPU evaluation runbook](FROZEN_PAIRED_INFERENCE.md); do not use the earlier 48-case development outputs as the 60-case baseline.


## Reproducible recovery of the original model pass

The 60-case paired GPU runner now writes the complete original-base output **before** attempting the adapted-model inference. Its explicit `--resume-base` mode checks that saved file's provenance, 60-case coverage, dataset SHA, model revision, decoding settings, and recomputed deterministic metrics; invalid files are rejected before any GPU call. This is protection against wasting scarce GPU time if loading the adapter fails, **not** proof of successful model training or inference in the current repository. See [the interruption-safe evaluation guide](RECOVERABLE_PAIRED_EVALUATION.md).


### Recovery of a completed base and tuned run without additional GPU time

Both full original and tuned 60-case model reports are persisted **before** native AuditKIT scoring in the guarded GPU runner. If that CPU scoring stage alone fails, `python -m scripts.finish_saved_pair` displays the prerequisite check without execution. After independently checking the previous inference logs, the explicit `--finalize --confirm-saved-inference` flags validate both saved files, matching dataset/model/adapter metadata and original per-case metrics before repeating only the CPU AuditKIT comparison. The paired receipt marks `offline_replayed_from_saved_predictions=true` and `gpu_executed_in_this_process=false`. No verified GPU evaluation exists yet. [Protocol](RECOVERABLE_PAIRED_EVALUATION.md).

## Track 2a: independently check public HF collection access

A public URL and an owner-entered `"verified": true` field are **not** proof that judges can browse dataset or model collections anonymously. The new `python -m scripts.verify_hf_collections` performs an offline preflight. After genuine collection/repo URLs exist, `python -m scripts.verify_hf_collections --verify-network` reads the collections, all referenced ungated repos, dataset card, original 48-case saved baseline, 72-case unscored challenge and their published checksums with **no Hugging Face token**. The matching verification receipt expires for the release gate after 72 hours. The dashboard reflects the current evidence. No HF data/model has been published by this change. See [HF verification runbook](HF_ANONYMOUS_PUBLICATION_CHECKS.md).

## Download reproducible, unpublished Track 2a evidence handoff

The passing main-branch GitHub Actions CPU toolkit job now generates `track2a_release_candidate_v2_UNPUBLISHED.zip` as part of its `policyproof-track2a-real-curatorkit-auditkit-20261103` artifact. This contains **actual regenerated CuratorKIT/AuditKIT CPU receipts** over the 48 original saved Tiny Aya base-model predictions and a separately curated but **unscored** 72-case challenge. Inspect it with `python -m scripts.build_track2a_release_candidate --inspect /path/to/zip`; the script rechecks exact original benchmark hashes, label/case mapping, toolkit receipts, stored baseline metrics and per-file SHA256 before accepting it.

The bundle is **unpublished**, its license has **not** been selected, and it contains no trained weights. It is a different archive from the originally pinned 20-file owner-reviewed ZIP. Publication rights, a public Hugging Face dataset/model collection, human review and genuine model evaluation remain outstanding. [Release-handoff guide](TRACK2A_REPRODUCIBLE_HANDOFF.md).

## Safe owner handoff for the fresh v2 research ZIP

Once the owner reviews the new reproducible 15-member ZIP, there is a distinct `scripts.publish_track2a_candidate` route. Its default verifies the source ZIP's **explicit trusted SHA256** without choosing a public license or calling HF. A license preview remains offline, and a real upload requires `--publish --confirm-rights --confirm-publication`, an approved license and privately stored HF_TOKEN. The script also validates staged-file hashes before uploading and requires public anonymous readback to succeed. **Neither the dataset nor the required model collection is published by this feature.** [Owner publication runbook](TRACK2A_V2_PUBLISHING.md).

### Public dataset proof now checks the entire release, not only required files

On a genuinely published Track 2a dataset, the owner can opt into anonymous verification via `python -m scripts.verify_hf_collections --dataset-only --verify-network`. The verifier checks **every** SHA256SUMS entry and blocks unsafe/extraneous files, then pins model/collection access by Git revision. The optional v2 publisher further ties the anonymous public dataset Git revision to the upload commit and public source ZIP hash to the owner-approved original archive digest. The repository's CI uses mock network responses plus a separate real SDK *offline* compatibility contract; neither test demonstrates that a Hugging Face upload has occurred. [Full instructions](HF_ANONYMOUS_PUBLICATION_CHECKS.md).


### Review source evidence is now required before any final-model inference

The CPU-only `python -m scripts.verify_review_provenance` defaults to **BLOCKED** because no actual human judgments have been received. Positive reviewer pseudonyms and a `review_complete` flag are no longer enough: the paired evaluator requires SHA-256 linked complete per-reviewer CSVs and a valid reconciliation of every frozen case plus each reviewer's own approval declaration. The checks do not independently prove human identity or translation correctness, and the example fixtures in unit tests are **simulated**, not scientific evidence. [Full protocol](REVIEW_SOURCE_PROVENANCE.md).

## Optional real GPU training milestone (not yet executed)

The Colab/Kaggle notebooks now offer an explicitly opted-in, **40-step PyTorch QLoRA** run using an AlignTune-loaded PEFT model, 66 CuratorKIT training rows, and 12 small training-only validation examples. This avoids previously observed TRL mixed-precision trainer startup failures, but must **not** be represented as a successful AlignTune trainer execution. There is no trained adapter to evaluate yet; the only saved Tiny Aya predictions in the repo remain the inspected 48-case unchanged-model development responses. See [GPU training instructions](GPU_TRAINING_NEXT_STEPS.md).


### Real 40-step checkpoint: next evaluation, not a quality claim

We have a validated `manual_torch_extended_fallback` checkpoint artifact from a user's GPU session, but the repository has **no real paired base/tuned model predictions or improvement scores** from this checkpoint yet. A new CPU-only **preflight** and opt-in [Colab paired GPU evaluation notebook](https://colab.research.google.com/github/betterlifestyle04-commits/PolicyProof-AI/blob/main/notebooks/06_dev48_pair_eval_t4.ipynb) prepare a 48-case **inspected development** comparison. A correct pair must regenerate the untouched base-model results using the same HF revision requested for the adapted model; the older saved 48-case base is unpinned and cannot serve as the control. No final 60-case results or human review are implied. [Evaluation and recovery protocol](EVALUATE_REAL_40_STEP_ADAPTER.md).
