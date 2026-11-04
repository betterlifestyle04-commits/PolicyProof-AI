# Judge quick start — PolicyProof AI (GPU-free verification)

**Public source:** https://github.com/betterlifestyle04-commits/PolicyProof-AI
**Live read-only judge demo:** https://policyproof-ai-observatory.onrender.com (Render free tier; startup may be slow after idle). The deployment is an observatory of saved outputs, not a live trained language model.  
**Official hackathon:** https://hackathon.lexsi.ai  
**Track status:** Track 1 fine-tuning attempt **unfinished**. Track 2a **actual CPU benchmark/toolkit chain completed**; official HF publication and submission pending.

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
