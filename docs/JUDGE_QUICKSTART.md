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
