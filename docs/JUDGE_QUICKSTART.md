# Judge quick start — PolicyProof AI (GPU-free verification)

**Public source:** https://github.com/betterlifestyle04-commits/PolicyProof-AI
**Live read-only judge demo:** https://policyproof-ai-observatory.onrender.com (Render free tier; startup may be slow after idle). The deployment is an observatory of saved outputs, not a live trained language model.  
**Official hackathon:** https://hackathon.lexsi.ai  
**Track:** 1 (Fine-tuning and Data Curation), with an additional experimental inference-time output guard.

## What works today

- A verified original 78-row CuratorKIT training-only archive, 26 rows per language (English/Norwegian/Arabic).
- 48 saved real-model-labelled Tiny Aya Global development responses; exact dataset checksum and deterministic metrics verified on GitHub Actions.
- Actual native **AuditKIT 1.1.3** replay against those saved baseline responses; fingerprint \`ae297dd84f990566\`.
- Fail-closed output schema/citation/abstention guard with real *before-and-after system-level* development metrics, no GPU.
- A four-page Streamlit **read-only** research observer, verified using Streamlit AppTest (not separately hosted).
- 93 passing CPU tests in the expanded suite, plus seven separate CPU-PyTorch recovery tests (two ML tests skipped in lightweight runner).

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
4. **Protocol & next steps:** see requirements for successful GPU training and measured model improvement.

## Where to read more

- [Baseline model evidence](BASELINE_2026-11-02.md)
- [Detailed guard results](INFERENCE_GUARD_RESULTS_2026-11-03.md)
- [Human final benchmark review protocol](FINAL_BENCHMARK_PROTOCOL.md)
- [GPU recovery notebook](../notebooks/04_kaggle_gpu_fallback.ipynb)
- [Current QA verification](QA_VERIFICATION_2026-11-03.md)
- [Hugging Face release checklist](HF_PUBLICATION_RUNBOOK.md)
- [Submission draft](LEXSI_SUBMISSION_DRAFT.md)

If any command fails, report the exact traceback and **do not convert the failure into a green "ready" status**. The benchmark and logs should remain immutable.
