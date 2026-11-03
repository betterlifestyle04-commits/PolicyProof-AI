# PolicyProof AI — QA and evidence status (2026-11-03)

This report distinguishes **verified CPU checks** from **unverified runtime/model-training claims**. It is not a certification that every environment or endpoint works.

## Final verified GitHub CI outcome

| CI workflow run | Result | Scope |
| --- | --- | --- |
| [#98](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41633373985) | **66 passed; 2 skipped** | Evidence tamper/consistency verifier and ID-matched comparison, source-policy isolation, existing CPU unit suite |
| [#105](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41633871947) | **73 passed; 2 skipped** | Added live Streamlit AppTest for four dashboard pages and fail-closed legacy training CLI |
| [#111](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41634264913) | **79 passed; 2 skipped** | Added static Python syntax, no-saved-output, default-off training, and obvious-secret checks on notebooks |
| [#116](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41634618707) | **81 passed; 2 skipped** | Added real AuditKIT API precomputed synthetic-oracle evaluation contract, no model generation |

| [#122](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41635240908) | **81 passed; 2 skipped** in standard suite **plus 7 passed** in a separate CPU-PyTorch job | Previously skipped LoRA FP32 precision repair, failed optimizer-state guard and combined recovery tests |

GitHub logs for the standard runner report **81 passed, 2 skipped** (excluding package installation and setup). The two skipped modules depend on PyTorch, absent from the lightweight CI; a **separate CPU-only PyTorch 2.10 job executed those seven guard tests successfully**. CUDA, actual LoRA training and GPU memory behavior remain unverified.

## Changes actually merged to main

- [PR #4](https://github.com/betterlifestyle04-commits/PolicyProof-AI/pull/4): baseline prediction completeness, deterministic metric and checksum verification; compare paired samples by ID rather than order.
- [PR #5](https://github.com/betterlifestyle04-commits/PolicyProof-AI/pull/5): actual Streamlit dashboard-page smoke tests, and block legacy 3.35B full-parameter `scripts.train --execute` path before a dangerous unsupported attempt.
- [PR #6](https://github.com/betterlifestyle04-commits/PolicyProof-AI/pull/6): static validation of Colab/Kaggle notebooks, default-off GPU flags and obvious hard-coded credentials.
- [PR #7](https://github.com/betterlifestyle04-commits/PolicyProof-AI/pull/7): real AuditKIT `ak.evaluate(model="precomputed")` interface exercised with clearly labeled **synthetic oracle responses**.

- [PR #8](https://github.com/betterlifestyle04-commits/PolicyProof-AI/pull/8): dedicated official PyTorch CPU wheel runner for seven LoRA training-state safety tests. All seven passed.

## Independently checked original conversation evidence

The supplied `policyproof_curatorkit_train_v1.zip` was examined without needing CUDA:

- ZIP SHA256: `c6340969501965bf0bbfe8fa9247f41150b59e999b4881e78a27f16a233b8948`
- Alpaca export: **78 rows**, ShareGPT export: **78 rows** (two representations of the same examples, not 156 unique samples)
- Language composition: 26 English, 26 Norwegian, 26 Arabic
- Abstention training labels: 6
- No mismatch in the dedicated internal checksum list
- Verified Alpaca JSONL SHA256: `4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c`

The supplied actual-model rescored baseline JSON was also inspected: **48 predictions with 48 unique IDs**, 48 score rows and matching recalculations from the stored score rows for strict accuracy (16.67%), schema validity (68.75%), unsupported abstention recall (88.89%), unsupported claim rate (11.11%), and raw citation-ID match (97.44%). This verifies **internal report consistency only**. An independent fresh GPU inference from its original model/weights and exact matching dataset bytes was **not** executed here.

## Verified functionality versus remaining gaps

| Component | Verified scope | Still outstanding |
| --- | --- | --- |
| CuratorKIT | Original archived ZIP integrity, balanced 78-row training data, no held-out policy IDs in its training instructions | Semantic data review, near-duplicate analysis, human validation |
| Dataset generation and provenance | CPU CI builds 78 train, 42 paraphrase, 48 development cases, checks source-ID separation and manifest | Real-world/generalization relevance |
| Candidate final benchmark | 60 rows, 20 per language, 8 new policy IDs, no overlap, structural checks | Independent bilingual gold-label review; no true blind evaluation yet |
| Dashboard | All four Streamlit pages render with AppTest without exceptions | Public hosting, external browser/mobile QA, judge access |
| Colab/Kaggle notebooks | JSON/Python static syntax, training-off flags, no committed outputs | External service authorization, GPU availability, full notebook execution |
| AuditKIT | Installed in CI; actual library evaluates 48 precomputed **synthetic-oracle** examples using project scorers | Replay of original real-model baseline JSON on exact dataset; later real tuned checkpoint |
| Training | Unsafe old `scripts.train --execute` route disabled; fallback scripts and integrity guards present | **No verified trained adapter; no completed optimizer step; no T4/P100 runtime test on fallback** |
| Hugging Face artifacts | Publication requirements documented | Adapter/model logs/dataset publication and license verification |
| Competition submission | Honest draft, risk register, protocol and deadlines prepared | Actual public artifacts, before/after measurements, submission confirmation |

## Risk notes

1. This project's central scientific hypothesis **has not yet been proven**. The synthetic dataset is small and template-like; exact-match metrics penalize correct paraphrases.
2. Google Colab explicitly blocked the owner's GPU session for usage limits; no T4 backend available at last screenshot.
3. A Kaggle [GPU fallback notebook](../notebooks/04_kaggle_gpu_fallback.ipynb) is prepared, not executed. The direct-PyTorch fallback, if it runs successfully, must not be labeled an AlignTune trainer success.
4. The **two skipped** CI test modules require PyTorch/GPU-adjacent training code; they remain uncovered on the tested CPU runner.
5. No real trained-model predictions, human semantic judgments, adapter hashes, or Hugging Face repository URLs should be invented.

## Priority path to eligible submission

1. Get permitted GPU access (free Kaggle quota, returned Colab quota, or an organizer-issued eligible Hugging Face credit).
2. Run a single controlled preflight on the curated original ZIP and 4-bit model, then a safely guarded optimizer smoke test; verify all steps and save the adapter ZIP.
3. Make a supported and truthful use of the required **minimum two Track 1 toolkits in a real chained pipeline**; the curator and actual AuditKIT integration are now independently exercised, but an end-to-end successful training chain still lacks evidence.
4. Independently review and freeze the new multilingual evaluation set; evaluate identical real base and tuned outputs, compare more than one metric, and preserve hashes.
5. Publish permitted artifacts to Hugging Face and verify public judge-accessible links; complete the [submission draft](LEXSI_SUBMISSION_DRAFT.md) and submit by **November 14, 2026** according to the [official hackathon](https://hackathon.lexsi.ai/).

**Conclusion:** CPU research scaffold and dashboard pass substantial automated testing; no credible basis yet to claim end-to-end GPU training or real-world alignment improvement.
