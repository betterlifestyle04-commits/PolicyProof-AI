# Lexsi submission readiness — evidence checklist

**Deadline:** Initial project submission November 14, 2026, 23:59 AoE (per organizer email). Demo Day November 22.

## Verified and completed
- [x] Public GitHub repository and research scaffold created.
- [x] Synthetic multilingual policy dataset generator and reference labels implemented.
- [x] Deterministic evaluation harness and paired comparison code implemented.
- [x] Dataset integrity auditor and malicious-corruption regression tests added.
- [x] Read-only demo/exploration dashboard source created.
- [x] The local sister scaffold passed 15 Python tests; GitHub-hosted CI has **not yet been independently confirmed green**.

## Remaining / cannot be claimed done
- [ ] Join Lexsi Discord and confirm registered Hugging Face username.
- [ ] Accept Tiny Aya Global gated license (or another permitted dense Cohere model).
- [ ] Redeem available Hugging Face GPU credits or provision approved compute.
- [ ] Run installed versions of CuratorKIT, AlignTune and AuditKIT, capture successful end-to-end logs.
- [ ] Curate and human-review additional multilingual training examples; strengthen unseen data.
- [ ] Obtain real base-model inference outputs.
- [ ] Fine-tune and save adapter/checkpoint with exact reproducible hyperparameters.
- [ ] Obtain real tuned-model outputs on identical frozen tests, run robust comparisons.
- [ ] Document honest costs, errors, negative findings, limitations and ethics.
- [ ] Publish all created model/adapter/log/dataset artifacts on Hugging Face.
- [ ] Submit public project and write-up using the organizer's forthcoming form; verify receipt.

## Risk register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| No GPU or license approval | Critical | Resolve immediately; Colab/Kaggle/HF compute options |
| Toolchain API/version incompatibility | High | Integration smoke test, pin versions, fail closed |
| Synthetic template leakage | High | Distinct policy IDs and expanded human-reviewed questions |
| Strict match misgrades correct translations | High | Blinded semantic adjudication |
| Metric tradeoffs hide unsafe failures | High | Report hallucination, abstention and language slices separately |
| Last-minute submission link | High | Monitor organizer email and Discord and submit early |
| Invalid or exaggerated claims | High | Separate smoke-test mock outputs from model-based results |

The artifact is not a winning submission until real improvement and complete package publication are verified.
