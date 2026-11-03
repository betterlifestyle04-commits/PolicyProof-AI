---
language:
- en
- "no"
- ar
task_categories:
- text-generation
size_categories:
- n<1K
tags:
- synthetic
- safety-evaluation
- prompt-injection
- grounded-qa
- structured-output
- multilingual
- lexsi-2026
pretty_name: PolicyProof AI multilingual fictional-policy safety benchmark
---

# PolicyProof AI — multilingual safety benchmark (research prototype)

**Draft dataset card for a future public Hugging Face dataset. No HF publication has yet been verified.** The underlying policies, questions and target answers are invented. There is no customer data or real financial policy.

## What's inside

There are TWO distinct datasets, which must never be conflated:

- **48 evaluated development cases:** 36 supported fictional policy questions, nine missing-evidence abstention cases and three prompt-injection probes. Each of English, Norwegian and Arabic has 16 questions. Their original recorded CohereLabs/tiny-aya-global base-model responses were evaluated through real AuditKIT 1.1.3. CuratorKIT 0.1.22 processed and preserved all 48 original case tags and reference labels.
- **72 NEW unscored challenge cases:** 12 each of normal controls, forged system-role instructions, fake policy sources, encoded instructions, multilingual override instructions and missing-evidence queries. Each language has 24 cases. CuratorKIT structurally curated all 72, but NO MODEL has answered them and NO HUMAN has independently reviewed their semantic labels.

**Never describe these 120 cases as a single 120-response model evaluation**. The second set is unscored and labels are synthetic.

## Provenance

- Original 48-case benchmark SHA-256: d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c
- CuratorKIT 48-case Alpaca export SHA-256: 7a3fbeec5eae88493f9d60f9cc0e2c588ec506b01e58dc5e57edd987508bc2ff
- New 72-case JSONL SHA-256: cdd5cf4e54caa6c79ce1f0cd5bd0f05344f0dc22039a2ad56a3c1c0ee2d32cd1
- New 72-case CuratorKIT export SHA-256: cf8b226ebff5a6a5e53e3132cec9581844d4d6fd92316885c59cbd535ec624df
- Tool versions: CuratorKIT 0.1.22, AuditKIT 1.1.3, CPU Python 3.11
- Full 20-file native evidence archive is available in [the public CI run](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41696571904).
- Source: https://github.com/betterlifestyle04-commits/PolicyProof-AI

## Results on the 48 inspected development cases only

| Metric | Unmodified model, recorded outputs |
| --- | ---: |
| Strict exact full task success | 8 / 48 (16.67%) |
| Complete required 3-key JSON format | 33 / 48 (68.75%) |
| Raw expected source ID on answerable cases | 38 / 39 (97.44%) |
| Strict valid expected source on answerable cases | 20 / 39 (51.28%) |
| Exact answer-text match on answerable cases | 0 / 39 (0%) |
| Correct abstention on unsupported cases | 8 / 9 (88.89%) |
| Substantive unsupported claim | 1 / 9 (11.11%) |

Citation-ID matching alone does not verify factual grounding. Exact text matching does not recognize valid paraphrases. The model revision hash was not recorded in the original saved report.

## Safety, labeling and methodology caveats

- All policy documents and cases are synthetic, small and templated. Four challenge source policies are intentionally repeated across attack variants; these cases are correlated.
- Questions are multilingual but evidence paragraphs remain English, confounding translation and evidence reasoning.
- Only three original prompt-injection cases have model responses, so jailbreak robustness is not established.
- Both sets have template-derived gold answers and lack independent bilingual label adjudication.
- Neither dataset is real-world regulatory guidance, a financial product policy, a production safety test or a claim of model improvement.
- An additional deterministic output guard may change JSON schema and abstention formatting but is **not** SafeTune model steering or model fine-tuning.
- Do not use either evaluation split for model fitting, hyperparameter tuning or any claimed held-out test once published.
- The gated original Tiny Aya weights are NOT licensed or distributed by this dataset.

## Rights and publishing

The dataset contents are original synthetic research fixtures written for PolicyProof AI. **The dataset owner must choose and verify an appropriate public data license** before making any Hugging Face repository public. This draft deliberately makes no unverified license assertion. Any publication should remain consistent with the applicable Tiny Aya model license and Lexsi rules.

For reproducibility use the GitHub source and native CuratorKIT + AuditKIT CI receipts rather than treating this card as a training log.
