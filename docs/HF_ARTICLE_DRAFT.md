# When citations look right but answers are wrong: an honest multilingual Tiny Aya policy-QA audit

*PolicyProof AI · Lexsi Labs AI Safety & Alignment Hackathon 2026 · Track 1 research draft*  
*Status: baseline reproducible; tuned model and resulting improvement **not yet demonstrated**. This is a publication draft, not a successful fine-tuning announcement.*

## Executive summary

PolicyProof AI studies whether an openly available multilingual model can provide short, correctly attributed, structured answers from supplied fictional policy statements and abstain when the evidence is insufficient. We built 78 synthetic training examples using **CuratorKIT** in English, Norwegian and Arabic, ran unmodified **CohereLabs/tiny-aya-global** on 48 policy-level-separated development cases, and replayed the saved responses with **AuditKIT 1.1.3**.

The baseline exhibited a revealing mismatch: the correct policy identifier appeared in **38 of 39 answerable outputs** when raw JSON was inspected (97.44%), while **none of those 39 answers exactly matched** the language-specific reference answer. Exact matching is over-strict on paraphrases, and matching a source ID does not establish that a response is factually supported. Five outputs had contradictory abstention fields, and one output cited an injected, non-policy note.

This is not evidence that the model is unsafe for financial deployment, nor that it can perform real banking analysis: all policies are fictional, the corpus is tiny, and no blinded semantic review was completed. The observation is narrower and useful—source citations, answer correctness, and abstention should be measured **separately**.

## Research question

Can a small, dense, open-weights multilingual LLM give evidence-grounded answers in multiple languages while respecting a machine-checkable schema and abstaining on unsupported requests? A robust answer must satisfy several distinct criteria: substantive accuracy, valid output shape, cited-policy identity, no hallucinated source, and correct abstention when evidence is missing.

## Dataset and evaluation design

- **Training-only:** 78 CuratorKIT-cleaned examples (26 per language), in Alpaca and ShareGPT formats. These are the same records in two formats, not 156 independent samples.
- **Development:** 42 paraphrases from training-source policies.
- **Previously unseen-source development evaluation:** 48 questions (16 per language) from six policy IDs excluded from training. Of these, 36 are answerable, 9 unsupported, and 3 document-injection cases.
- **Candidate independent evaluation:** 60 questions spanning eight new policy IDs, with 20 per language. Language/evidence signoff is still pending. This candidate set is not eligible for a claim of genuine blind results after being inspected for development.

All questions, policy texts and intended reference answers are fictional. Source policy documents are English even when questions and labels are Norwegian or Arabic, which introduces a translation component. Code and data-source integrity checks are available in the public [GitHub repository](https://github.com/betterlifestyle04-commits/PolicyProof-AI).

## Base model and reproducibility

The original user-run model was **CohereLabs/tiny-aya-global**, a 3.35B dense multilingual open-weights model, used in FP16 on a Google Colab Tesla T4 after gated access was accepted. No fine-tuning was performed for the recorded baseline.

- Original code commit: \`ef712529d083f0f3c161f8839e2337f94fa2af66\`
- Development dataset SHA-256: \`d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c\`
- Exact upstream model weight revision: **not recorded** (null in original report)
- Original predictions and conservative v2 scorer data: [preserved JSON](../artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json)

On November 3, 2026, a fresh GitHub Actions CPU job verified that all 48 saved predictions matched the generated dataset, recomputed the stored deterministic metrics, then replayed the original responses using **AuditKIT 1.1.3** and its native precomputed-inference API. AuditKIT fingerprint: \`ae297dd84f990566\`. No new model inference or human judging was performed by the replay.

## Baseline results — strictly deterministic, not semantic judging

| Metric | Unmodified Tiny Aya development result |
| --- | ---: |
| Strict end-to-end exact match | **8/48 = 16.67%** |
| Complete, correct three-field JSON schema | **33/48 = 68.75%** |
| Parseable JSON objects (including incomplete schema) | **48/48 = 100%** |
| Required-field omissions | **15/48 = 31.25%** |
| Contradictory abstention claims | **5/48 = 10.42%** |
| Raw expected source ID, answerable questions | **38/39 = 97.44%** |
| Correct source under strict JSON schema, answerable questions | **20/39 = 51.28%** |
| Exact supported answer match, answerable questions | **0/39 = 0%** |
| Correct abstention on unsupported questions | **8/9 = 88.89%** |
| Unsupported cases emitting a substantive claim | **1/9 = 11.11%** |

The 39 answerable questions consist of 36 standard cases and three injection probes. The 9 unsupported cases form the abstention denominator. For source-ID auditing, the 97.44% figure does **not** mean that those answers are semantically correct.

**AuditKIT unconditional averages over all 48:** strict 16.67%, complete JSON schema 68.75%, raw expected policy ID 79.17%, correct unsupported abstention 16.67%. These unconditional numbers differ from conditional citation and abstention rates because the denominators are different.

## A conservative inference-time safety layer (separate experiment)

We also implemented a deterministic output validator that checks three-key JSON shape, source-ID presence in the supplied policy text, invalid citations to untrusted markers, and contradictory abstention fields. It never examines the reference answer and therefore cannot act as an oracle. The output is normalized to explicit abstention when any validation fails.

**IMPORTANT:** This layer does not verify the substantive truth of a valid-shaped answer, and may reduce answer coverage. The measured before-and-after on saved baseline outputs will be inserted only after the CPU experiment and integrity checks pass. Even if successful, this would be an **inference-time output safeguard**, not a trained-model improvement or an AlignTune result.

## Fine-tuning status and negative engineering findings

Using AlignTune and TRL on a free Colab T4, the 4-bit Tiny Aya base and 3,833,856 trainable LoRA parameters were successfully initialized with all 78 training examples. Optimizer execution repeatedly failed due to mismatched BF16 gradients with PyTorch FP16 unscaling. A separate direct-PyTorch fallback encountered an incompatible tokenizer Encoding type. Corrective code, tests and Kaggle/Colab recovery notebooks were prepared, but a successful optimizer step and saved adapter **have not been observed**.

Do not present this as a completed fine-tuning experiment. Colab GPU quota was also exhausted on November 3. Any later direct-PyTorch result must be named as such, not credited to the AlignTune training loop.

## Next experiment (pending GPU and independent review)

1. Obtain a legitimate GPU runtime and load the verified CuratorKIT archive, Tiny Aya revision and LoRA setup.
2. Complete optimizer steps with logs and hash-verified saved adapter, recording the actual trainer backend.
3. Independently review the new evaluation questions and labels and freeze a hash.
4. Generate base and adapted model responses under identical deterministic settings; keep the 48-case development diagnostic separate from the final evaluation.
5. Compare strictly structured outputs, conditional citation/abstention performance, unsupported claims, injection handling and language-specific regressions. Include a separately adjudicated semantic support check and a transparent coverage tradeoff.
6. Publish model adapter, curated data, logs and evaluation artifacts on Hugging Face under the appropriate base-model noncommercial terms.

## Limitations and responsible interpretation

PolicyProof AI is a research prototype built on tiny, template-like fictional data. It is not a reliable production banking system. The baseline's apparent citation strength is an easy-to-misinterpret source-ID statistic; exact-match answer scores can also penalize correct paraphrases. Negative experiments and the absence of an exact original model-revision SHA materially limit reproducibility. We disclose those failures rather than inventing missing training outcomes or claiming a safety improvement.

**Project:** https://github.com/betterlifestyle04-commits/PolicyProof-AI  
**Hackathon:** https://hackathon.lexsi.ai/  
**Research artifact:** pending verified Hugging Face publication and official submission.

*Publication gate: do not remove the status disclaimer or label a trained checkpoint until the legitimate paired evaluation, licenses, and public artifacts have been verified.*
