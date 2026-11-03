# PolicyProof AI — honest 3-minute demo script (draft)

**Recording is not created.** Use screen recording plus spoken narration, with a view of the public GitHub repo and Streamlit observer. Target approximately 3 minutes; do not claim the missing trained adapter exists.

## 0:00–0:20 — Question and problem

**On screen:** Title, three language tabs or example questions.

**Narration:** "Can a small multilingual language model answer a policy question using only the policy in front of it—and admit when the answer isn't there? PolicyProof AI investigates that problem in English, Norwegian and Arabic, using invented financial-service policies."

## 0:20–0:45 — Data and toolkit

**On screen:** CuratorKIT verification report, 78-row ZIP hashes and language counts.

**Narration:** "We created 78 synthetic training records—26 per language—and used Lexsi CuratorKIT to validate and export them. They are fictional and templated, not customer records. We separate training policies from test-policy IDs to reduce leakage."

## 0:45–1:20 — Genuine saved base-model evidence

**On screen:** Research dashboard → Evaluation runs → preserved Tiny Aya original responses.

**Narration:** "Tiny Aya Global ran on a Colab T4 with the original, unmodified weights. We preserved 48 outputs. The correct policy ID appeared in 38 of 39 raw answerable outputs, yet exact supported-answer text matching was zero of 39. That's why we never treat a citation match as proof of correctness. The original complete JSON schema rate was 68.75 percent."

## 1:20–1:45 — Actual AuditKIT integration

**On screen:** GitHub Actions real baseline CI run and AuditKIT fingerprint.

**Narration:** "We then replayed those exact saved Tiny Aya responses through AuditKIT 1.1.3 without a GPU. The benchmark checksum and all 48 prediction IDs matched. AuditKIT produced fingerprint ae297dd84f990566. It tests the evaluation pipeline, not a new model run."

## 1:45–2:25 — What our safety guard actually changed

**On screen:** [Guarded output development results](INFERENCE_GUARD_RESULTS_2026-11-03.md), with a concrete malformed JSON example.

**Narration:** "We built a small fail-closed output guard. It detects missing required JSON fields, contradictory abstention claims, and invalid policy source identifiers. On the same recorded development responses, JSON schema compliance changed from 68.75 to 100 percent; correct abstention on nine unsupported questions changed from eight to nine. But the guard did not fix factual content: exact supported-answer match remained zero. This is a system-level validation layer, not fine-tuning."

## 2:25–2:50 — Limitations and training blockers

**On screen:** Submission gate NOT_READY, optional Colab GPU blocker.

**Narration:** "Our attempted 4-bit LoRA training revealed BF16 and FP16 gradient-scaler incompatibilities, and Colab later exhausted its GPU allocation. We do not claim successful fine-tuning or improved model weights. The research corpus is small, fictional and not independently reviewed for semantic correctness."

## 2:50–3:00 — Reproducibility

**On screen:** Source README, one-command smoke checks, official site.

**Narration:** "All source code, original baseline outputs, evaluation scripts and limitations are public on GitHub. The remaining steps are a genuine checkpoint, independent final comparison and public Hugging Face artifacts. PolicyProof AI is an honest, reproducible research prototype."

## Optional alternate ending ONLY after real training succeeds

Replace the 2:25–2:50 section with a demonstration of real optimizer logs, saved adapter hash, identical base-versus-tuned examples, conditional and unconditional metrics, and review of regressions. **Never show a mock or missing result as trained-model evidence.**
