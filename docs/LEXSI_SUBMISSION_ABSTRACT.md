# PolicyProof AI — Lexsi submission abstract (draft, 247 words)

**Working draft only. Do not claim a trained adapter or submit as a completed Track 1 outcome yet.** Official limit is under 500 words. This text is based on already-verified GitHub Actions evidence from November 3, 2026.

PolicyProof AI investigates a common failure in multilingual document question-answering: a model can cite the expected policy identifier without producing a correct, well-formed, or evidence-supported answer. We study this behavior in English, Norwegian, and Arabic using fictional financial-service policy statements.

CuratorKIT processed 78 synthetic training examples (26 per language). We preserved 48 outputs from the unchanged Cohere Labs Tiny Aya Global base model and verified their exact evaluation-dataset hash, prediction IDs, and deterministic scores. AuditKIT 1.1.3 then replayed those saved model outputs through its native evaluation interface without additional GPU inference.

On this previously inspected development benchmark, strict exact-match accuracy was 16.67%, complete JSON schema compliance 68.75%, and correct abstention on nine unsupported questions 88.89%. Raw source identifiers matched 38 of 39 answerable cases, but exact supported-answer text matched zero. Source-ID matching alone therefore cannot establish answer correctness.

We also evaluated a deterministic inference-time output guard on the same preserved responses. It improved schema compliance to 100% and unsupported-question abstention from eight to nine cases, while exact supported-answer correctness remained zero. This is an output-validation intervention, not a fine-tuned model or demonstrated improvement in semantic accuracy.

We report negative engineering results from an attempted AlignTune QLoRA run, small-sample uncertainty, and the limitations of synthetic, unreviewed evaluation data. The source code, public read-only dashboard, baseline responses, evaluation scripts, and reproducibility instructions are available. A successfully trained adapter, independent final model comparison, and complete public Hugging Face collections remain outstanding; no model-training success or production safety claim is made.

---

**Public demo:** https://policyproof-ai-observatory.onrender.com

**GitHub:** https://github.com/betterlifestyle04-commits/PolicyProof-AI

**HF model collection:** NOT PUBLISHED

**HF dataset collection:** NOT PUBLISHED

**Official submission confirmation:** NOT RECEIVED
