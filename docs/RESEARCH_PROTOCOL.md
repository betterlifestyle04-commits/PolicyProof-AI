# Research protocol — PolicyProof AI

**Status:** Draft protocol, not measured model performance.

## Hypothesis
CuratorKIT-curated, multilingual policy-grounded supervised fine-tuning of an approved Cohere Labs dense open-weights model using AlignTune improves citation-supported factual answering, valid JSON and calibrated abstention on held-out policies. The null hypothesis is that there is no meaningful gain; the analysis must report negative findings.

## Dataset and splits

- **Train:** 78 synthetic EN/NO/AR examples from 12 invented policy IDs; not human-validated.
- **Paraphrase test:** 42 synthetic examples on **the same policy facts** (not document-generalization evidence).
- **Unseen-policy test:** 48 synthetic examples based on six disjoint fictional policies, including nine unsupported questions and three document prompt-injection cases.
- Run the integrity guard before training and each evaluation: python -m scripts.audit_dataset --data-dir data.
- Never use held-out labels, source text, question templates or evaluation predictions for training, checkpoint selection or prompt iteration without declaring subsequent test contamination.
- All three languages currently use English-language evidence, so translated answer correctness partly measures translation capability.

## Experimental arms

A. **Base checkpoint**, identical prompt and decoding parameters, no training.
B. **Raw synthetic data fine-tune** (ablation): requires versioned training data with no CuratorKIT quality filtering.
C. **CuratorKIT-curated data fine-tune** (primary): same base checkpoint, compute, seed, optimizer, dataset size where feasible.
D. **Optional data mix ablation**: curated multilingual vs English-only training to isolate cross-language transfer.

At least A and C must actually run. For any additional arm record checkpoints, model license, hardware, total training tokens and evaluation hashes. If fine-tuning fails, disclose it; do not report simulated results in place of measurements.

## Metrics and statistical method

1. **Strict end-to-end accuracy** — a correct supported answer **and** correct source ID, or correct evidence-based abstention.
2. **Citation correctness** — correct policy ID among answerable questions. Citation correctness alone does not guarantee grounding.
3. **Answer exact match** — normalized output against one reference string. This is a narrow metric; valid paraphrases may be falsely rejected.
4. **JSON schema validity** — exact expected keys and primitive types.
5. **Abstention recall** — proportion of genuinely unsupported questions abstained on.
6. **Unsupported-claim rate** — proportion of unsupported questions answered with a claim.
7. **Language and adversarial slices** — EN/NO/AR and injected-context cases shown separately, not pooled into a single optimistic figure.

Use identical frozen evaluation examples, prompt and decoding settings. Use the paired bootstrap interval implemented by scripts.compare on the strict accuracy delta. Report absolute sample counts. An interval is not evidence of improved real-world safety by itself.

## Human review

Have at least two blind reviewers check a predefined stratified sample of model outputs for semantic correctness and evidence support, including translations and refusals. Track agreement, disagreements, source citations, and final adjudication separately from deterministic exact-match metrics. Do not use examples used to choose checkpoint/prompts in final held-out claims.

## Reproducibility manifest

For every model run record: exact model revision and model repository, tokenizer revision, adapter or merged checkpoint SHA-256, dataset manifest SHA-256, evaluation split SHA-256, random seed, hardware type, precision, training configuration and code commit hash, dependency versions, wall-clock runtime, GPU hours, and raw predictions. Keep secrets out of artifacts.

## Decision rules

Publish only improvements supported by real model outputs; note regressions in any language or safety slice. The main headline should use a held-out split and be replicated across at least three seeds if hardware budget permits. Treat all synthetic results as preliminary.

## Official links

https://hackathon.lexsi.ai
https://github.com/Lexsi-Labs/CuratorKIT
https://github.com/Lexsi-Labs/aligntune
https://github.com/Lexsi-Labs/AuditKIT
