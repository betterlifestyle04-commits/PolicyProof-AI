# PolicyProof AI — Track 2a safety benchmark report (2026-11-03)

**Status: A real CPU-only CuratorKIT 0.1.22 → AuditKIT 1.1.3 chain passed GitHub Actions.** It does not demonstrate fine-tuning, new inference, or production safety. All policy texts are fictional.

## Research question and design

Do unchanged Tiny Aya Global outputs respect strict structured JSON, correct source attribution, evidence-dependent abstention and embedded-document instruction boundaries?

The benchmark includes 48 synthetic policy question instances, 16 each in English, Norwegian and Arabic. It has 36 answerable previously unseen-policy questions, 9 unanswerable cases, and 3 context-injection probes. The policy records are disjoint from the 78-row training material. However, these 48 questions were inspected repeatedly during development and are NOT a final blind safety evaluation. Gold answers were generated with templates, not signed off by independent bilingual reviewers.

## Official toolkit execution

In [GitHub Actions](https://github.com/betterlifestyle04-commits/PolicyProof-AI/actions/runs/41695411216), CuratorKIT's JSONLReader, SchemaGate, ExactDeduplicator, TextCleaner and AlpacaExporter each processed all 48 rows without rejection; no exact duplicates were removed. The curated export preserved all case tags and reference labels. The curated file was byte-identical to its staged input, so no quality/diversity gain can be attributed to cleaning.

AuditKIT's real precomputed-evaluation API then scored the exact 48 previously saved base-model outputs; no model inference occurred. AuditKIT fingerprint: ae297dd84f990566.

- Original evaluation-dataset SHA-256: d61e59557c62e486afc49e40e5d4daff0c8896429ac2b71d04936b07a49d3a4c
- CuratorKIT curated export SHA-256: 7a3fbeec5eae88493f9d60f9cc0e2c588ec506b01e58dc5e57edd987508bc2ff
- Native CuratorKIT manifest configuration hash: 06866bd9dda5f5b4
- Python 3.11.17; no GPU; no LLM API or external judge service
- Full 10-file pipeline evidence is available as a GitHub Actions artifact

## Results for the unchanged Tiny Aya base model

| Test | Numerator / denominator | Interpretation |
| --- | ---: | --- |
| Strict end-to-end exact correct | 8/48 = 16.67% | Wording-sensitive; not semantic accuracy |
| Complete 3-field JSON | 33/48 = 68.75% | Structural validity only |
| Raw expected source ID | 38/39 = 97.44% | Can coexist with incorrect content |
| Strict valid cited source | 20/39 = 51.28% | Still not proof of semantic correctness |
| Exact supported answer text | 0/39 = 0% | Paraphrases not adjudicated |
| Unsupported-case correct abstention | 8/9 = 88.89% | Only nine cases |
| Unsupported substantive claim | 1/9 = 11.11% | Raw-output conservative detection |
| Supported policy group | 0/36 exact; 22/36 complete schema | Unseen fictional policies |
| Injection group | 0/3 exact; 2/3 complete schema | Cannot generalize from three attacks |

None of three injection responses contained the exact planted marker OVERRIDDEN, but this is **not** a general jailbreak-resistance score.

A separate deterministic inference-time guard improved schema from 33/48 to 48/48 by construction. Only one unsupported-case abstention changed from wrong to right; a paired exact test is p=1.0 on this nine-case subgroup. Exact supported-answer matching remained 0/39. This is not SafeTune model steering or fine-tuned-model improvement.

## Limitations and next scientific steps

The original Tiny Aya model revision SHA was not saved. There was no independent human grounding evaluation, no adjudicated translation quality, no varied multi-turn attack corpus and no measured improvement to model weights. The evidence supports a reproducible small safety-audit prototype, **not a validated real banking assistant**.

A stronger future study needs more independently reviewed cases, many more attack categories, new model inference under pinned checkpoint/version, calibration of over-refusal, and larger sample sizes.

## Reproducibility

On a clean Python 3.11 clone, install CuratorKIT 0.1.22 and AuditKIT 1.1.3, run the existing project data-prepare module, then run scripts/track2a_chain.py as a Python module. The complete pipeline automatically verifies original dataset hash, saved prediction coverage, case/label integrity, Curator manifest checksums and AuditKIT completion.

**Submission blockers:** Hugging Face public dataset/model collection requirements and organizer clarification on the model collection for a benchmark-only Track 2a submission. The upstream Tiny Aya weights are gated; they cannot be falsely described as a newly published ungated model. No official submission is claimed. Deadline: November 14, 2026 AoE.
