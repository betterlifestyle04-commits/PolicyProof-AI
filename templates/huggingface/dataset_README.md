---
language:
  - en
  - "no"
  - ar
task_categories:
  - text-generation
pretty_name: PolicyProof AI fictional multilingual policy QA
tags:
  - synthetic
  - multilingual
  - policy-qa
  - lexsi-2026
size_categories:
  - n<1K
---

# PolicyProof AI — CuratorKIT fictional multilingual policy QA dataset

**Publication template, not a claim of semantic validation.** Upload only data that the project owner has permission to publish, and select an appropriate dataset license before making a public Hugging Face repository. Do not upload access credentials.

## Dataset summary

This research dataset consists of 78 **synthetic** question/answer pairs derived from invented financial-service policy statements. English, Norwegian and Arabic each contribute 26 training rows. CuratorKIT processed the records into training-ready Alpaca and ShareGPT exports. These are two representations of the **same 78 records**, not 156 unique examples.

The source documents are entirely fictional. The text does not contain actual banking policy or client information. The task asks the model to respond in a strict JSON object with: answer, source_id, abstain.

## Provenance and integrity

- CuratorKIT version at original curation: 0.1.22.
- Source/data preparation and schema guard: [GitHub](https://github.com/betterlifestyle04-commits/PolicyProof-AI).
- Original Alpaca SHA256: \`4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c\`.
- Original CuratorKIT archive SHA256: \`c6340969501965bf0bbfe8fa9247f41150b59e999b4881e78a27f16a233b8948\`.
- Six training examples have abstention labels.
- Translations and reference answers are not yet independently reviewed by qualified bilingual experts.
- All source policies and tasks are synthetic; do not generalize scores to real bank operations.

## Recommended splits

The CuratorKIT archive is **training only**. The separately generated 42-case paraphrase set is development material. The 48-case unseen-policy set was inspected during development and is also **not** a blind final benchmark. The independent 60-case candidate final holdout was authored later and requires two-person language/evidence review; it must not be included in training.

## Uses and limitations

Appropriate: controlled research, regression testing for structured output, abstention behaviour and source-ID matching in three languages.

Inappropriate: real financial compliance advice, production safety certification, or claims of factual multilingual reliability. A correct source ID can coincide with a wrong answer, and strict exact-match scores punish valid paraphrases.

## How to verify

Download the published training ZIP, run the repository's \`scripts.verify_curator_archive\`, and compare the SHA-256 against the values above. Do not upload any private Hugging Face access tokens.

**Responsible-use note:** This is a small, templated, non-representative research dataset. No formal de-identification guarantee is relevant because the source data are fabricated, not customer records.
