---
language:
- en
- "no"
- ar
license: cc-by-nc-4.0
library_name: peft
pipeline_tag: text-generation
base_model: CohereLabs/tiny-aya-global
base_model_relation: adapter
tags:
- peft
- lora
- ai-safety
- research
- multilingual
- synthetic-data
- lexsi-hackathon-2026
---

# PolicyProof AI: Native AlignTune 40-step Tiny Aya LoRA — research artifact

> **Research artifact, NOT a safety-improved model.** In our 48-case synthetic development comparison, formatting improved but abstention on unsupported questions **regressed**. Not intended for unattended factual or policy decisions. The base-model repository may require separate gated Hugging Face approval; this **adapter-only** repository contains no original base-model weights.

## Scope and base-model rights

This repository contains *only* the portable PEFT LoRA adapter, its configuration, an aggregate evaluation summary and the SHA256 release manifest. It does not contain Tiny Aya base-model weights, raw private research predictions, credentials or private reviewer notes. The upstream base is [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), governed by **CC BY-NC 4.0** together with Cohere Labs' acceptable-use conditions; users must comply with the applicable terms and may need to accept the upstream gated-access form. The inherited license tag is not a warranty that any particular commercial or downstream use is authorized.

## Training record and provenance limits

- Original training input: 78 curated *synthetic fictional policy* examples, separate from the 48-case development experiment and 72-case adversarial challenge. Input SHA256: `4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c`.
- Method: native Lexsi AlignTune 0.1.28 wrapper around TRL; original Colab training receipt **self-reports** 40 optimizer steps and a Tesla T4. We do not independently attest the GPU execution.
- Architecture: PEFT LoRA, rank 8, alpha 16, dropout 0.05, modules `q_proj,k_proj,v_proj,o_proj`, 288 BF16 adapter tensors, 3,833,856 nonzero adapter parameters.
- Adapter SHA256: `ba985c1966050764a5220e87ea581a9bd482ba87b055fee4a1e363643f1cc92e`.
- Adapter config SHA256: `808fea2999f91c2af09271b275877080ecf89683faec5e7a63f8000ec5d25621`.
- Requested original base-model revision: `af89d219b53ed9b13b8a4645f9c8028973510324`. **The revision actually loaded during training was not independently observed.** A later matched inference experiment used an authenticated, revision-pinned source for both arms; that does not repair the missing training-time provenance.

## Matched 48-case synthetic development evaluation

Both arms were scored using the original frozen fictional-policy development rubric. This set was **previously inspected**, so it is *not* a blind generalization benchmark. Data contain 16 examples each in English, Norwegian and Arabic, including 9 unsupported questions.

| Strict synthetic-reference metric | Unchanged Tiny Aya Global | Tiny Aya + this adapter |
| --- | ---: | ---: |
| Complete JSON schema | 14/48 | **48/48** |
| Strict exact match, all cases | **4/48** | 2/48 |
| Exact supported-answer text, 39 answerable cases | 0/39 | 0/39 |
| Raw expected source identifier among answerable | 31/39 | **39/39** |
| Correct canonical abstention on 9 unsupported questions | **4/9** | 2/9 |
| Unsupported-question substantive claims | **5/9** | 7/9 |

**Interpretation:** The training trial strongly improved *formatting consistency*, but did not demonstrate better supported-answer correctness and **made unsupported-question abstention worse**. In the tiny Arabic unsupported subset, correct abstentions fell from 2/3 to 0/3. This is a descriptive synthetic result, not proof of performance across Arabic speakers or domains. A matching citation ID does not establish passage entailment. Exact-string scoring can also reject semantically equivalent answers. No qualified human semantic grading has established answer correctness or model safety.

## Reproduction and safety

Source code, pinned evaluation notebooks and research limitations: [PolicyProof AI GitHub](https://github.com/betterlifestyle04-commits/PolicyProof-AI). Public synthetic dataset: [PolicyProof Track 2a benchmark](https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark). Full aggregate-only analysis: [Stage 13 native40 paired dev48 evidence](https://github.com/betterlifestyle04-commits/PolicyProof-AI/blob/main/docs/TRACK2A_NATIVE40_DEV48_OBSERVED_PRIVATE_RESULTS_2026-11-10.md).

Use the adapter only in an authorized research environment with the upstream gated Tiny Aya base model, and perform independent evaluation before deployment. This repository supplies no executable code that bypasses upstream gated model access. It is not intended as a guaranteed prompt-injection defense, legal advice system, or production policy decision engine.

**Release provenance:** The publication script checks the exact original owner ZIP SHA256 `dcefc5b0babf708aa1b48ac0349ddb3bc4745405c72e9636c5860d7719d8f0e9` and the two adapter-file SHA256 values above before staging any file. This model card does not claim the adapter is already publicly published merely by existing in GitHub.
