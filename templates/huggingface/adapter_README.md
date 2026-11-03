---
base_model: CohereLabs/tiny-aya-global
base_model_relation: adapter
library_name: peft
pipeline_tag: text-generation
language:
  - en
  - "no"
  - ar
tags:
  - peft
  - lora
  - qlora
  - multilingual
  - research
  - lexsi-2026
license: cc-by-nc-4.0
---

# PolicyProof AI — Tiny Aya Global LoRA adapter (publication template)

**STOP — NOT A TRAINED MODEL RELEASE YET.** This file is a template in the GitHub source repository. No trained model or adapter checkpoint has been successfully verified at the time this draft was prepared. **Do not upload this as a completed adapter until the items below are filled with real logs, hashes, evaluation metrics, and a valid PEFT checkpoint.**

## Model description

- **Base:** [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global), gated open weights, 3.35 billion parameters, multilingual.
- **Intended task:** JSON-only answers from supplied fictional policy passages, explicit source ID, and abstention for insufficient evidence.
- **Languages:** English, Norwegian, Arabic.
- **Use restrictions:** Research only. No banking/financial advice. Verify base model terms, including Cohere Labs Acceptable Use Policy and the applicable noncommercial license, before releasing or using an adapter.
- **Trainer:** **PENDING** — record whether the official AlignTune trainer completed or whether a direct PyTorch fallback was used. Do not describe the fallback as AlignTune training.
- **Checkpoint revision / SHA-256:** **PENDING**
- **Verified optimizer steps:** **PENDING**
- **Dataset SHA-256:** \`4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c\` (original CuratorKIT 78-row training-only Alpaca export)
- **Adapter weights:** **NOT PRESENT / UNVERIFIED**

## Training data

The original CuratorKIT export contains 78 manually templated fictional policy-training examples, balanced at 26 examples in each language. These are not actual customer policies or banking records. Independently human validated for semantic correctness: **NO**.

The previously inspected 48-case development benchmark must not be used for model selection, training or a final blind evaluation. The separate 60-case candidate benchmark also requires independent language/evidence review before any final claim.

## Evaluation: DO NOT FILL WITHOUT REAL TUNED PREDICTIONS

| Measure | Unchanged original base, development split | This adapter |
| --- | ---: | --- |
| Strict exact end-to-end score | 16.67% (8/48) | **PENDING** |
| Complete JSON schema | 68.75% (33/48) | **PENDING** |
| Raw correct source ID (on 39 answerable cases) | 97.44% (38/39) | **PENDING** |
| Correct abstention (on 9 unsupported cases) | 88.89% (8/9) | **PENDING** |
| Unsupported substantive claims (on 9 unsupported cases) | 11.11% (1/9) | **PENDING** |
| Independent semantic assessment | Not measured | **PENDING** |

The base-model metrics above are strict *development* diagnostics from the user's saved real-model-labelled outputs. They do not prove semantic grounding or performance of this adapter.

## Reproducibility checklist

- [ ] Pin base model exact revision and tokenizer revision
- [ ] Record package versions, hardware and training backend
- [ ] Attach unmodified training log / completed optimizer step count
- [ ] Upload nonempty adapter_config.json and adapter_model.safetensors
- [ ] Record SHA-256 of every saved file
- [ ] Evaluate both models on identical frozen, reviewed samples
- [ ] Publish unconditional **and** conditional metrics with denominators
- [ ] Provide failure examples and limitations
- [ ] Ensure judge access without pasting private Hugging Face tokens

Source: https://github.com/betterlifestyle04-commits/PolicyProof-AI  
Research is experimental and is **not** a certified safety or financial compliance system.
