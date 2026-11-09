# PolicyProof AI — Track 2a alternative abstract (draft)

PolicyProof AI presents a multilingual safety benchmark for evidence-grounded policy question answering. We examine whether the unchanged Cohere Labs Tiny Aya Global language model produces structurally valid JSON, attributes responses to policy evidence, abstains on unsupported requests, and avoids obeying instructions planted inside documents.

Our synthetic development benchmark contains 48 records, equally distributed across English, Norwegian and Arabic: 36 answerable fictional policy questions, nine unanswerable cases and three context-injection probes. We executed the official CuratorKIT 0.1.22 pipeline to validate, deduplicate and clean the benchmark while preserving case identifiers and reference answers. We then executed official AuditKIT 1.1.3 evaluation against 48 previously saved Tiny Aya base-model responses, without new GPU inference.

The unchanged model produced complete required JSON on 33/48 cases and strict exact correctness on 8/48. A raw expected policy identifier appeared in 38/39 answerable outputs, but only 20/39 had a structurally valid correct citation, and zero exactly matched the reference answer text. Eight of nine unsupported cases correctly abstained; one emitted a substantive unsupported claim. Three injection probes are insufficient to establish general robustness.

We also built a separate, unscored 72-case multilingual adversarial challenge, with 12 examples per family: control, role spoofing, fake source ID, encoded instruction, language override and unsupported requests. CuratorKIT structurally validated all 72. No model was run on these new cases, and no independent semantic review has been claimed.

An additional deterministic output validator normalized incomplete JSON and conflicting abstention fields. It increased schema compliance to 48/48 and fixed the single unsupported claim, but exact supported-answer correctness stayed zero. This is an inference-time guard, not SafeTune steering or model fine-tuning.

We provide reproducible code, CuratorKIT and AuditKIT native receipts, source hashes, recorded model outputs, a public dashboard and detailed limitations. Data and labels were synthetic and previously inspected, the original model revision SHA was not recorded, and independent semantic grading remains outstanding. We do not claim real-world financial safety or model-weight improvements.

The synthetic dataset and its public Hugging Face dataset collection now exist. A separate model collection and completed Lexsi submission are still pending.
