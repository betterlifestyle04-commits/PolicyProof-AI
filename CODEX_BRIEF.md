# Codex build brief — PolicyProof AI / Lexsi 2026

Do not claim model training or improvements until GPU-produced artifacts and raw predictions exist. Current repo intentionally contains a reproducible Python benchmark and mock smoke tests only.

Next priorities:
1. Verify CuratorKIT export behavior with its actually installed version; pin versions and run a local curation fixture.
2. Make a human-reviewed, non-template multilingual dataset with larger unseen-policy and adversarial holdouts. Record data licenses and reviewer decisions.
3. Obtain gated access to CohereLabs/tiny-aya-global; choose GPU hardware with known cost. Run reproducible base inference before training.
4. Validate AlignTune training recipe and LoRA/QLoRA feasibility against GPU VRAM; save adapter, logs, random seed and exact hyperparameters.
5. Run AuditKIT on real base and tuned predictions for both splits, plus blinded semantic checking. Document uncertainty and regressions.
6. Publish dataset/model cards, evaluation traces, training evidence and short demo. Complete hackathon form when received.
7. Keep GitHub Actions green. Do not write tokens or private records to the repository.

Challenge the research design: one translated fact can be trivially memorized, exact-match metrics misgrade legitimate rephrases, test templates can leak shape, and fictional policy answers are not validated for production finance. Never describe synthetic, mock or oracle scores as actual safety improvement.
