# PolicyProof AI — Lexsi Labs Hackathon 2026

Track 1 research project: multilingual, evidence-grounded policy QA with measurable abstention and citation quality.

**Status:** Under development. No real model training, quality gain, or final hackathon submission has yet been verified.

The bundled policy corpus is entirely fictional. The experiment is designed for English, Norwegian and Arabic; it evaluates a held-out-policy set separately from paraphrases. Simulated evaluation outputs are tests of the harness, not research results.

## Scientific protocol

1. Prepare training and evaluation data from fictional source policies, keeping unseen document IDs completely separate.
2. Curate training data using official Lexsi CuratorKIT and inspect provenance before proceeding.
3. Evaluate the gated `CohereLabs/tiny-aya-global` base model against unchanged test sets.
4. Train using Lexsi AlignTune on approved GPU hardware. Record exact configs and save checkpoints.
5. Re-evaluate with AuditKIT and deterministic validators. Publish raw predictions and a paired comparison with uncertainty.

## Development commands

```bash
python -m scripts.prepare
python -m pytest -q
python -m scripts.evaluate --split unseen-policy --engine mock-abstain --output artifacts/mock.json
python -m scripts.train  # dry-run only; training requires --execute
```

## Research integrity

Do not report mock scores as performance. Model access, GPU training, and the real external Lexsi integration have not been confirmed. Never commit secrets or private banking documents.

Official event: https://hackathon.lexsi.ai · CuratorKIT: https://github.com/Lexsi-Labs/CuratorKIT · AlignTune: https://github.com/Lexsi-Labs/aligntune · AuditKIT: https://github.com/Lexsi-Labs/AuditKIT
