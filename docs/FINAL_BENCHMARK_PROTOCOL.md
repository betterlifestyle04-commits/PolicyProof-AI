# Candidate final evaluation benchmark — protocol v1

**Status:** Authored and versioned; human review not yet performed. **Do not evaluate or tune on this set during development.** This file is part of a public repository and therefore cannot be claimed as a secret or formally blind benchmark.

## Composition

- **60 fictional evaluation cases** across eight new source-policy IDs: `POL-F101`–`POL-F108`.
- **20 cases in each language**: English, Norwegian and Arabic.
- **48 answerable** (two question formulations per policy per language), **nine unsupported**, **three malicious document-injection** probes.
- All policy evidence is in English; questions and reference answers are multilingual. This partially measures translation and is disclosed.
- Source documents and labels are manually drafted, *not independent human-validated*.
- Source-policy IDs do not occur in the original 12 training policies or six already-inspected development policies.

Original source: [final_holdout_sources_v1.json](../data/final_holdout_sources_v1.json).  
Candidate JSONL: [final_holdout_v1.jsonl](../data/final_holdout_v1.jsonl).  
Validation command: `python -m scripts.validate_final_holdout` (no GPU required).

## Reviewer protocol before freezing

Have at least **two independent reviewers**, including Norwegian and Arabic language competence, review without seeing model predictions:

1. For every source, confirm that the question is answerable solely from its evidence, and the reference answer is semantically correct.
2. For all three languages, check translations convey the **same policy fact**, including numeric thresholds, time units, exclusions and negations.
3. Identify ambiguous questions, mismatched answer forms (especially the negative-question variant), or policies admitting multiple source IDs.
4. For unsupported cases, verify that **none of the eight policies supplies the answer**; unsupported is not a synonym for difficult.
5. For injection cases, confirm the malicious `UNTRUSTED-NOTE` is **not** a policy source and should not be cited or followed.
6. Resolve disagreements and log reviewer IDs/pseudonyms, adjudication, final label version, and dataset SHA256. Do not silently rewrite questions after evaluating candidate models.

**If corrections are needed:** version the dataset as v2 *before* running either base or trained model on it. Freeze a single hash and do not revise it for improved outcomes.

## Comparison protocol

- Run base and trained Tiny Aya on **identical** prompts, identical datasets and deterministic generation settings.
- Record exact base SHA, adapter SHA, tokenizer version, code commit and decoding options.
- Preserve all raw predictions and use AuditKIT plus domain-specific deterministic scorers, separately reporting:
  - Exact schema validity
  - Citation source-ID match on answerable questions
  - Abstention recall on unsupported questions
  - Unsupported-claim rate
  - Strict end-to-end exact match (does **not** equal semantic accuracy)
  - Bilingual reviewer-judged semantic correctness and evidence support
  - Language-specific differences and injection-specific failures
- Check denominators; unsupported cases should never artificially inflate source-citation scores.
- Paired significance/bootstraps may describe uncertainty, but the small and synthetic sample cannot establish broad real-world safety.

## Integrity safeguards

`scripts.validate_final_holdout` checks counts, categorical balance, unique IDs, policy source isolation, reference sources, abstention schema and known training/dev overlap. Its passing status would establish **structural** validity only. It cannot validate bilingual meaning, answer quality, or reviewer independence.

**Do not include the final benchmark in CuratorKIT training, trainer datasets, prompt selection, hyperparameter tuning, checkpoint choice or debugging feedback loops.** The original 48-case baseline is already a **development diagnostic**, not a fresh final holdout.


## Independent reviewer tooling (added November 4, 2026)

For a versioned reviewer-specific CSV with complete fictional evidence and proposed labels, use the dashboard's **Final benchmark review** page or `python -m scripts.review_holdout --export reviewer_A`. Independent reviewers record three `yes`/`no` judgments per case and keep the original dataset digest intact. Reconcile two or more completed CSVs using `python -m scripts.review_holdout --summarize reviews/reviewer_A.csv reviews/reviewer_B.csv`. The reconciliation detects missing/negative/disputed judgments and altered source material, **but intentionally never creates a sign-off**. See [full independent-review workflow](FINAL_REVIEW_WORKFLOW.md).
