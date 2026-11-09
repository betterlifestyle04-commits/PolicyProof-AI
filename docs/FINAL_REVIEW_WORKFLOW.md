# Final holdout: independent multilingual gold-label review

**Status (November 4, 2026): NOT REVIEWED.** This is a reproducible, CPU-only workflow for assembling and reconciling reviewer feedback on the 60-case final *candidate* benchmark. It is not evidence of model inference, fine-tuning, validated translations, or two independent human reviewers.

## Scope and trust model

- 60 fictional-policy cases: 20 English, 20 Norwegian, and 20 Arabic.
- 48 supported answers, 9 unsupported prompts, 3 injection probes. All labels are **proposed**.
- Reviewers see the question, complete fictional context, proposed answer/source, and abstention proposal. They **do not see model predictions**.
- Three verdicts for **each case**: `evidence_support_yes_no`, `translation_faithful_yes_no`, `question_unambiguous_yes_no`. Allowed values: lowercase `yes`, `no`, or blank until judged.
- At least two **different actual humans** should independently review every case, with suitable Norwegian, Arabic and evidence interpretation competence. The validator only checks distinct *codes*, not real identity or language proficiency.
- The public final candidate is **not genuinely blind**; it remains a versioned, held-out development protocol, not proof of generalization to unseen real financial policy.

## November 5 — non-destructive private review outputs

The 60-case reviewer export and reconciliation CLI now **refuse to overwrite** any existing file, including a blank template that has already been filled by a human. Output destinations must be **direct** files inside the private git-ignored `reviews/` folder; absolute paths, nested paths, traversal, symlinks, public `docs/` and frozen dataset paths are refused. CLI input paths for reconciliation are interpreted relative to the canonical repository root. The reader also refuses symlinked, oversized, malformed, short or extra-column reviewer files. The tool checks no more than four independent reviewer CSVs.

If you rerun an export and see a `FileExistsError`, **preserve the existing file**. If it contains genuine votes, do not delete it to make a command pass. Use a distinct unused reviewer code or filename **only for a genuinely new human review round**. If starting a new round, back up the previous files and their SHA-256 digests first. A new reconciliation output must use an unused filename; old reports are immutable evidence.

This protection prevents accidental loss or public-file corruption. It does **not** establish reviewer independence, language competence, semantic correctness, or authorization to mark a signoff `review_complete=true`.

## Generate reviewer-specific packs (CPU only)

Each reviewer should download a separate packet from the Streamlit **Final benchmark review** page, or run:

```bash
python -m scripts.review_holdout --export reviewer_A \
  --output reviews/reviewer_A.csv
python -m scripts.review_holdout --export reviewer_B \
  --output reviews/reviewer_B.csv
```

These files contain all 60 fictional examples, full proposed labels, source evidence, reviewer pseudonym and repeated SHA-256 of the untouched candidate dataset. **Every verdict is blank.** The script does not generate reviews, model outputs or inference.

Review in a spreadsheet editor or CSV tool. Do not change case IDs, question/context, gold labels, source IDs, dataset SHA, or reviewer code. Write just `yes` or `no` into the three verdict columns, and optional `review_notes` for a proposed correction. For unsupported items, check all supplied policy records to confirm the answer really is missing. For injection items, do not treat attacker notes as real sources. Check translations, negations, units and paraphrases. Reviewers should not view each other's filled packets before completing their own.

## Easier option: each human can review in a private offline browser

The live **Final benchmark review** page now offers **Download private offline reviewer workbench (HTML)** next to the blank CSV. Pick a **different pseudonym for each actual independent reviewer**, download the HTML packet for each one and send it privately to the intended qualified person. Its only content is the 60 frozen fictional questions, complete source evidence and **proposed gold labels**; **no model predictions** and **no prefilled votes** are included.

The reviewer opens the downloaded `.html` file directly in a normal browser (no log-in or internet connection needed), reads each case and records three judgments: source support, translation fidelity and question clarity. The interface shows the exact original label and source, a progress bar, language filter, and a distinct No decision. It **does not run a model, send answers to the server, auto-approve any case, or prove reviewer qualifications**.

The reviewer should click **Export private CSV / save progress** frequently (or use **Ctrl+S** / **Cmd+S** to request a review CSV download instead of accidentally saving the original blank HTML). The offline page now prominently shows **UNSAVED PRIVATE JUDGMENTS** after any new vote, cleared vote, or edited note, and requests confirmation before closing or reloading a page that still contains unsaved work. After an export it says **CSV download requested**, rather than claiming that a file was saved: the human must confirm that the downloaded CSV actually exists. Browser leave-page dialogs are not guaranteed in every browser or on mobile. The page deliberately uses **no localStorage, sessionStorage, network synchronization or other automatic storage** for private votes. A browser crash can still lose work since the last real CSV download.

An incomplete CSV is only a private backup, **not an approved review**. After reopening the same unmodified HTML, the reviewer can import their **own** saved CSV to continue; the browser verifies all original case IDs, reviewer code, dataset hash, source text, proposed labels and 60-case row order before accepting votes. It rejects altered files and exports an Excel-compatible UTF-8-with-BOM CSV with the exact existing validator schema.

To create the equivalent standalone HTML from the repository without downloading it from Streamlit:

```bash
python -m scripts.review_offline_html --reviewer reviewer_A
python -m scripts.review_offline_html --reviewer reviewer_B
```

This creates **new files** `reviews/reviewer_A_offline.html` and `reviews/reviewer_B_offline.html`; `reviews/` remains private and git-ignored. It refuses existing filenames, traversal paths and symbolic links. **Do not upload completed CSVs or private notes to the public GitHub repository or public Streamlit deployment.** If using a spreadsheet application, import CSV columns as literal text, not executable formulas. Preserve every source CSV and review note exactly as produced.

Once an actual reviewer has exported a complete CSV, rename or copy the original files into distinct unused direct paths such as `reviews/reviewer_A.csv` and `reviews/reviewer_B.csv`. **Do not falsify blank or "no" verdicts to make the gate green.** The existing strict `scripts.review_holdout --summarize` command remains the sole automated reconciliation step; it must still require human adjudication and approval.

## Consolidate the files, without inventing sign-off

```bash
python -m scripts.review_holdout \
  --summarize reviews/reviewer_A.csv reviews/reviewer_B.csv \
  --output reviews/reconciliation.json
```

This command checks all 60 cases and the exact original content; rejects changed question text, changed source evidence, changed proposed gold, wrong dataset SHA, missing/duplicate case IDs, unknown verdict values, and repeated reviewer codes; then reports:

- `REVIEW_INCOMPLETE`: some verdicts blank.
- `REQUIRES_CORRECTIONS_OR_ADJUDICATION`: at least one negative verdict or disagreement.
- `READY_FOR_MANUAL_ADJUDICATION_AND_SIGNOFF`: all verdicts are present and unanimous `yes`, but reviewers must still independently sign off.

The JSON includes every affected case/criterion, unanimous agreement percentage, per-reviewer coverage, and file SHA-256s. It always returns `review_complete: false`, `independent_human_identity_verified: false`, and `requires_manual_signoff: true`. Agreement can be perfect even when both humans are wrong.

## What remains manual

Reviewers must resolve disagreements, check proficiency and independence, and explicitly sign their conclusions. If source material, questions or gold answers require any correction, **create benchmark v2 with a new hash before model evaluation**; do not quietly edit the frozen v1 file. Keep the old files as provenance and produce new review packets for the revised version.

Only after human review and adjudication should the separate `docs/final_benchmark_review_signoff.json` be updated with real reviewer information and the exact dataset hash. The script intentionally does **not** update this signoff file or assert that any person reviewed the set. Existing submission checks must not treat an unsigned reconciliation as a completed review.

No model on this final candidate should be prompted, trained, tuned or scored during the above review process. Do not select hyperparameters or stop training based on these labels. The original 48-case saved baseline remains the development diagnostic.

## Practical privacy and evidence hygiene

Use fictional policy records only. Reviewers should use pseudonyms, not emails or full names, in public files. The Streamlit server serves blank CSVs but does not accept or store completed human reviews. Keep filled files private until the reviewers authorize publication. A human review only validates the suitability of a synthetic benchmark; it cannot establish real-world financial advice reliability or model safety.

See [final benchmark protocol](FINAL_BENCHMARK_PROTOCOL.md) and [submission readiness](SUBMISSION_READINESS.md).

## New: original reviewer files are now mandatory for GPU review signoff

The guarded paired evaluator no longer accepts `review_complete: true` and two pseudonyms alone. The reviewer must retain **private, SHA-256-linked copies** of both full 60-case annotated CSVs and an unmodified reconciliation report under `reviews/`. Before accepting review completion, `python -m scripts.verify_review_provenance` now rechecks the full CSVs against the original frozen benchmark bytes and requires all 180 judgments per reviewer to be affirmative, no disagreements or negatives, exactly matching reconciliation content, and explicit individual manual approval declarations with declared Norwegian/Arabic/English competence across the review team. `reviews/` is gitignored.

**This remains consistency validation, not proof of human independence, language expertise or correct semantics**. Never generate positive votes automatically or mark signoff complete merely to pass CI. All existing actual human work must be retained privately; the public repository remains **NOT REVIEWED**. For the exact v1 review-evidence schema and instructions, see [source-linked review provenance](REVIEW_SOURCE_PROVENANCE.md).


### Optional safe local signoff draft

After real independent reviewer CSVs and reconciliation are complete and unanimous, run `python -m scripts.prepare_review_signoff --review-csvs reviews/reviewer_A.csv reviews/reviewer_B.csv --reconciliation reviews/reconciliation.json --output reviews/signoff_draft.json`. This verifies frozen source hashes and creates a **private unsigned draft** with all reviewer approval, independence, and blinding flags `false`, and no competence claims. It never updates the public signoff. Only real humans can choose to approve their review and complete those declarations. See [exact procedure](REVIEW_SOURCE_PROVENANCE.md).
