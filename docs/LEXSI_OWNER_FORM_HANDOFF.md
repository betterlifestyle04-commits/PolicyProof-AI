# Lexsi 2026 — owner-only submission-form handoff (Track 2a)

**Status as of November 9, 2026: UNPUBLISHED DATA / UNVERIFIED COLLECTIONS / NOT SUBMITTED.** This is an offline form-preparation guide, never a submission receipt.

## Official organizer requirements — checked from [hackathon.lexsi.ai](https://hackathon.lexsi.ai/) on November 9

The [official submission page](https://hackathon.lexsi.ai/submission/) links to the organizer's live form. The published deadline is **November 14, 2026, end of day Anywhere on Earth (AoE)**; aim to submit earlier to leave time for corrections and confirmation. The organizer asks for:

1. The exact registration email, entered by the owner **in the official form**, not in this public code or an artifact.
2. An abstract **under 500 words**.
3. A public/ungated Hugging Face **model collection**, whose referenced model repositories are public.
4. A separate public/ungated Hugging Face **dataset collection**, whose referenced dataset/log/benchmark repositories are public.
5. A write-up or article; a reproducible GitHub README is acceptable where HF Articles access is unavailable.
6. A GitHub repository or notebooks, the model and stack used, track, and problem statement.

The organizer says **multiple submissions are allowed** and the **latest before the deadline counts**; do not confuse a previous draft with a submitted entry. Track 2a permits a dataset-and-report safety-benchmark deliverable, but the form still asks for a *model collection*. **Ask the organizer in its official support channel if a benchmark-only unchanged-base study qualifies** before choosing any irrelevant or gated collection.

## Get a copy-ready, truthful draft without touching the organizer form

```bash
python -m scripts.lexsi_submission_handoff
```

The output is a local, machine-readable form field map. It validates that the **unchanged historical Track 2a abstract is 50–499 words** (strictly under 500) and preserves the original 48-case / separate 72-case limitations. It reruns the original source-linked 48-case CuratorKIT/AuditKIT evidence and frozen 72-case challenge readiness check.

If new original 72-response evidence is accepted later, the draft statement that **"no model was run"** becomes stale, so the command **refuses to include its old abstract for form pasting** until a human updates and rechecks it. If authorized public HF collections are verified later, the statement that publication is **"still pending"** also becomes stale, and the old abstract is blocked.

Collection links remain **null** unless the existing release checker finds a **fresh anonymous HF collection-access receipt bound to the exact publication manifest**; merely writing a plausible URL or toggling a JSON `verified` flag never unlocks a copy-ready collection. The code leaves `registered_email=null`, `official_submission_completed=false` and `organizer_confirmation_verified=false` on every run. No external network, login, GPU inference, model training, human judgment, HF upload or submission occurs.

To retain a read-only draft in a new, immutable local output:

```bash
python -m scripts.lexsi_submission_handoff \
  --output artifacts/my_lexsi_track2a_form_draft_UNSUBMITTED.json
```

Do **not** overwrite prior drafts; a changing report requires a new filename. For a strict local gate (still not an external eligibility decision):

```bash
python -m scripts.lexsi_submission_handoff --strict
```

A nonzero exit is expected while verified HF collections are missing, the 72-case evidence conflicts with old claims, or a required source-chain check fails. **Do not bypass it by claiming a base model was fine-tuned or attaching an unrelated HF collection.**

## Exactly what to do manually when authorized public collections exist

Inspect the truthful handoff and its `required_actions` list. In a private browser, re-open and independently verify the write-up, source code, public datasets and all model/dataset collection members **without signing in**. Resolve the model-collection requirement with the organizer if our Track 2a unchanged-base study makes it unclear. Review rights, the gated/noncommercial Tiny Aya model license, any third-party material and selected public dataset license before publishing. Enter the actual registered email **only inside the organizer's secure form**.

Only then paste the up-to-date under-500-word abstract, verified public links, track/model/toolkit information and write-up into the [official form](https://hackathon.lexsi.ai/submission/). Confirm the organizer actually accepted the entry and retain their real confirmation or submission ID privately. The handoff script **never asserts that this step happened**.

For judging without a GPU, use the separately generated CI `policyproof-lexsi-judge-evidence-UNSUBMITTED` ZIP and [judge quickstart](JUDGE_QUICKSTART.md). That ZIP reproduces original 48 baseline source scores and preserves the new curated 72 examples, but is **not** substitute for public collections or a submitted organizer form.
