# Lexsi 2026 — owner-only submission-form handoff (Track 2a)

> **Research update — 10 November 2026:** The historical abstract `TRACK2A_SUBMISSION_ABSTRACT.md` and any old form handoff that describes the 72-case challenge as “never run” or claims the 40-step model is “not evaluated” are **obsolete**. The owner has supplied a private source-hashed 72-case response archive and a **paired native AlignTune dev48** archive. The adapter changed valid JSON from **14/48 to 48/48**, but supported-answer exact correctness remained **0/39** and unsupported abstention deteriorated **4/9 to 2/9**. Use the [current source-limited abstract draft](TRACK2A_CURRENT_SUBMISSION_ABSTRACT_2026-11-10.md), cross-check against [the native40 aggregate results](TRACK2A_NATIVE40_DEV48_OBSERVED_PRIVATE_RESULTS_2026-11-10.md), and **do not claim a model safety improvement**. Neither the private model nor these raw responses have been uploaded to Hugging Face. The required public model collection remains outstanding and the official form has **not** been submitted. The older CLI form helper still validates a historical abstract and should **not be treated as a current copy-ready submission** until its evidence gating is revised.


**Status as of November 9, 2026: SYNTHETIC DATASET + DATASET COLLECTION PUBLIC / MODEL COLLECTION NOT VERIFIED / NOT SUBMITTED.** The public dataset is [shalnark01/policyproof-track2a-benchmark](https://huggingface.co/datasets/shalnark01/policyproof-track2a-benchmark) in the [PolicyProof safety-datasets collection](https://huggingface.co/collections/shalnark01/policyproof-ai-multilingual-safety-datasets). The new read-only pinned source checksum verification runs separately; neither a public link nor this handoff is an official submission receipt.

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

## Offline copyable owner form checklist (November 9)

The **Judge verification** dashboard now has a second download:
**"Download offline read-only Lexsi submission checklist (HTML; NOT SUBMITTED)"**.
This is a more legible counterpart to the existing source-checked JSON. It
includes the exact historical under-500-word abstract (unless newer evidence
makes the text stale), the field values, the remaining owner actions, the
deadline and conspicuous missing-model/dataset-collection warnings.

For a reproducible local, write-once HTML copy without opening a browser or
uploading data, run:

```bash
python -m scripts.lexsi_submission_offline_html
```

Open `artifacts/lexsi_track2a_owner_form_offline_UNSUBMITTED.html` locally.
The producer is strictly offline, output-allowlisted, and refuses to overwrite
a previous file. Private registration email and HF token are never requested
or embedded. Its browser page has a restrictive Content Security Policy and
no script, login, form submission, clipboard access or persistence. The only
external link points to the actual Lexsi submission form for **manual** use.

A successful native-toolkit CI run also uploads this exact HTML inside the
existing **owner form draft UNSUBMITTED** artifact alongside the JSON. The
HTML renderer refuses inconsistent readiness, stale-but-exposed abstracts,
self-reported publication completion and collection links lacking the source
handoff's fresh source-bound anonymous-access receipt. It does **not** independently
verify external access by itself; that is done by the existing source-bound
handoff/verifier upstream.

**This is not a final submission.** Once actual HF publication/72-case
measurements change the evidence, regenerate rather than reusing an old HTML
copy. Never infer that an offline form or successful tests establish rights,
independent semantic evaluation, organizer acceptance or a real fine-tune.
