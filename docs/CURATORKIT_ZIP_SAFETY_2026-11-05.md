# CuratorKIT archive safety boundary — November 5, 2026

This is a **training-data provenance and ZIP integrity safeguard**, not a model-quality or real-world policy correctness result. The original 78-example CuratorKIT data are synthetic, and no AlignTune optimizer steps are executed by this protection.

## Why this change was needed

The previous validator required essential ZIP members and checked per-file SHA-256 hashes but did not reject all oversized *aggregate* expansion, very high compression ratios, symlinks/special Unix members, anomalous names, or duplicate JSON fields and `NaN` in trusted-looking metadata. An oversized or ambiguous archive could therefore reach the restoration path.

## How the validator now fails closed

The updated `scripts/verify_curator_archive.py` checks the archive before any member is extracted:

| Control | New boundary |
| --- | --- |
| Total compressed ZIP file | At most 35 MB; symlinked source ZIP rejected |
| Member inventory | At most 80 directory/file entries |
| Declared total decompressed file data | At most 60 MB |
| Individual file | At most 20 MB |
| Compression expansion | At most 500:1 per nonempty file |
| Checksums metadata | At most 100 KB |
| JSON manifest | At most 1 MB |
| Provenance JSON | At most 250 KB |
| Paths | No absolute, traversal, backslashes, Windows drive prefixes, control characters, invalid segments or excessive depth |
| Unix entry types | No symlink, device, FIFO or other special file |
| Structured data | No duplicate JSON keys or nonfinite JSON literals |

The integrity checker streams member SHA-256 computation, preserving existing per-entry hash verification. ZIP archive declarations alone are **not** trusted to prove original authorship or accurate semantics. An attacker able to re-sign all archive members can still forge internally consistent *synthetic* metadata; the independently pinned original training-file SHA in the Colab restore workflow is a separate requirement.

## Extraction and source-swap defense

The updated `restore_curated_archive` computes the submitted ZIP's SHA-256 before checking the archive, after validation, and again after staging. Any source modification between validation and extraction aborts before destination writes. It then verifies the exact original 78-row training dataset digest, stages allowlisted files outside the target directory, and refuses to overwrite existing conflicting files.

The accepted original **training JSONL SHA-256** is `4b14eb262b52eaba00794f070a4e69ae543bd2d13d060d2d06e4832c1a84294c`. A current-upload ZIP digest generated during restoration proves that the bytes did not change *during this process*; it is **not** an independent signature of origin.

## What runs where

1. [CuratorKIT training-data export notebook](../notebooks/02_curatorkit_training_data.ipynb) runs `check_train_data`, compares the original SHA, creates the ZIP, and invokes `check_zip` before offering the download.
2. [AlignTune/T4 training notebook](../notebooks/03_aligntune_qlora_t4.ipynb) calls the strict `restore_curated_archive` before model loading. Existing different files are never overwritten.
3. Continuous integration exercises malicious and benign synthetic ZIP fixtures: excessive membership, compression bombs, aggregate expansion, Unix symlinks and devices, traversal, signed duplicate JSON fields, nonfinite JSON, source swapping and idempotent restore.

To validate a local export without a GPU:

```bash
python -m scripts.verify_curator_archive /path/to/policyproof_curatorkit_train_v1.zip
```

If verification fails, **stop and inspect the original file**. Do not turn off hash checks, increase archive limits, or relabel a different export as the original merely to unblock training. If an otherwise legitimate newer CuratorKIT version changes the export, treat it as a distinct curation experiment and re-audit dataset provenance separately.

### Remaining limitations

- Tests execute on synthetic CPU fixtures; we did not rerun the owner's original historical ZIP in this GitHub CI.
- These controls cannot establish independent curation quality, training completion, human policy correctness or submission acceptance.
- The full original ZIP's authorship is not cryptographically signed. SHA-256 protects a specifically chosen source from modification only after that source is independently authenticated.
