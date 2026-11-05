# Pinned AlignTune and TRL source compatibility check — November 5, 2026

**Research integrity: installed SDK source compatibility, not GPU runtime validation.**

We previously relied on synthetic/mocked AlignTune training-config tests. The sixth GitHub Actions CI job downloads two REAL published SDK distributions: **AlignTune 0.1.28 and TRL 1.7.1**, strictly pinned, using pip's no-dependencies mode. It analyzes the **installed original Python source** via importlib.metadata and the standard-library AST parser, without importing the ML stacks.

The official upstream AlignTune 0.1.28 tag is Git commit '57e9a95e79ace15014beede175c4c3d72405f3b3'. Its published 'create_sft_trainer' factory puts leftover arguments in 'train.extra_params'; 'build_sft_config' constructs a TRL SFTConfig and applies the 'extract_extra_and_missing_params' stage *afterward*. That late stage can overwrite preselected precision flags and training budgets. We now explicitly audit the installed implementation, rather than relying on our fake alone.

## The source-compatibility gate checks

1. Exact installed versions, no floating/automatic upgrades.
2. Safe original source modules for the AlignTune factory, TRL builder, extractor, precision config, and TRL SFTConfig.
3. The real factory still wires explicit training arguments into the model/train configuration and preserves late 'extra_params=kwargs' behavior.
4. The real SFT builder still creates a TRL SFTConfig with 'max_steps' and 'output_dir', then runs the late parameter extractor using the expected arguments.
5. The extractor still merges its extra parameters, and TRL's SFTConfig still exposes the 'max_length' declaration.
6. The CI artifact records SHA-256 hashes of the installed Python source modules and explicit limitations. Any unexpected source structure fails closed.

## Reproduction, CPU only

Run from the repository root in a fresh Python 3.11 environment:

    python -m pip install --no-deps 'aligntune==0.1.28' 'trl==1.7.1'
    python -m scripts.check_pinned_align_sdk_contract
    python -m pytest -q tests/test_pinned_align_sdk_contract.py

The sixth CI job uploads the JSON receipt as 'policyproof-real-pinned-align-trl-sdk-contract-cpu'.

## Crucial limits

The wheels are installed without their heavyweight ML dependencies. We do **not** import either package, instantiate the real TRL SFTConfig, load Tiny Aya weights, run CUDA kernels, execute AlignTune optimizer steps, or infer any model improvements. Those are distinct validations. The actual runtime and GPU checks can only be evidenced by a real, fresh Stage 11 T4 session with saved adapter, raw logs, and an independently checked model revision.

This test improves upstream compatibility monitoring. It does not make PolicyProof AI clinically or financially reliable, establish release eligibility, authenticate any human review, or confirm official hackathon submission.

See [Stage 11 native AlignTune procedure](NATIVE_ALIGNTUNE_40STEP_T4_2026-11-05.md) and [release evidence gates](RELEASE_EVIDENCE_GATES_2026-11-05.md).
