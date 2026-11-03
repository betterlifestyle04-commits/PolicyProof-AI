"""PolicyProof AI training CLI safety gate.

The historical \`--execute\` route did not configure 4-bit QLoRA/PEFT and could
attempt full 3.35B-parameter training on a T4. It is intentionally disabled
until the official AlignTune trainer has a *verified* supported recipe.
Use the Stage 3/4 notebooks for gated restoration and explicit training.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from policyproof.core import ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="CohereLabs/tiny-aya-global")
    p.add_argument("--dataset", default="artifacts/verified_curator_train")
    p.add_argument("--dataset-config", default="sft_alpaca")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--execute", action="store_true",
                   help="Disabled until official AlignTune 4-bit recipe is verified")
    a = p.parse_args()
    root = (ROOT / a.dataset).resolve()
    if a.execute:
        p.error(
            "Unsafe legacy full-parameter training path disabled. "
            "The official AlignTune/TRL path has not completed a verified "
            "T4 optimizer step. Use the documented notebooks/03_aligntune_qlora_t4.ipynb "
            "for model setup, or the explicit PyTorch fallback in "
            "notebooks/04_kaggle_gpu_fallback.ipynb. Do not claim the fallback "
            "demonstrates official AlignTune training."
        )
    print("DRY RUN ONLY — No model loaded; zero optimizer steps.")
    print("model:", a.model)
    print("training_data:", root)
    print("config:", a.dataset_config)
    print("steps:", a.max_steps)
    print("batch_size:", a.batch_size)
    print("learning_rate:", a.learning_rate)
    print("An approved gated model, verified CuratorKIT ZIP and available CUDA GPU are still required.")


if __name__ == "__main__":
    main()
