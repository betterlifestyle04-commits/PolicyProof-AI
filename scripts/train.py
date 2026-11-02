"""AlignTune GPU run. Nothing is trained without --execute."""
import argparse
from policyproof.core import ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="CohereLabs/tiny-aya-global")
    p.add_argument("--dataset", default="artifacts/curated")
    p.add_argument("--dataset-config", default="sft_sharegpt")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    target = ROOT / a.dataset
    if not a.execute:
        print("DRY RUN ONLY — model:", a.model, "dataset:", target,
              "config:", a.dataset_config, "steps:", a.max_steps)
        print("Need accepted gated HF license, validated CuratorKIT exports, and CUDA GPU.")
        return
    if not (target / "manifest.json").exists():
        raise SystemExit("CuratorKIT export manifest missing; abort training")
    try:
        from aligntune.core.backend_factory import create_sft_trainer
    except ImportError as e:
        raise SystemExit("Missing AlignTune package") from e
    # Confirm the installed version supports this model and chosen dataset config
    # before spending credits. 3.35B model likely requires quantized adapters.
    trainer = create_sft_trainer(model_name=a.model, dataset_name=str(target),
                                config_name=a.dataset_config, backend="trl",
                                num_epochs=a.epochs, max_steps=a.max_steps,
                                batch_size=a.batch_size, learning_rate=a.learning_rate)
    trainer.train()
    print("Trainer returned; inspect checkpoint files and logs before claiming success.")

if __name__ == "__main__":
    main()
