"""Official Lexsi CuratorKIT integration. Execute only after dependency installation."""
import argparse
from policyproof.core import ROOT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="artifacts/curated")
    a = p.parse_args()
    source = ROOT / "data/train_alpaca.jsonl"
    if not source.exists():
        raise SystemExit("Prepare data first: python -m scripts.prepare")
    try:
        from curatorkit import Curator, CuratorConfig
    except ImportError as e:
        raise SystemExit('Install: pip install "curatorkit[connectors]"') from e
    target = ROOT / a.output
    result = Curator(CuratorConfig(
        dataset=str(source), clean=True, dedup="exact",
        export_formats=["alpaca", "sharegpt"], output_dir=str(target)
    )).run()
    result.print_summary()
    if not (target / "manifest.json").exists():
        raise SystemExit("CuratorKIT manifest missing — do not train")
    print("Inspect manifest, data quality, export configs, provenance and license before training.")

if __name__ == "__main__":
    main()
