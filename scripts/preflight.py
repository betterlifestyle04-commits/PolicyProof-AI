"""Read-only hardware/dependency preflight. Never prints credentials or API tokens."""
from __future__ import annotations
import argparse
from importlib import metadata, util
import json
import os
from platform import python_version
from pathlib import Path
import sys

PACKAGES = {"curatorkit": "curatorkit", "aligntune": "aligntune", "auditkit": "auditkit",
            "transformers": "transformers", "datasets": "datasets",
            "accelerate": "accelerate", "peft": "peft", "torch": "torch"}


def collect() -> dict:
    package_info = {}
    for display, module in PACKAGES.items():
        available = util.find_spec(module) is not None
        try:
            version = metadata.version(display) if available else None
        except metadata.PackageNotFoundError:
            version = "unknown"
        package_info[display] = {"installed": available, "version": version}
    cuda = {"available": False, "device_count": 0, "devices": []}
    if package_info["torch"]["installed"]:
        try:
            import torch
            cuda["available"] = bool(torch.cuda.is_available())
            cuda["device_count"] = int(torch.cuda.device_count())
            if cuda["available"]:
                cuda["devices"] = [torch.cuda.get_device_name(i)
                                   for i in range(cuda["device_count"])]
        except (OSError, RuntimeError, ImportError):
            pass
    token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    missing = [p for p, x in package_info.items() if not x["installed"]]
    return {"python": python_version(), "python_311_plus": sys.version_info[:2] >= (3, 11),
            "libraries": package_info, "cuda": cuda,
            "hf_environment_token_present": token_present,
            "model_license_accepted": "not checked; verify on Hugging Face website",
            "missing_libraries": missing,
            "training_preflight_ready": sys.version_info[:2] >= (3, 11) and
                not missing and cuda["available"] and token_present,
            "caution": "Installed tools and token do NOT prove model access, credit balance or tool API compatibility."}


def main():
    ap = argparse.ArgumentParser(description="Check hardware and dependencies without printing secrets")
    ap.add_argument("--output", help="Optional JSON report path")
    ap.add_argument("--strict", action="store_true", help="Exit 2 unless all stated prerequisites are present")
    args = ap.parse_args()
    report = collect()
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict and not report["training_preflight_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
