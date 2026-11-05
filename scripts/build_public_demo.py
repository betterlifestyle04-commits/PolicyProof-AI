"""Reproduce public CPU-only judge-demo artifacts during build.
No GPU, model/token, external API call, or user secrets required.
Fails the deployment if any baseline integrity check fails.
"""
from __future__ import annotations
from pathlib import Path
import json
import subprocess
import sys

from policyproof.core import ROOT


def execute_module(module: str, *args: str) -> None:
    command = [sys.executable, "-m", module, *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    original = "artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json"
    cases = "data/unseen_policy_test.jsonl"
    execute_module("scripts.prepare")
    execute_module("scripts.validate_final_holdout")
    execute_module("scripts.verify_evidence", "--baseline", original,
                   "--dataset", cases, "--out", "artifacts/verified_real_baseline.json")
    execute_module("scripts.auditkit_replay", "--report", original,
                   "--dataset", cases, "--output", "artifacts/auditkit_actual_baseline_replay.json")
    execute_module("scripts.inference_guard", "--baseline", original,
                   "--dataset", cases, "--output", "artifacts/guarded_development_evaluation.json")
    execute_module("scripts.paired_guard_uncertainty", "--baseline", original,
                   "--dataset", cases,
                   "--output", "artifacts/paired_guard_uncertainty.json")
    execute_module("scripts.numeric_evidence_audit", "--report", original,
                   "--dataset", cases, "--output", "artifacts/dev48_numeric_risk_audit.json")
    execute_module("scripts.source_topic_audit", "--report", original,
                   "--dataset", cases, "--output", "artifacts/dev48_topic_relevance_audit.json")
    execute_module("scripts.submission_gate", "--output", "artifacts/submission_gate.json")
    # Must be NOT_READY until legitimate fine-tune, paired results & HF evidence exist.
    state=json.loads((ROOT/"artifacts/submission_gate.json").read_text(encoding="utf-8"))
    if state.get("ready_for_submission") or state.get("status") != "NOT_READY":
        raise RuntimeError("Demo release unexpectedly claims complete submission")
    print("PUBLIC CPU DEMO READY: baseline, AuditKIT, guard, numeric/topic triage, NOT_READY receipt verified")


if __name__ == "__main__":
    main()
