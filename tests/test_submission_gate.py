"""Submission gate must never equate CPU research checks to trained model quality."""
from pathlib import Path
import subprocess
import sys

from policyproof.core import ROOT
from scripts.submission_gate import evaluate_readiness


def test_current_worktree_not_ready_without_real_checkpoint():
    state=evaluate_readiness(ROOT)
    assert state["status"]=="NOT_READY"
    assert state["ready_for_submission"] is False
    assert state["checks"]["saved_baseline"]["passed"] is True
    assert state["checks"]["trained_adapter"]["passed"] is False
    assert state["checks"]["paired_finetuned_evaluation"]["passed"] is False
    assert state["checks"]["huggingface_artifacts"]["passed"] is False


def test_empty_worktree_is_not_ready(tmp_path):
    report=evaluate_readiness(tmp_path)
    assert report["ready_for_submission"] is False
    assert report["checks"]["saved_baseline"]["passed"] is False


def test_strict_command_fails_when_submission_not_ready():
    result=subprocess.run([sys.executable,"-m","scripts.submission_gate","--strict"],
        cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert result.returncode==2
    assert '"status": "NOT_READY"' in result.stdout
