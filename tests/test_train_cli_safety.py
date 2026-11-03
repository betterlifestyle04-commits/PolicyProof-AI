"""Verify training CLI cannot silently launch full 3.35B-parameter training."""
import subprocess
import sys


def test_default_train_cli_is_safe_dry_run():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.train"],
        capture_output=True, text=True, timeout=15
    )
    assert proc.returncode == 0
    assert "DRY RUN ONLY" in proc.stdout
    assert "zero optimizer steps" in proc.stdout
    assert "verified_curator_train" in proc.stdout


def test_execute_fails_closed_before_importing_heavy_ml():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.train", "--execute"],
        capture_output=True, text=True, timeout=15
    )
    assert proc.returncode != 0
    assert "Unsafe legacy full-parameter training path disabled" in proc.stderr
    assert "QLoRA" in proc.stderr or "notebooks" in proc.stderr


def test_train_cli_help_is_available():
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.train", "--help"],
        capture_output=True, text=True, timeout=15
    )
    assert proc.returncode == 0
    assert "--execute" in proc.stdout
