"""Static CPU-only checks for opt-in Colab GPU and exact source-bound I/O.

These tests compile cells, verify guards, and review their original storage
contract. They never execute a GPU model, Hugging Face login, Google Drive
mount, network request, training run, or human evaluation.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/12_track2a_72_drive_resumable_t4.ipynb"
TRUSTED_SHA = "af89d219b53ed9b13b8a4645f9c8028973510324"
FROZEN = "cdd5cf4e54caa6c79ce1f0cd5bd0f05344f0dc22039a2ad56a3c1c0ee2d32cd1"


def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
    assert len(notebook["cells"]) >= 17
    assert all(isinstance(c["source"], list) for c in notebook["cells"])
    assert all(c.get("outputs") == [] and c.get("execution_count") is None
               for c in notebook["cells"] if c["cell_type"] == "code")
    return notebook["cells"]


def code_cells():
    return ["".join(c["source"]) for c in cells() if c["cell_type"] == "code"]


def test_every_colab_code_cell_syntax_checks_without_gpu_or_dependencies():
    code = code_cells()
    assert len(code) >= 8
    for index, cell in enumerate(code):
        compile(cell, f"Stage12_GPU_NOT_EXECUTED_cell_{index}", "exec")
    assert sum(c.count("import torch") for c in code) >= 1


def test_all_license_gpu_score_export_switches_default_off():
    settings = code_cells()[0]
    for setting in (
        "VERIFY_HF_ACCOUNT_AND_MODEL_SHA",
        "MODEL_LICENSE_AND_COST_APPROVED",
        "RUN_ORIGINAL_72_GPU",
        "RUN_CPU_AUDITKIT",
        "EXPORT_RESULTS",
        "EXPORT_PARTIAL_BACKUP",
    ):
        assert f"{setting} = False" in settings
        assert not any(line.strip().startswith(setting + " = True")
                       for cell in code_cells() for line in cell.splitlines())
    assert f'MODEL_SHA = "{TRUSTED_SHA}"' in settings
    assert "TRAINING" in settings
    assert not any("torch.optim" in s or "trainer.train(" in s
                   for s in code_cells())


def test_genuine_gpu_call_only_inside_explicit_owner_guard_and_never_faked():
    code = code_cells()
    gpu = [c for c in code if "execute_gpu(" in c]
    assert len(gpu) == 1
    tree = ast.parse(gpu[0])
    permitted = [
        node for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "RUN_ORIGINAL_72_GPU"
    ]
    assert len(permitted) == 1
    body = ast.unparse(permitted[0])
    assert "MODEL_LICENSE_AND_COST_APPROVED is True" in body
    assert "REMOTE_SHA_VERIFIED is True" in body
    assert "not original_report.exists()" in body
    assert "checkpoint_dir=CHECKPOINT" in body
    assert "resume=RESUME_ORIGINAL_CHECKPOINT" in body
    assert "original_report" in gpu[0]
    assert not any("make_gold(" in cell or "fake_gpu(" in cell
                   for cell in code)


def test_real_auditkit_scoring_and_export_are_separate_opt_in_steps():
    code = code_cells()
    audit = [c for c in code if "score_saved_72(" in c]
    assert len(audit) == 1
    ast.parse(audit[0])
    assert "if RUN_CPU_AUDITKIT:" in audit[0]
    assert "verify_completed_72(" in audit[0]
    assert "target_dir.exists()" in audit[0]
    export = [c for c in code if "OWNER_SCOPE_MANIFEST.json" in c]
    assert len(export) == 1
    assert "if EXPORT_RESULTS:" in export[0]
    assert "verify_completed_72(" in export[0]
    assert "PARTIAL_UNVERIFIED_checkpoint.zip" in export[0]
    assert "read_checkpoint(" in export[0]
    assert '"original_gpu_provenance_independently_verified": False' in export[0]
    assert '"human_semantic_review_complete": False' in export[0]
    assert '"official_lexsi_submission_completed": False' in export[0]


def test_private_drive_not_published_and_original_code_unchanged_on_resume():
    code = code_cells()
    bootstrap = next(s for s in code if "drive.mount(" in s)
    assert "if SAVE_CHECKPOINTS_IN_PRIVATE_DRIVE:" in bootstrap
    assert "drive.mount(" in bootstrap
    assert "MyDrive/PolicyProof_Lexsi_72" in bootstrap
    assert "remote" in bootstrap and "get-url" in bootstrap
    assert "rev-parse" in bootstrap
    assert "git\", \"clone\"" in bootstrap
    assert "pull" not in bootstrap and "reset" not in bootstrap
    assert "git\", \"push\"" not in "\n".join(code)
    assert "ROOT / FROZEN_DATA" in "\n".join(code)
    preflight = next(s for s in code if "RESUME_ORIGINAL_CHECKPOINT" in s)
    assert "KNOWN_CHALLENGE_SHA256" in preflight
    assert "progress_dir.exists()" in preflight
    assert "preflight(ROOT" in preflight
    assert "READY_FOR_EXPLICIT_GPU_EXECUTION" in preflight


def test_no_hardcoded_tokens_api_calls_or_unreviewed_gold_scores():
    blob = "\n".join(code_cells())
    assert "hf_" not in blob
    assert "ghp_" not in blob
    assert "GITHUB_TOKEN" not in blob
    assert "os.environ[\"HF_TOKEN\"]" not in blob
    assert "github.com/betterlifestyle04-commits/PolicyProof-AI.git" in blob
    assert "notebook_login()" in blob
    assert "MODEL_SHA" in blob
    assert "original_gpu_provenance_independently_verified" in blob
    assert "TRAINING" in blob


def test_markdown_warns_original_72_unscored_and_independent_review_missing():
    doc = "\n".join(
        "".join(c["source"]) for c in cells() if c["cell_type"] == "markdown"
    )
    assert FROZEN in doc
    assert "72" in doc and "unscored" in doc.lower()
    assert "independent" in doc.lower()
    assert "final-60" in doc.lower()
    assert "noncommercial" in doc.lower()
    assert "review" in doc.lower()
    assert "private" in doc.lower()
    assert "notebook tests" in doc.lower()
