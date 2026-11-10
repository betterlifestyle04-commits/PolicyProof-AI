"""Stage15 must not train, infer, expose owner mapping or forge reviewer votes."""
from __future__ import annotations

import json
from pathlib import Path


def notebook():
    p=(Path(__file__).resolve().parents[1] /
       "notebooks/15_native40_fresh_gold_rescore_and_blind_human_review.ipynb")
    return json.loads(p.read_text(encoding="utf-8"))


def test_all_stage15_python_cells_compile():
    book=notebook()
    assert book["nbformat"] == 4
    for cell in book["cells"]:
        if cell["cell_type"]=="code":
            compile("".join(cell["source"]),"stage15", "exec")


def test_stage15_has_original_immutable_bundle_hash_and_cpu_only():
    source="\n".join("".join(cell["source"]) for cell in notebook()["cells"])
    assert "e0319c83fbb87ef8570ca0c42caf68bea4a4fef1f8ae320bbe11e9a7c1761c68" in source
    assert "rescore_original_bundle(ROOT,BUNDLE,EXPECTED)" in source
    assert "all_original_saved_case_scores_exactly_reproduced" in source
    assert "RUN_NATIVE_TRAINING = True" not in source
    assert "trainer.train(" not in source
    assert "model.generate(" not in source
    assert "RUN_DEV48_GPU_EVAL = True" not in source
    assert "hf_hub_upload(" not in source


def test_stage15_blinded_export_is_explicitly_opt_in_and_secret_separated():
    book=notebook()
    code="\n".join("".join(cell["source"]) for cell in book["cells"]
                   if cell["cell_type"]=="code")
    assert "EXPORT_PRIVATE_BLINDED_REVIEW_PACKET = False" in code
    assert "if EXPORT_PRIVATE_BLINDED_REVIEW_PACKET:" in code
    assert "dev48_blank.csv" in code
    assert "dev48_private_mapping.json" in code
    assert "human_semantic_review_performed" in code
    assert "files.download(str(ROOT/\"reviews/dev48_blank.csv\"))" in code
    assert "files.download(str(ROOT/\"reviews/dev48_private_mapping.json\"))" in code
    assert '"reviewer_code":"invented"' not in code
    assert "semantic_correct = 1" not in code


def test_judge_report_explains_no_human_safety_or_final_blind_claim():
    root=Path(__file__).resolve().parents[1]
    report=(root/"docs/NATIVE40_FRESH_GOLD_RESCORE_2026-11-11.md").read_text()
    assert "96 original raw responses" in report
    assert "human" in report.lower()
    assert "4/9" in report and "2/9" in report
    assert "0/39" in report
    assert "previously inspected" in report
    assert "NOT been completed" in report
