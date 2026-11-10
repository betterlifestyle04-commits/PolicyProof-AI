"""Notebook structural safety checks; does not execute remote notebook code."""
from pathlib import Path
import json
import pytest
from scripts.validate_notebooks import validate_notebook


@pytest.mark.parametrize("file", [
    "02_curatorkit_training_data.ipynb",
    "03_aligntune_qlora_t4.ipynb",
    "04_kaggle_gpu_fallback.ipynb",
    "11_native_aligntune_40step_t4.ipynb",
])
def test_committed_notebooks_have_valid_code_and_safe_defaults(file):
    path=Path(__file__).resolve().parents[1]/"notebooks"/file
    report=validate_notebook(path)
    assert report["code_cells"]>=3
    assert report["syntax_verified"] is True
    assert report["executed"] is False


def test_notebook_syntax_error_is_rejected(tmp_path):
    path=tmp_path/"broken.ipynb"
    cells=[{"cell_type":"code","source":"def broken(:\n    pass",
            "metadata":{},"execution_count":None,"outputs":[]}
           for _ in range(3)]
    path.write_text(json.dumps({"nbformat":4,"nbformat_minor":5,"cells":cells}))
    with pytest.raises(ValueError,match="invalid Python syntax"):
        validate_notebook(path)


def test_notebook_rejects_enabled_training(tmp_path):
    path=tmp_path/"03_aligntune_qlora_t4.ipynb"
    cells=[{"cell_type":"code","source":"RUN_TRAINING=True\n",
            "metadata":{},"execution_count":None,"outputs":[]}
           for _ in range(3)]
    path.write_text(json.dumps({"nbformat":4,"cells":cells}))
    with pytest.raises(ValueError, match="automatic training"):
        validate_notebook(path)


def test_notebook_rejects_hardcoded_token(tmp_path):
    path=tmp_path/"secret.ipynb"
    cells=[{"cell_type":"code","source":'TOKEN="hf_'+("X"*32)+'"\n',
            "metadata":{},"execution_count":None,"outputs":[]}
           for _ in range(3)]
    path.write_text(json.dumps({"nbformat":4,"cells":cells}))
    with pytest.raises(ValueError,match="hardcoded token"):
        validate_notebook(path)



def test_new_standalone_native40_notebook_is_immutable_training_off_by_default():
    project = Path(__file__).resolve().parents[1]
    source_path = project/"notebooks/11_native_aligntune_40step_t4.ipynb"
    report = validate_notebook(source_path)
    assert "RUN_NATIVE_TRAINING" in report["training_flags_off"]
    assert report["executed"] is False
    notebook = json.loads(source_path.read_text())
    source = "\n".join(
        "".join(x["source"]) if isinstance(x["source"], list) else x["source"]
        for x in notebook["cells"] if x["cell_type"] == "code"
    )
    assert "optimizer_steps=40" in source
    assert "loaded_revision_evidence=loaded_revision" in source
    assert "RUN_NATIVE_TRAINING = False" in source


@pytest.mark.parametrize("replacement", [
    "RUN_NATIVE_TRAINING = True",
    "RUN_NATIVE_TRAINING = 1",
])
def test_native40_notebook_rejects_auto_training(tmp_path, replacement):
    origin = Path(__file__).resolve().parents[1]/"notebooks/11_native_aligntune_40step_t4.ipynb"
    notebook = json.loads(origin.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "RUN_NATIVE_TRAINING = False" in src:
                cell["source"] = src.replace(
                    "RUN_NATIVE_TRAINING = False", replacement
                )
                break
    target = tmp_path / "11_native_aligntune_40step_t4.ipynb"
    target.write_text(json.dumps(notebook))
    if replacement.endswith("True"):
        with pytest.raises(ValueError, match="automatic training"):
            validate_notebook(target)
    else:
        with pytest.raises(ValueError, match="native 40-step run must default OFF"):
            validate_notebook(target)


def test_native40_notebook_cannot_remove_artifact_verification(tmp_path):
    origin = Path(__file__).resolve().parents[1]/"notebooks/11_native_aligntune_40step_t4.ipynb"
    notebook = json.loads(origin.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = "".join(cell["source"]).replace(
                "inspect_training_archive(", "do_not_inspect_training_archive("
            )
    target = tmp_path / "11_native_aligntune_40step_t4.ipynb"
    target.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="native 40-step run must default OFF"):
        validate_notebook(target)

def test_native40_notebook_cannot_bypass_text_only_sdk_environment_preflight(tmp_path):
    origin = Path(__file__).resolve().parents[1] / "notebooks/11_native_aligntune_40step_t4.ipynb"
    notebook = json.loads(origin.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = "".join(cell["source"]).replace(
                "env_report = verify_text_only_native_env()",
                "env_report = {'status': 'faked'}",
            )
    target = tmp_path / "11_native_aligntune_40step_t4.ipynb"
    target.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="native 40-step run must default OFF"):
        validate_notebook(target)

