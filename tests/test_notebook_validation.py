"""Notebook structural safety checks; does not execute remote notebook code."""
from pathlib import Path
import json
import pytest
from scripts.validate_notebooks import validate_notebook


@pytest.mark.parametrize("file", [
    "02_curatorkit_training_data.ipynb",
    "03_aligntune_qlora_t4.ipynb",
    "04_kaggle_gpu_fallback.ipynb",
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
