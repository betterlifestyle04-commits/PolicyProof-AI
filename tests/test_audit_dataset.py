"""Dataset audit regression tests; validates adversarial corruption is caught."""
import json
from pathlib import Path
import shutil
import pytest
from scripts.audit_dataset import audit

ROOT = Path(__file__).resolve().parents[1]


def fixture_dir(tmp_path):
    out = tmp_path / "data"
    out.mkdir()
    for name in ("train.jsonl", "test.jsonl", "unseen_policy_test.jsonl", "manifest.json"):
        shutil.copy(ROOT / "data" / name, out / name)
    return out


def mutate_jsonl(path, mutate):
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")


def test_happy_path_dataset_audit():
    result = audit(ROOT / "data")
    assert result["result"] == "pass"
    assert result["counts"] == {"train": 78, "paraphrase": 42, "unseen_policy": 48}
    assert result["train_unseen_policy_overlap"] == 0


def test_detects_stale_hash(tmp_path):
    out = fixture_dir(tmp_path)
    (out / "train.jsonl").write_text((out / "train.jsonl").read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        audit(out)


def test_detects_leaked_policy(tmp_path):
    out = fixture_dir(tmp_path)
    def change(rows):
        rows[0]["context"] += "\n[POL-001] injected training policy"
    mutate_jsonl(out / "unseen_policy_test.jsonl", change)
    with pytest.raises(ValueError, match="leakage"):
        audit(out)


def test_detects_bad_gold_citation(tmp_path):
    out = fixture_dir(tmp_path)
    def change(rows):
        rows[0]["expected_source_id"] = "POL-DOES-NOT-EXIST"
    mutate_jsonl(out / "test.jsonl", change)
    with pytest.raises(ValueError, match="gold citation"):
        audit(out)


def test_detects_duplicate_ids(tmp_path):
    out = fixture_dir(tmp_path)
    rows = [json.loads(s) for s in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    def change(other):
        other[0]["id"] = rows[0]["id"]
    mutate_jsonl(out / "test.jsonl", change)
    with pytest.raises(ValueError, match="duplicate ID"):
        audit(out)
