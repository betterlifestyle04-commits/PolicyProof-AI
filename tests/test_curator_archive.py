"""Integrity and zip-slip tests for official CuratorKIT export validator."""
import hashlib
import json
import zipfile

import pytest
from scripts.verify_curator_archive import check_zip


def make_archive(path, *, malicious=False, tamper=False):
    alpaca = []
    share = []
    for lang in ("English", "Norwegian", "Arabic"):
        prompt = (
            "Return JSON. Answer in " + lang + ".\n"
            "POLICY:\n[POL-001] The fictional account has no monthly fee.\n"
            "QUESTION:\nWhat is the monthly fee?"
        )
        reply = json.dumps({"answer": "Zero", "source_id": "POL-001", "abstain": False})
        alpaca.append({"instruction": prompt, "input": "", "output": reply})
        share.append({"conversations": [{"from": "human", "value": prompt},
                                        {"from": "gpt", "value": reply}]})
    payload = {
        "manifest.json": json.dumps({
            "stage_counts": {"AlpacaExporter": {"exported_count": 3}},
            "tool_versions": {"curatorkit": "0.1.22"},
        }).encode(),
        "lexsi_provenance.json": b'{"library":"curatorkit"}',
        "rejected.jsonl": b"",
        "sft_alpaca.jsonl": "".join(json.dumps(row) + "\n" for row in alpaca).encode(),
        "sft_sharegpt.jsonl": "".join(json.dumps(row) + "\n" for row in share).encode(),
    }
    checksums = "".join(
        hashlib.sha256(content).hexdigest() + "  " + name + "\n"
        for name, content in payload.items()
    )
    payload["checksums.txt"] = checksums.encode()
    if tamper:
        payload["sft_alpaca.jsonl"] += b"corruption"
    if malicious:
        payload["../unsafe.txt"] = b"untrusted"
    with zipfile.ZipFile(path, "w") as z:
        for name, content in payload.items():
            z.writestr(name, content)


def test_valid_minimal_zip(tmp_path):
    path = tmp_path / "clean.zip"
    make_archive(path)
    report = check_zip(path)
    assert report["row_count"] == 3
    assert report["languages"] == {"English": 1, "Norwegian": 1, "Arabic": 1}
    assert report["abstentions"] == 0
    assert report["status"] == "pass"


def test_reject_modified_file(tmp_path):
    path = tmp_path / "corrupt.zip"
    make_archive(path, tamper=True)
    with pytest.raises(ValueError, match="Checksum verification failed"):
        check_zip(path)


def test_reject_archive_path_traversal(tmp_path):
    path = tmp_path / "unsafe.zip"
    make_archive(path, malicious=True)
    with pytest.raises(ValueError, match="Unsafe archive"):
        check_zip(path)
