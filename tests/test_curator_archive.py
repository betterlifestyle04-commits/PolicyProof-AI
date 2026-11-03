"""CuratorKIT ZIP checker negative tests without external packages."""
import hashlib
import io
import json
import zipfile

import pytest
from scripts.verify_curator_archive import check_zip


def make_archive(path, *, malicious=False, tamper=False):
    prompt = (
        'Return JSON. Answer in English.\n'
        'POLICY: [POL-001] A fictional account has no monthly fee.\n'
        'QUESTION: What is the monthly fee?'
    )
    row = {"instruction": prompt, "input": "",
           "output": json.dumps({"answer":"Zero", "source_id":"POL-001", "abstain":False})}
    payload = {
        "manifest.json": json.dumps({"stage_counts":{"AlpacaExporter":{"exported_count":1}},
                                      "tool_versions":{"curatorkit":"0.1.22"}}).encode(),
        "lexsi_provenance.json": b'{"library":"curatorkit"}',
        "rejected.jsonl": b"",
        "sft_alpaca.jsonl": (json.dumps(row)+"\n").encode(),
        "sft_sharegpt.jsonl": b'{"conversations":[{"from":"human","value":"hello"}]}\n',
    }
    hashes = "".join(hashlib.sha256(value).hexdigest()+"  "+name+"\n" for name,value in payload.items())
    payload["checksums.txt"] = hashes.encode()
    if tamper:
        payload["sft_alpaca.jsonl"] += b"corruption"
    if malicious:
        payload["../unsafe.txt"] = b"untrusted"
    with zipfile.ZipFile(path,"w") as z:
        for name,value in payload.items():
            z.writestr(name,value)


def test_valid_minimal_zip(tmp_path):
    path=tmp_path/"clean.zip"
    make_archive(path)
    result=check_zip(path)
    assert result["row_count"]==1
    assert result["abstentions"]==0
    assert result["status"]=="pass"


def test_reject_file_modification(tmp_path):
    path=tmp_path/"bad.zip"
    make_archive(path,tamper=True)
    with pytest.raises(ValueError,match="Checksum verification failed"):
        check_zip(path)


def test_reject_unsafe_archive_member(tmp_path):
    path=tmp_path/"unsafe.zip"
    make_archive(path,malicious=True)
    with pytest.raises(ValueError,match="Unsafe archive"):
        check_zip(path)
