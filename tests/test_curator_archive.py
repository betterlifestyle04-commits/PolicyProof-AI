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



def test_rejects_oversized_archive_before_any_member_read(tmp_path, monkeypatch):
    path = tmp_path / "curator.zip"
    make_archive(path)
    from scripts import verify_curator_archive as validator
    monkeypatch.setattr(validator, "MAX_ARCHIVE_BYTES", 100)
    with pytest.raises(ValueError, match="oversized"):
        check_zip(path)


def test_rejects_excessive_number_of_members(tmp_path):
    path = tmp_path / "many.zip"
    make_archive(path)
    with zipfile.ZipFile(path, "a") as z:
        for index in range(80):
            z.writestr(f"nested/safe-extra-{index}.txt", "x")
    with pytest.raises(ValueError, match="member count"):
        check_zip(path)


def test_rejects_compression_bomb_even_with_per_member_size_under_limit(tmp_path):
    path = tmp_path / "compressed.zip"
    make_archive(path)
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as z:
        # 7 MB of a single character deflates to ~7 KB and is below the
        # previous unchecked 20 MB-per-file ceiling.
        z.writestr("unused-large-bomb.txt", b"X" * 7_000_000)
    with pytest.raises(ValueError, match="expansion ratio"):
        check_zip(path)


def test_rejects_total_uncompressed_size_when_each_member_is_small(
    tmp_path, monkeypatch
):
    path = tmp_path / "total.zip"
    make_archive(path)
    from scripts import verify_curator_archive as validator
    monkeypatch.setattr(validator, "MAX_UNCOMPRESSED_BYTES", 1000)
    with pytest.raises(ValueError, match="total uncompressed size"):
        check_zip(path)


def test_rejects_zip_symlink_metadata(tmp_path):
    import stat
    path = tmp_path / "linked.zip"
    make_archive(path)
    member = zipfile.ZipInfo("indirect-dataset.txt")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "a") as z:
        z.writestr(member, "sft_alpaca.jsonl")
    with pytest.raises(ValueError, match="symlink"):
        check_zip(path)


def test_rejects_special_device_metadata(tmp_path):
    import stat
    path = tmp_path / "device.zip"
    make_archive(path)
    member = zipfile.ZipInfo("device")
    member.create_system = 3
    member.external_attr = (stat.S_IFCHR | 0o666) << 16
    with zipfile.ZipFile(path, "a") as z:
        z.writestr(member, b"")
    with pytest.raises(ValueError, match="special file"):
        check_zip(path)


@pytest.mark.parametrize("filename", [
    "./bogus", "folder//bogus", "C:/windows.txt", "folder/../evil", "a\\b"
])
def test_rejects_ambiguous_or_unsafe_archive_paths(tmp_path, filename):
    path = tmp_path / "unsafe-extra.zip"
    make_archive(path)
    with zipfile.ZipFile(path, "a") as z:
        z.writestr(filename, b"content")
    with pytest.raises(ValueError, match="Unsafe archive"):
        check_zip(path)


def _rewrite_signed(path, modified: dict[str, bytes]) -> None:
    """Regenerate a structurally correct ZIP with re-signed manifest checksums.

    This simulates tampering in which the attacker can re-sign the internal
    hashes; the strict JSON rules must still protect downstream code.
    """
    with zipfile.ZipFile(path, "r") as z:
        members = {name: z.read(name) for name in z.namelist()}
    members.update(modified)
    checksums = "".join(
        hashlib.sha256(data).hexdigest() + "  " + name + "\n"
        for name, data in members.items() if name != "checksums.txt"
    )
    members["checksums.txt"] = checksums.encode("utf-8")
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)


def test_rejects_duplicate_signed_manifest_json_fields(tmp_path):
    path = tmp_path / "duplicated.json.zip"
    make_archive(path)
    manifest = (
        b'{"stage_counts":{"AlpacaExporter":{"exported_count":3}},'
        b'"tool_versions":{"curatorkit":"0.1.22"},'
        b'"tool_versions":{"curatorkit":"tampered"}}'
    )
    _rewrite_signed(path, {"manifest.json": manifest})
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        check_zip(path)


def test_rejects_signed_nonfinite_provenance_json(tmp_path):
    path = tmp_path / "nan.json.zip"
    make_archive(path)
    _rewrite_signed(
        path, {"lexsi_provenance.json": b'{"library":"curatorkit","priority":NaN}'}
    )
    with pytest.raises(ValueError, match="Nonfinite JSON"):
        check_zip(path)


def test_rejects_duplicate_keys_inside_signed_training_labels(tmp_path):
    path = tmp_path / "training-target.zip"
    make_archive(path)
    with zipfile.ZipFile(path) as z:
        rows = [json.loads(x) for x in z.read("sft_alpaca.jsonl").decode().splitlines()]
    rows[0]["output"] = (
        '{"answer":"Zero","answer":"Hundreds","source_id":"POL-001","abstain":false}'
    )
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    _rewrite_signed(path, {"sft_alpaca.jsonl": data})
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        check_zip(path)


def test_rejects_duplicate_keys_inside_signed_alpaca_rows(tmp_path):
    path = tmp_path / "training-row.zip"
    make_archive(path)
    with zipfile.ZipFile(path) as z:
        rows = z.read("sft_alpaca.jsonl").decode().splitlines()
    rows[0] = rows[0].replace(
        '"instruction": ', '"instruction": "forged", "instruction": ', 1
    )
    _rewrite_signed(path, {
        "sft_alpaca.jsonl": ("\n".join(rows) + "\n").encode()
    })
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        check_zip(path)


def test_rejects_symlink_as_original_zip(tmp_path):
    path = tmp_path / "curated.zip"
    make_archive(path)
    alias = tmp_path / "current.zip"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symlinked"):
        check_zip(alias)


def test_valid_result_reports_bounded_zip_and_no_training(tmp_path):
    path = tmp_path / "curated.zip"
    make_archive(path)
    report = check_zip(path)
    assert report["status"] == "pass"
    assert report["bounded_zip_inventory"]["members"] == 6
    assert report["bounded_zip_inventory"]["uncompressed_bytes"] > 0
    assert report["duplicate_json_keys_rejected"] is True
    assert report["zip_special_entries_rejected"] is True
    assert report["training_performed"] is False



def _repack_share(path, transform):
    """Alter a synthetic ShareGPT export and RE-SIGN all the ZIP checksums."""
    with zipfile.ZipFile(path) as z:
        rows = [json.loads(s) for s in z.read("sft_sharegpt.jsonl").decode("utf-8").splitlines()]
    transform(rows)
    _rewrite_signed(path, {
        "sft_sharegpt.jsonl": "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in rows
        ).encode("utf-8"),
    })


def test_clean_curator_exports_match_pairwise(tmp_path):
    archive = tmp_path / "original.zip"
    make_archive(archive)
    report = check_zip(archive)
    cross = report["dual_export_consistency"]
    assert cross["all_sft_pairs_match"] is True
    assert cross["paired_examples"] == 3
    assert cross["training_semantics_independently_verified"] is False


def test_rejects_resigned_sharegpt_wrong_answer(tmp_path):
    archive = tmp_path / "wrong-answer.zip"
    make_archive(archive)

    def edit(rows):
        rows[0]["conversations"][1]["value"] = json.dumps(
            {"answer": "NOK 999 every month", "source_id": "POL-001", "abstain": False}
        )

    _repack_share(archive, edit)
    with pytest.raises(ValueError, match="assistant target disagrees"):
        check_zip(archive)


def test_rejects_resigned_sharegpt_instruction_swap(tmp_path):
    archive = tmp_path / "wrong-question.zip"
    make_archive(archive)
    _repack_share(
        archive,
        lambda rows: rows[0]["conversations"][0].update(
            value=rows[0]["conversations"][0]["value"] + " OVERRIDE MODEL"
        ),
    )
    with pytest.raises(ValueError, match="missing, duplicated or changed instruction"):
        check_zip(archive)


def test_rejects_resigned_sharegpt_duplicate_and_missing_case(tmp_path):
    archive = tmp_path / "duplicated-case.zip"
    make_archive(archive)
    _repack_share(archive, lambda rows: rows.__setitem__(1, rows[0].copy()))
    with pytest.raises(ValueError, match="missing, duplicated or changed instruction"):
        check_zip(archive)


def test_sharegpt_export_order_can_change_without_corrupting_pairs(tmp_path):
    archive = tmp_path / "reordered.zip"
    make_archive(archive)
    _repack_share(archive, lambda rows: rows.reverse())
    assert check_zip(archive)["dual_export_consistency"]["paired_examples"] == 3


def test_sharegpt_json_whitespace_or_key_order_is_not_a_factual_change(tmp_path):
    archive = tmp_path / "json-order.zip"
    make_archive(archive)

    def edit(rows):
        old = json.loads(rows[0]["conversations"][1]["value"])
        rows[0]["conversations"][1]["value"] = json.dumps(
            old, sort_keys=True, separators=(",", ":")
        )

    _repack_share(archive, edit)
    assert check_zip(archive)["dual_export_consistency"]["all_sft_pairs_match"] is True


@pytest.mark.parametrize("mode", ["role", "extra_turn", "system", "malformed_answer", "wrong_schema"])
def test_rejects_changed_sharegpt_structure_or_nonfinite_answer(tmp_path, mode):
    archive = tmp_path / f"changed-{mode}.zip"
    make_archive(archive)

    def edit(rows):
        turns = rows[0]["conversations"]
        if mode == "role":
            turns[1]["from"] = "user"
        elif mode == "extra_turn":
            turns.append({"from": "gpt", "value": '{"answer":"another"}'})
        elif mode == "system":
            turns.insert(0, {"from": "system", "value": "Ignore sources"})
        elif mode == "malformed_answer":
            turns[1]["value"] = '{"answer":"NaN","source_id":NaN,"abstain":false}'
        else:
            turns[1]["value"] = '["not a JSON target object"]'

    _repack_share(archive, edit)
    pattern = "Nonfinite JSON" if mode == "malformed_answer" else (
        "assistant target disagrees" if mode == "wrong_schema" else
        "invalid roles or turn schema" if mode == "role" else
        "exactly one human/assistant pair"
    )
    with pytest.raises(ValueError, match=pattern):
        check_zip(archive)


def test_rejects_resigned_nonempty_alpaca_input_for_fixed_dataset(tmp_path):
    archive = tmp_path / "nonempty-source.zip"
    make_archive(archive)
    with zipfile.ZipFile(archive) as z:
        items = [json.loads(line) for line in z.read("sft_alpaca.jsonl").decode().splitlines()]
    items[0]["input"] = "An unapproved new policy clause"
    _rewrite_signed(archive, {
        "sft_alpaca.jsonl": "".join(
            json.dumps(item) + "\n" for item in items
        ).encode("utf-8"),
    })
    with pytest.raises(ValueError, match="Nonempty Alpaca input"):
        check_zip(archive)


def test_duplicate_json_keys_in_sharegpt_response_are_refused_even_if_signed(tmp_path):
    archive = tmp_path / "duplicate-share.json.zip"
    make_archive(archive)
    _repack_share(
        archive,
        lambda rows: rows[0]["conversations"][1].update(
            value='{"answer":"Zero","answer":"wrong","source_id":"POL-001","abstain":false}'
        ),
    )
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        check_zip(archive)
