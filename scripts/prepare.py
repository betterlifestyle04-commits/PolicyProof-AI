"""Prepare deterministic fixture data; does not call a model or spend GPU credits."""
import json
from policyproof.core import ROOT, prepare_rows, make_prompt, make_gold, sha256, write_jsonl

def main():
    fixture = json.loads((ROOT / "data/policies.json").read_text(encoding="utf-8"))
    train, para, unseen = prepare_rows(fixture)
    inputs = {
        "train.jsonl": train, "test.jsonl": para,
        "unseen_policy_test.jsonl": unseen,
        "train_alpaca.jsonl": [
            {"instruction": make_prompt(x), "input": "", "output": make_gold(x["expected_answer"], x["expected_source_id"])}
            for x in train
        ],
        "train_messages.jsonl": [
            {"messages": [{"role": "user", "content": make_prompt(x)},
                          {"role": "assistant", "content": make_gold(x["expected_answer"], x["expected_source_id"])}]}
            for x in train
        ]
    }
    for name, rows in inputs.items():
        write_jsonl(ROOT / "data" / name, rows)
    manifest = {
        "dataset": "fictional-policyproof-research-fixtures", "synthetic": True,
        "human_reviewed": False, "counts": {"train": len(train), "paraphrase": len(para), "unseen": len(unseen)},
        "warning": "Question variants are template-generated; no real model results or safety claims.",
        "sha256": {name: sha256(ROOT / "data" / name) for name in inputs}
    }
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Prepared 78 train + 42 paraphrase + 48 unseen-policy cases (all synthetic).")

if __name__ == "__main__":
    main()
