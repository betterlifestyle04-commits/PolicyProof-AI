"""Strict paired comparison: excludes simulated and unverified prediction files."""
import argparse
import json
from pathlib import Path
import random
from policyproof.core import aggregate

def compare(base, tuned, iterations=4000, seed=2026):
    if any(r.get("engine") != "transformers" or not r.get("is_real_model_result")
           for r in (base, tuned)):
        raise ValueError("Only actual Transformers inference reports can be compared")
    if base.get("dataset_sha256") != tuned.get("dataset_sha256") or base.get("split") != tuned.get("split"):
        raise ValueError("Dataset bytes and split must match exactly")
    a = {x["id"]: x for x in base["scores"]}
    b = {x["id"]: x for x in tuned["scores"]}
    if not a or set(a) != set(b) or len(a) != len(base["scores"]) or len(b) != len(tuned["scores"]):
        raise ValueError("Scores contain duplicate or mismatched sample IDs")
    ids = sorted(a)
    for i in ids:
        if a[i]["lang"] != b[i]["lang"] or a[i]["category"] != b[i]["category"]:
            raise ValueError("Inconsistent sample metadata")
    differences = [b[i]["correct"] - a[i]["correct"] for i in ids]
    n = len(differences)
    rng = random.Random(seed)
    boots = sorted(100 * sum(differences[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(iterations))
    return dict(split=base["split"], samples=n,
                base_model=base["model"], tuned_model=tuned["model"],
                base_metrics=aggregate([a[i] for i in ids]),
                tuned_metrics=aggregate([b[i] for i in ids]),
                paired_accuracy_delta_pp=round(100 * sum(differences) / n, 2),
                paired_bootstrap_95_ci_pp=[round(boots[int(.025 * iterations)], 2),
                                           round(boots[min(iterations - 1, int(.975 * iterations))], 2)],
                seed=seed, iterations=iterations,
                limitation="Synthetic questions and exact strings; not verified real-world safety")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--tuned", required=True)
    p.add_argument("--output", default="artifacts/comparison.json")
    a = p.parse_args()
    base = json.loads(Path(a.base).read_text(encoding="utf-8"))
    tuned = json.loads(Path(a.tuned).read_text(encoding="utf-8"))
    data = compare(base, tuned)
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(target, data["paired_accuracy_delta_pp"], data["paired_bootstrap_95_ci_pp"])

if __name__ == "__main__":
    main()
