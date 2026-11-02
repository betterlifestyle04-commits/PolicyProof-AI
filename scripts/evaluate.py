"""PolicyProof evaluation. Mock scores are NOT trained-model achievements."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from policyproof.core import ROOT, read_jsonl, sha256, make_gold, make_prompt, score, aggregate

def predict_transformers(model):
    try:
        from transformers import pipeline
    except ImportError as e:
        raise SystemExit("Install transformers and accept gated model license.") from e
    pipe = pipeline("text-generation", model=model, device_map="auto")
    def run(prompt):
        output = pipe([{"role": "user", "content": prompt}], max_new_tokens=160, do_sample=False)
        value = output[0]["generated_text"]
        if isinstance(value, list):
            return value[-1]["content"]
        return value[len(prompt):] if isinstance(value, str) and value.startswith(prompt) else value
    return run

def evaluate_auditkit(rows, lookup):
    """Optional official AuditKIT diagnostic, not a replacement for our citation scorer."""
    import auditkit as ak
    samples = [ak.Sample(input=make_prompt(x), target=make_gold(x["expected_answer"], x["expected_source_id"]))
               for x in rows]
    by_prompt = {make_prompt(x): lookup[x["id"]] for x in rows}
    result = ak.evaluate(samples, model=lambda prompts: [by_prompt[p] for p in prompts],
                         metrics=["exact_match", "f1_score"])
    return str(result.summary())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["paraphrase", "unseen-policy"], default="unseen-policy")
    parser.add_argument("--engine", choices=["mock-oracle", "mock-abstain", "predictions", "transformers"],
                        default="mock-abstain")
    parser.add_argument("--model", default="CohereLabs/tiny-aya-global")
    parser.add_argument("--predictions", help="Unverified external JSONL with {id,prediction}")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    parser.add_argument("--auditkit", action="store_true")
    a = parser.parse_args()
    input_path = ROOT / ("data/test.jsonl" if a.split == "paraphrase" else "data/unseen_policy_test.jsonl")
    if not input_path.exists():
        raise SystemExit("Run python -m scripts.prepare first")
    rows = read_jsonl(input_path)
    if a.engine == "mock-oracle":
        lookup = {x["id"]: make_gold(x["expected_answer"], x["expected_source_id"]) for x in rows}
    elif a.engine == "mock-abstain":
        lookup = {x["id"]: make_gold("INSUFFICIENT_EVIDENCE", None) for x in rows}
    elif a.engine == "predictions":
        if not a.predictions:
            parser.error("--predictions is required for --engine predictions")
        items = read_jsonl(Path(a.predictions))
        lookup = {x["id"]: x["prediction"] for x in items}
        if len(lookup) != len(items) or set(lookup) != {x["id"] for x in rows}:
            raise SystemExit("Prediction IDs must match the entire evaluation split exactly")
    else:
        model = predict_transformers(a.model)
        lookup = {x["id"]: model(make_prompt(x)) for x in rows}
    scores = [score(x, lookup[x["id"]]) for x in rows]
    report = dict(
        project="PolicyProof AI / Lexsi 2026", created_utc=datetime.now(timezone.utc).isoformat(),
        engine=a.engine, split=a.split, model=a.model if a.engine == "transformers" else None,
        dataset_sha256=sha256(input_path), is_real_model_result=(a.engine == "transformers"),
        warning=("Software-only mock score, not actual model performance" if a.engine.startswith("mock-") else
                 "External prediction provenance unverified; do not report as model performance" if a.engine == "predictions" else
                 "Real model inference; check checkpoint and training provenance manually"),
        metrics=aggregate(scores), scores=scores,
        predictions=[dict(id=x["id"], raw=lookup[x["id"]]) for x in rows])
    if a.auditkit:
        report["auditkit_summary"] = evaluate_auditkit(rows, lookup)
    dest = ROOT / a.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(dest), "engine": a.engine, "metrics": report["metrics"]}, indent=2))

if __name__ == "__main__":
    main()
