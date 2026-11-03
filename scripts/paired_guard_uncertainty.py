"""Small-sample paired evaluation of the already-saved output-guard experiment.

No model inference, no tuning, no user data. Descriptive McNemar sign
tests are computed *only on discordant paired cases*; they do not establish
real-world safety or semantic quality. No scipy dependency.
"""
from __future__ import annotations
from collections import Counter
from math import comb, sqrt
from pathlib import Path
import argparse
import json

from policyproof.core import ROOT, read_jsonl, sha256, score
from scripts.inference_guard import safe_output
from scripts.auditkit_replay import check_saved_outputs


def wilson_interval(successes: int, n: int, z: float = 1.95996398454) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*sqrt((p*(1-p)/n) + (z*z/(4*n*n)))/denom
    return [round(max(0,100*(center-half)),2),
            round(min(100,100*(center+half)),2)]


def exact_discordance_p(improved: int, regressed: int) -> float:
    """Two-sided exact McNemar/binomial test, no asymptotic approximation."""
    if min(improved, regressed) < 0:
        raise ValueError("Counts cannot be negative")
    n = improved + regressed
    if not n:
        return 1.0
    smaller = min(improved, regressed)
    return min(1.0, 2 * sum(comb(n,k) for k in range(smaller+1)) / (2**n))


def paired_measure(before: list[dict], after: list[dict], *, field: str,
                   denominator: str | None = None, direction: int = 1) -> dict:
    if len(before)!=len(after) or any(a["id"]!=b["id"] for a,b in zip(before,after)):
        raise ValueError("Pair IDs must match in the same order")
    if direction not in (-1,1):
        raise ValueError("Direction must be +1 for beneficial success, -1 for beneficial reduction")
    pairs = [(b,a) for b,a in zip(before,after) if denominator is None or b[denominator]]
    raw_before=sum(b[field] for b,_ in pairs)
    raw_after=sum(a[field] for _,a in pairs)
    improved=sum((a[field]-b[field])*direction == 1 for b,a in pairs)
    regressed=sum((a[field]-b[field])*direction == -1 for b,a in pairs)
    assert improved+regressed<=len(pairs)
    return {
        "denominator":len(pairs),
        "raw_before":raw_before,"raw_after":raw_after,
        "rate_before_pct":round(raw_before/len(pairs)*100,2) if pairs else None,
        "rate_after_pct":round(raw_after/len(pairs)*100,2) if pairs else None,
        "wilson_95_before_pct":wilson_interval(raw_before,len(pairs)),
        "wilson_95_after_pct":wilson_interval(raw_after,len(pairs)),
        "improved_cases":improved,"regressed_cases":regressed,
        "unchanged_cases":len(pairs)-improved-regressed,
        "exploratory_exact_mcnemar_two_sided_p":round(exact_discordance_p(improved,regressed),8),
        "beneficial_direction":"higher" if direction==1 else "lower",
    }


def analyze(baseline: dict, rows: list[dict], digest: str) -> dict:
    saved=check_saved_outputs(baseline,rows,digest)
    before=[]
    after=[]
    for row in rows:
        original=saved[row["id"]]
        guarded=safe_output(original,context=row["context"]).output
        before.append(score(row, original))
        after.append(score(row, guarded))
    metrics = {
        "schema_compliance":("format_valid",None,1),
        "strict_exact":("correct",None,1),
        "correct_citation_on_supported":("citation_correct","supported_total",1),
        "exact_answer_on_supported":("supported_answer_correct","supported_total",1),
        "correct_abstention_on_unsupported":("abstention_correct","abstention_needed",1),
        "unsupported_claim_on_unsupported":("unsupported_claim","abstention_needed",-1),
        "accepted_answer_coverage":("answered",None,1),
    }
    result={}
    for name,(field,den,sign) in metrics.items():
        result[name]=paired_measure(before,after,field=field,denominator=den,direction=sign)
    languages={}
    for language in ("en","no","ar"):
        b=[s for s in before if s["lang"]==language]
        a=[s for s in after if s["lang"]==language]
        languages[language]={
            "strict_exact": paired_measure(b,a,field="correct"),
            "schema_compliance": paired_measure(b,a,field="format_valid"),
            "supported_exact":paired_measure(b,a,field="supported_answer_correct",denominator="supported_total"),
        }
    return {
        "status":"descriptive_small_sample_analysis",
        "samples":len(rows),"original_base_model_reported":baseline.get("model"),
        "model_revision_independently_verified":False,
        "training_performed":False,"new_model_inference_performed":False,
        "inspected_development_data":True,"dataset_sha256":digest,
        "metrics":result,"languages":languages,
        "limitations":[
            "Exact McNemar p values are descriptive, exploratory, not a safety-certification test",
            "Wilson intervals reflect within-sample binomial uncertainty, not distribution shift or model epistemic uncertainty",
            "This is a tiny synthetic, previously inspected development set; no independent semantic review",
            "Schema compliance is enforced by construction, not improved model understanding",
            "The change in unsupported abstention is only ONE case out of NINE; strong generalization is unsupported",
            "The guard does NOT improve exact supported-answer correctness (0/39 both times)",
            "No fine-tuned adapter, no real-world banking evaluation, no blinded model comparison",
        ],
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",default="artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json")
    p.add_argument("--dataset",default="data/unseen_policy_test.jsonl")
    p.add_argument("--output",default="artifacts/paired_guard_uncertainty.json")
    a=p.parse_args()
    original=Path(a.baseline)
    cases=Path(a.dataset)
    target=Path(a.output)
    if target.resolve() in (original.resolve(),cases.resolve()):
        raise ValueError("Cannot overwrite original evidence")
    report=analyze(json.loads(original.read_text(encoding="utf-8")),
                   read_jsonl(cases),sha256(cases))
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("Descriptive paired research diagnostics, NOT a trained-model result:")
    for k,m in report["metrics"].items():
        print(f"{k}: {m['raw_before']}/{m['denominator']} -> {m['raw_after']}/{m['denominator']}; "
              f"improved={m['improved_cases']} regressed={m['regressed_cases']} "
              f"p={m['exploratory_exact_mcnemar_two_sided_p']}")
    print("Saved",target)


if __name__=="__main__":
    main()
