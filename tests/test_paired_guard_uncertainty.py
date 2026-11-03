import json
import math

from policyproof.core import ROOT, read_jsonl, sha256
from scripts.paired_guard_uncertainty import wilson_interval, exact_discordance_p, paired_measure, analyze


def test_wilson_interval():
    bounds = wilson_interval(1,9)
    assert 0 < bounds[0] < 11.12 < bounds[1] < 50
    assert wilson_interval(0,0) is None


def test_exact_discordance_pvalues():
    assert exact_discordance_p(1,0)==1.0
    assert math.isclose(exact_discordance_p(15,0), 0.00006103515625)
    assert exact_discordance_p(0,0)==1.0


def test_pair_ids_checked():
    try:
        paired_measure([{"id":"a","correct":1}],[{"id":"b","correct":0}],field="correct")
    except ValueError:
        pass
    else:
        raise AssertionError("Pair mismatch allowed")


def test_original_baseline_48_pairs():
    original = json.loads((ROOT/"artifacts/real_baseline/tiny_aya_base_unseen_policy-rescored-v2.json").read_text())
    dataset = ROOT/"data/unseen_policy_test.jsonl"
    result = analyze(original,read_jsonl(dataset),sha256(dataset))
    assert result["samples"]==48
    assert result["training_performed"] is False
    assert result["metrics"]["schema_compliance"]["improved_cases"]==15
    assert result["metrics"]["schema_compliance"]["regressed_cases"]==0
    a=result["metrics"]["correct_abstention_on_unsupported"]
    assert (a["raw_before"],a["raw_after"],a["denominator"])==(8,9,9)
    assert a["exploratory_exact_mcnemar_two_sided_p"]==1.0
    answer=result["metrics"]["exact_answer_on_supported"]
    assert (answer["raw_before"],answer["raw_after"],answer["denominator"])==(0,0,39)
