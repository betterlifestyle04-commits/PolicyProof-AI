"""Read-only research dashboard: streamlit run research_dashboard.py.

Does not run AI inference. Synthetic scores are explicitly identified.
"""
from __future__ import annotations
import json
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="PolicyProof | Research Observatory", page_icon="🛡️", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1250px; padding-top: 2rem;}
h1, h2, h3 {letter-spacing: -.03em;}
[data-testid="stMetricValue"] {font-weight: 800;}
</style>""", unsafe_allow_html=True)


def jsonl(file):
    if not file.exists():
        return []
    return [json.loads(x) for x in file.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_json(file):
    if not file.is_file():
        return None
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError):
        return None


st.title("PolicyProof AI")
st.caption("LEXSI 2026  /  MULTILINGUAL AI SAFETY RESEARCH  /  PROTOTYPE")
st.warning("No verified model fine-tuning or safety improvement yet. All policies are fictional and benchmark questions are synthetic.")

pages = ["Research overview", "Dataset explorer", "Evaluation runs", "Protocol & next steps"]
page = st.sidebar.radio("Explore", pages)
st.sidebar.markdown("[GitHub source](https://github.com/betterlifestyle04-commits/PolicyProof-AI)")
st.sidebar.caption("For educational research only. No banking advice.")

if page == "Research overview":
    st.subheader("Experiment status")
    manifest = load_json(DATA / "manifest.json")
    if not manifest:
        st.info("Run python -m scripts.prepare to generate the dataset.")
    counts = {k: len(jsonl(DATA / n)) for k, n in {
        "Training": "train.jsonl", "Paraphrase test": "test.jsonl",
        "Unseen-policy test": "unseen_policy_test.jsonl"}.items()}
    c = st.columns(3)
    for col, (name, count) in zip(c, counts.items()):
        col.metric(name, count)
    st.markdown("""**Scientific objective.** Test whether multilingual training improves grounded answers,
    correct citations, and abstention on held-out fictional policies. Compare the unchanged base model
    against a tuned model using identical test examples, controls, and documented checkpoints.""")
    st.info("Software fixture verification is not the same as model performance. Only actual model-produced predictions can be used for research claims.")
    with st.expander("Dataset manifest and provenance"):
        st.json(manifest or {"status": "not generated"})

elif page == "Dataset explorer":
    st.subheader("Evidence and evaluation questions")
    split = st.selectbox("Sample collection", ["train", "test", "unseen_policy_test"])
    rows = jsonl(DATA / f"{split}.jsonl")
    if not rows:
        st.info("Generate fixtures with python -m scripts.prepare.")
    else:
        language = st.selectbox("Language", ["All", "en", "no", "ar"])
        category = st.selectbox("Case type", ["All"] + sorted({r.get("category", "standard") for r in rows}))
        selected = [r for r in rows if (language == "All" or r["language"] == language)
                    and (category == "All" or r.get("category", "standard") == category)]
        st.caption(f"{len(selected)} cases · Questions are synthetic, not independently human-reviewed")
        st.dataframe([{"ID": r["id"], "Language": r["language"],
                       "Question": r["question"], "Source": r["expected_source_id"],
                       "Abstain": r["should_abstain"]} for r in selected], use_container_width=True)
        if selected:
            r = st.selectbox("Inspect evidence", selected, format_func=lambda r: f'{r["id"]}: {r["question"][:65]}')
            a, b = st.columns(2)
            a.markdown("**Evidence supplied to the model**")
            a.code(r["context"], language="text")
            b.markdown("**Reference label, not a model response**")
            b.json({"answer": r["expected_answer"], "source_id": r["expected_source_id"],
                    "abstain": r["should_abstain"]})

elif page == "Evaluation runs":
    st.subheader("Recorded evaluation artifacts")
    # Read original, preserved user-provided model predictions from a nested
    # archive directory as well as local mock and evaluation run artifacts.
    runs = sorted(ARTIFACTS.rglob("*.json")) if ARTIFACTS.exists() else []
    outputs = [(f, load_json(f)) for f in runs]
    outputs = [(f, o) for f, o in outputs if isinstance(o, dict) and "metrics" in o and "engine" in o]
    if not outputs:
        st.info("No evaluation runs yet. Try the documented mock smoke tests first.")
    else:
        chosen = st.selectbox("Artifact", outputs, format_func=lambda o: o[0].name)
        file, report = chosen
        engine = report.get("engine")
        if str(engine).startswith("mock"):
            st.error("SIMULATED TEST HARNESS. These metrics MUST NOT be cited as model performance.")
        elif engine == "predictions":
            st.warning("Imported external predictions: provenance unverified. These are not independently confirmed model results.")
        else:
            st.warning("Model inference reported by artifact. Verify model ID, checkpoint hashes, training logs and run provenance separately.")
        st.caption(f"Engine: {engine} | Split: {report.get('split', 'unknown')} | Model: {report.get('model', 'n/a')}")
        m = report["metrics"]
        labels = ["strict_accuracy_pct", "json_schema_valid_pct", "correct_citation_pct",
                  "abstention_recall_pct", "unsupported_claim_rate_pct"]
        cols = st.columns(len(labels))
        for col, k in zip(cols, labels):
            val = m.get(k)
            col.metric(k.replace("_pct", "").replace("_", " ").title(),
                       "N/A" if val is None else f"{val:.1f}%")
        st.subheader("Language slices")
        stats = m.get("by_language", {})
        if stats:
            st.bar_chart({k: v["strict_accuracy_pct"] for k, v in stats.items()})
        if engine == "transformers":
            st.caption(
                "Preserved model outputs from the owner's Colab run, not independently "
                "re-inferred here. The original report lacks a model-revision SHA. "
                "This is a development benchmark, not a final blind evaluation."
            )
            if report.get("is_finetuned") is not True:
                st.info("This is the **unchanged base model**. No trained adapter or improvement is claimed.")
        if report.get("predictions"):
            with st.expander("Inspect saved model responses (all fictional policies)"):
                # Show raw text; never reinterpret missing JSON keys as correct outputs.
                answers = report["predictions"]
                st.dataframe(
                    [{"ID": row.get("id"), "Raw model response": row.get("raw", "")}
                     for row in answers],
                    use_container_width=True,
                    hide_index=True,
                )
        with st.expander("Raw score rows and file provenance"):
            st.json({"dataset_sha256": report.get("dataset_sha256"),
                     "model_revision": report.get("model_revision") or "not recorded",
                     "is_real_model_result": report.get("is_real_model_result"),
                     "is_finetuned": report.get("is_finetuned"),
                     "scorer_version": report.get("scorer_version"),
                     "warning": report.get("warning")})
            st.dataframe(report.get("scores", []), use_container_width=True)

else:
    st.subheader("Experimental protocol")
    st.markdown("""
    1. Human-review and version original policy examples and multilingual translations.
    2. Curate training data using Lexsi **CuratorKIT**. Inspect provenance and split isolation.
    3. Accept gated **Tiny Aya** license and measure unchanged baseline on identical held-out cases.
    4. Train using **AlignTune** on approved hardware. Record exact recipe and checkpoint hash.
    5. Score baseline and tuned checkpoints using **AuditKIT** plus deterministic checks.
    6. Report paired bootstrap intervals, language slices, harmful failures, and negative results.
    7. Publish reproducible artifacts and submit before the announced deadline.
    """)
    st.caption("Do not claim production safety, source accuracy, or improvement from synthetic/mock figures.")
