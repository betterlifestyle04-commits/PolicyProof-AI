"""AlignTune configuration-only preflight for the curated PolicyProof dataset.

NO model weights, inference or GPU training are started by this module.
The official create_sft_trainer() factory instantiates a TRL trainer wrapper;
the heavyweight setup_model(), train() and save_model() calls are not made.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import json
import re

from policyproof.core import read_jsonl, sha256


def check_train_data(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing curated training file: {path}")
    rows = read_jsonl(path)
    if len(rows) != 78:
        raise ValueError(f"Expected 78 original synthetic training examples, found {len(rows)}")
    required = {"instruction", "input", "output"}
    languages: Counter[str] = Counter()
    unknown = 0
    for i, row in enumerate(rows):
        if set(row) != required:
            raise ValueError(f"Training row {i}: incorrect Alpaca columns")
        instruction, target = row["instruction"], json.loads(row["output"])
        found = re.findall(r"Answer in (English|Norwegian|Arabic)\.", instruction)
        if len(found) != 1:
            raise ValueError(f"Training row {i}: cannot identify language")
        languages[found[0]] += 1
        context_ids = set(re.findall(r"\[(POL-[A-Za-z0-9-]+)\]", instruction))
        if not context_ids or any(not re.fullmatch(r"POL-\d{3}", x) for x in context_ids):
            raise ValueError(f"Training row {i}: unseen or malformed policy ID present")
        if set(target) != {"answer", "source_id", "abstain"}:
            raise ValueError(f"Training row {i}: malformed target keys")
        if type(target["abstain"]) is not bool:
            raise ValueError(f"Training row {i}: invalid abstention type")
        if target["abstain"]:
            unknown += 1
            if target["answer"] != "INSUFFICIENT_EVIDENCE" or target["source_id"] is not None:
                raise ValueError(f"Training row {i}: malformed abstention target")
        elif target["source_id"] not in context_ids:
            raise ValueError(f"Training row {i}: unsupported target source ID")
    if dict(languages) != {"English": 26, "Norwegian": 26, "Arabic": 26}:
        raise ValueError(f"Unexpected language counts: {languages}")
    return {"row_count": len(rows), "languages": dict(languages),
            "abstention_count": unknown, "dataset_sha256": sha256(path)}


def make_training_wrapper(dataset_dir: Path):
    """Factory wrapper only; no model weights or training."""
    from aligntune.core.backend_factory import create_sft_trainer
    return create_sft_trainer(
        model_name="CohereLabs/tiny-aya-global",
        dataset_name=str(dataset_dir),
        config_name="sft_alpaca",
        backend="trl",
        task_type="instruction_following",
        split="train",
        output_dir=str(dataset_dir.parent / "align_qlora_smoke"),
        num_epochs=1,
        max_steps=10,
        batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_seq_length=1024,
        instruction_column="instruction",
        input_column="input",
        response_column="output",
        peft_enabled=True,
        lora_r=8, lora_alpha=16, lora_dropout=0.05,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        quantization={"load_in_4bit": True,
                      "bnb_4bit_compute_dtype": "float16",
                      "bnb_4bit_quant_type": "nf4"},
        use_gradient_checkpointing=True,
        use_flash_attention_2=False,
        attn_implementation="sdpa",
        fp16=True, bf16=False,
        report_to="none",
        precision="fp16",
        seed=42,
    )


def instantiate_config_only(dataset_dir: Path):
    trainer = make_training_wrapper(dataset_dir)
    config = trainer.config
    assert config.model.peft.enabled is True
    assert config.model.peft.rank == 8
    assert config.model.quantization.get("load_in_4bit") is True
    assert config.train.max_steps == 10
    assert config.train.per_device_batch_size == 1
    assert config.model.max_seq_length == 1024
    assert config.dataset.config_name == "sft_alpaca"
    assert trainer.model is None, "Factory unexpectedly loaded model weights"
    return {
        "trainer_class": type(trainer).__name__,
        "backend": "trl",
        "model": config.model.name_or_path,
        "config_name": config.dataset.config_name,
        "training_steps": config.train.max_steps,
        "micro_batch": config.train.per_device_batch_size,
        "max_seq_length": config.model.max_seq_length,
        "peft_enabled": config.model.peft.enabled,
        "lora_rank": config.model.peft.rank,
        "quantized_4bit": config.model.quantization["load_in_4bit"],
        "loaded_model_weights": False,
        "trained_model": False,
        "model_specific_compatibility_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Safe AlignTune factory dry run")
    parser.add_argument("--curated-dir", default="artifacts/verified_curator_train")
    args = parser.parse_args()
    directory = Path(args.curated_dir).resolve()
    data = check_train_data(directory / "sft_alpaca.jsonl")
    cfg = instantiate_config_only(directory)
    print(json.dumps({"status": "config_only_valid", "data": data, "trainer": cfg},
                     ensure_ascii=False, indent=2))
    print("No model weights loaded; no GPU training performed.")


if __name__ == "__main__":
    main()
