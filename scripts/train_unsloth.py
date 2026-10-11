#!/usr/bin/env python3
"""Fine-tune an instruct model with Unsloth QLoRA and TRL SFTTrainer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validate_dataset import as_messages, read_records, validate_records

DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
DEFAULT_SYSTEM = "You are a careful, evidence-grounded assistant specialized in financial modeling. State assumptions and units clearly."


def load_and_check(path: Path) -> list[dict]:
    records = read_records(path)
    errors, _ = validate_records(records)
    if errors:
        raise ValueError("; ".join(errors[:8]))
    return records


def make_prompt_completion(record: dict, tokenizer) -> dict:
    messages = as_messages(record)
    if not messages:
        raise ValueError("record has no messages")
    # If the record is Alpaca-style, add a consistent system instruction.
    if "messages" not in record:
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})
    answer = messages[-1]["content"].strip()
    prompt_messages = messages[:-1]
    prompt = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    eos = getattr(tokenizer, "eos_token", None) or ""
    return {"prompt": prompt, "completion": answer + eos}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--val-file", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-coder-lora"))
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument("--preflight-only", action="store_true", help="Validate files/config without importing ML packages or downloading models")
    args = parser.parse_args()
    if args.max_seq_length < 128 or args.rank < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("max-seq-length must be >= 128 and rank/batch/gradient-accumulation must be positive")
    if args.max_steps == 0 or args.eval_ratio < 0 or args.eval_ratio >= 1:
        parser.error("max-steps cannot be zero; eval-ratio must be in [0, 1)")
    if not args.train_file.is_file():
        parser.error(f"training file not found: {args.train_file}")
    if args.val_file and not args.val_file.is_file():
        parser.error(f"validation file not found: {args.val_file}")
    try:
        train_records = load_and_check(args.train_file)
        val_records = load_and_check(args.val_file) if args.val_file else []
    except (OSError, ValueError) as exc:
        print(f"dataset preflight failed: {exc}", file=sys.stderr)
        return 2
    if not train_records:
        print("dataset preflight failed: training file contains no examples", file=sys.stderr)
        return 2
    print(json.dumps({"train_examples": len(train_records), "validation_examples": len(val_records),
                      "model": args.model, "sequence_length": args.max_seq_length,
                      "output_dir": str(args.output_dir)}, ensure_ascii=False))
    if args.preflight_only:
        return 0

    try:
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        print(f"missing training dependency: {exc}; install Unsloth in a supported GPU environment", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA GPU is unavailable. QLoRA training requires a supported GPU; preflight completed without training.", file=sys.stderr)
        return 2

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    model.print_trainable_parameters()

    train_rows = [make_prompt_completion(row, tokenizer) for row in train_records]
    train_ds = Dataset.from_list(train_rows)
    eval_ds = None
    if val_records:
        eval_ds = Dataset.from_list([make_prompt_completion(row, tokenizer) for row in val_records])
    elif args.eval_ratio > 0 and len(train_ds) >= 3:
        n_eval = max(1, round(len(train_ds) * args.eval_ratio))
        n_eval = min(n_eval, len(train_ds) - 1)
        split = train_ds.train_test_split(test_size=n_eval, seed=3407, shuffle=True)
        train_ds, eval_ds = split["train"], split["test"]

    use_bf16 = bool(torch.cuda.is_bf16_supported())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_seq_length,
        packing=False,
        completion_only_loss=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=5,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    result = trainer.train()
    print(result)
    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved LoRA adapter and tokenizer to {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
