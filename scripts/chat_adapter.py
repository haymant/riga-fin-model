#!/usr/bin/env python3
"""Run one prompt against a locally saved Unsloth LoRA adapter."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="Local adapter directory saved by train_unsloth.py")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", default="You are a careful, evidence-grounded assistant specialized in financial modeling.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()
    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError as exc:
        print(f"missing training dependency: {exc}; install Unsloth in the training environment", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("CUDA GPU is unavailable; this adapter chat script is configured for GPU inference.", file=sys.stderr)
        return 2
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    output = model.generate(input_ids=input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
    answer = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    print(answer.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
