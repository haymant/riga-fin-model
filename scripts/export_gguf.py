#!/usr/bin/env python3
"""Merge a LoRA adapter into its base model and export a quantized GGUF file.

Why this exists
---------------
The base checkpoints Unsloth publishes as ``*-bnb-4bit`` are stored *already
4-bit quantized*. Merging an adapter into one requires dequantizing to bf16,
which for a 3B model needs ~6 GB of VRAM. On a small GPU (e.g. an 8 GB card
shared with a desktop session) the GPU merge OOMs, so this script does the merge
on the **CPU** against the full-precision base model, then converts and
quantizes with llama.cpp. It needs RAM, not VRAM.

Pipeline
--------
1. Load the full-precision base (``--base-model``) on CPU in bf16.
2. Attach the LoRA adapter, merge it, and save a merged HF model.
3. ``llama.cpp/convert_hf_to_gguf.py`` -> f16 GGUF.
4. ``llama.cpp/build/bin/llama-quantize`` -> the requested quant.

Prerequisites
-------------
- ``pip install gguf`` (plus transformers/peft/torch, already in the training env)
- A llama.cpp checkout with the ``llama-quantize`` binary built, and its
  ``convert_hf_to_gguf.py`` script.

Example
-------
    python scripts/export_gguf.py \
        --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
        --adapter outputs/qwen-coder-3b-lora/adapter \
        --output-dir outputs/qwen-coder-3b-gguf \
        --llama-cpp /tmp/llama.cpp \
        --quant q4_k_m
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def merge_adapter(base_model: str, adapter: str, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading base {base_model} on CPU (bf16) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    print(f"attaching adapter {adapter} ...", flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    print("merging adapter into base ...", flush=True)
    model = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving merged model to {merged_dir} ...", flush=True)
    model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
    AutoTokenizer.from_pretrained(base_model).save_pretrained(merged_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--llama-cpp", required=True, type=Path, help="llama.cpp checkout")
    parser.add_argument("--quant", default="q4_k_m", help="llama-quantize type, e.g. q4_k_m, q8_0, f16")
    parser.add_argument("--merged-dir", type=Path, help="reuse an existing merged model instead of merging")
    parser.add_argument("--skip-merge", action="store_true", help="reuse --merged-dir as-is")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_dir: Path = args.merged_dir or (output_dir / "merged-hf")

    if not args.skip_merge:
        if not args.adapter.is_dir():
            parser.error(f"adapter directory not found: {args.adapter}")
        merge_adapter(args.base_model, str(args.adapter), merged_dir)

    convert_script = args.llama_cpp / "convert_hf_to_gguf.py"
    quantize_bin = args.llama_cpp / "build" / "bin" / "llama-quantize"
    if not convert_script.is_file():
        parser.error(f"convert script not found: {convert_script}")
    if not quantize_bin.is_file():
        parser.error(f"llama-quantize not found: {quantize_bin} (build it first)")

    f16_path = output_dir / "model-f16.gguf"
    run([sys.executable, str(convert_script), str(merged_dir),
         "--outfile", str(f16_path), "--outtype", "f16"])

    if args.quant == "f16":
        print(f"GGUF written: {f16_path}")
        return 0

    quant_path = output_dir / f"model-{args.quant}.gguf"
    run([str(quantize_bin), str(f16_path), str(quant_path), args.quant])
    print(f"GGUF written: {quant_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
