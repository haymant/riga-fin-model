---
name: finetune
description: Run supervised fine-tuning of a Hugging Face causal language model using Unsloth QLoRA and validate the resulting adapter.
---

# Fine-tune with Unsloth + Hugging Face

Use this skill when asked to train or tune a model from an existing instruction/chat dataset. This project targets text-only SFT with ChatML/conversational or Alpaca-style JSONL. Do not start a long training run until the dataset and GPU are checked.

## Required inputs

- A base instruct model ID or local model path compatible with Unsloth.
- A training JSONL file; optionally a separate curated validation JSONL.
- Output directory and target hardware constraints.
- Confirmation that the dataset and model licenses permit the planned use.

If an important input is missing, use the repository defaults only for a reversible smoke run; do not invent domain facts or training examples.

## Workflow

1. **Inspect and preflight**
   - Confirm Python 3.10–3.12, available disk, GPU/driver with `nvidia-smi`, and a clean virtual environment.
   - Install `unsloth` using the platform's current official installation guidance. Install `requirements-distill.txt` only if the document-generation utilities are also needed.
   - Run `python scripts/validate_dataset.py PATH.jsonl` and address malformed rows, empty turns, duplicates, leakage, and ungrounded responses.
   - Run `python scripts/train_unsloth.py --train-file PATH.jsonl --preflight-only` before model download.
   - For 8 GB VRAM, start with a 1B–3B model, `--batch-size 1`, `--max-seq-length 512` or `1024`, `--rank 8` or `16`, and gradient accumulation 8. A 7B/8B model is a tight configuration; reduce sequence length/rank if OOM occurs.

2. **Select and record configuration**
   - Use an instruct model whose tokenizer has a valid chat template. The attached guide suggests Qwen2.5-Coder; this project's CLI defaults to a 3B 4-bit Qwen-family checkpoint. Specify a gated model only after Hugging Face authentication is available.
   - Start conservatively: QLoRA 4-bit, LoRA `r=16`, alpha equal to rank, batch size 1, gradient accumulation 8, learning rate `2e-4`, and 60 steps for a smoke experiment. Prefer the 3B model before trying the 7B-class model on an 8 GB card.
   - Use a separate validation file if available. Otherwise, a deterministic small validation split is created when the dataset is large enough; disable with `--eval-ratio 0` only when appropriate.
   - Do not push model weights to a Hub or publish artifacts unless the user explicitly requests it.

3. **Train**

```bash
python scripts/train_unsloth.py \
  --train-file data/processed/train.jsonl \
  --model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
  --output-dir outputs/qwen-coder-3b-lora \
  --max-seq-length 1024 --rank 16 --batch-size 1 \
  --gradient-accumulation-steps 8 --max-steps 60
```

   - Add `--val-file data/processed/valid.jsonl` for an independent held-out set.
   - The script passes prompt/completion examples to TRL `SFTTrainer`; it renders all messages before the final assistant turn with the tokenizer's chat template and masks prompt tokens using completion-only loss.
   - Save only the adapter and tokenizer in `<output-dir>/adapter`; periodic training checkpoints are in `<output-dir>`.

4. **Evaluate and report**
   - Record final train/eval loss and check for unstable loss, OOM, truncation, and overfitting. Loss is a diagnostic, not proof of domain correctness.
   - Manually compare the base model and adapter on held-out, source-grounded prompts that cover the intended domain, edge cases, abstention when evidence is missing, and response formatting. Use `python scripts/chat_adapter.py --adapter outputs/qwen-coder-3b-lora/adapter --prompt "..."` for a quick GPU-backed adapter check.
   - Inspect several outputs against the original documents. Do not claim a successful domain adaptation based only on a completed training loop.
   - Report exact model revision/ID, dataset hash and counts, seed, hyperparameters, GPU, software versions, output adapter path, and limitations. Preserve the validation set separately.

## Dataset requirements

Every JSONL line should have `messages` with `role`/`content` entries ending in an assistant turn, or Alpaca fields `instruction`, optional `input`, and `output`. Generated records may include provenance fields (`source`, `location`, `chunk_id`, `teacher_model`); the trainer uses those only as metadata and does not include them in model input. Ensure the final assistant answer is accurate and properly scoped.

## Recovery

- **CUDA OOM:** close GPU-heavy processes; reduce model size, sequence length, or LoRA rank; keep batch size 1; check free VRAM with `nvidia-smi`.
- **Chat-template error:** choose a compatible instruct checkpoint or inspect/fix the tokenizer template; do not silently fall back to an unrelated formatting scheme.
- **TRL/Unsloth mismatch:** create a fresh venv, install Unsloth per its current matrix, record package versions, then rerun preflight. Avoid arbitrary dependency downgrades in an existing environment.
- **No CUDA:** do not attempt QLoRA training as if it were validated. Run the data/preflight tests and move the actual training to a supported GPU host.
