# Riga Fin Model — Unsloth + Hugging Face fine-tuning

A reproducible starter project for turning domain documents into an evidence-grounded instruction dataset, validating it, and fine-tuning a Hugging Face causal language model with Unsloth QLoRA.

This scaffold follows the attached POC guide: Python 3.10–3.12, 4-bit QLoRA, small batches, gradient accumulation, Unsloth gradient checkpointing, ChatML-style examples, and assistant/completion-only loss. The default settings are deliberately conservative for an 8 GB GPU. A 7B model can be tight on 8 GB; start with a 1B–3B model or lower sequence length/rank if memory is limited.

## Project structure

- `skills/finetune/SKILL.md` — agent workflow to validate, configure, run, evaluate, and save a fine-tune.
- `.skills/distill-dataset/SKILL.md` — agent workflow to extract documents, generate grounded examples with a larger teacher model, and review the result.
- `scripts/generate_seed_dataset.py` — emit a curated, cited starter SFT dataset grounded in the reference corpus (100 train + 12 held-out validation examples).
- `scripts/prepare_documents.py` — extract text from Markdown, TXT, PDF, or DOCX and write traceable chunks.
- `scripts/distill_dataset.py` — call an OpenAI-compatible teacher endpoint and save ChatML JSONL with provenance.
- `scripts/validate_dataset.py` — check supported JSONL formats, role sequences, missing values, and duplicates.
- `scripts/train_unsloth.py` — format prompt/completion pairs with the selected tokenizer’s chat template and train a LoRA adapter using Unsloth + TRL.
- `tests/` — dependency-light pipeline smoke tests; no model download or external API call is needed.

The draft guide itself is not copied into the repository. Add your domain documents under `data/raw/` (or another private folder) and keep source material, generated datasets, checkpoints, credentials, and model artifacts out of Git.

## 1. Environment

Use Linux or WSL2, Python 3.10–3.12, a working NVIDIA driver/CUDA stack for training, and enough disk for model weights and checkpoints. For an RTX 3060 Ti (8 GB), start with batch size 1, sequence length 512–1024, LoRA rank 8–16, and gradient accumulation 8. 7B QLoRA is possible but may OOM depending on context length, model implementation, and other GPU processes.

This repository uses [`uv`](https://docs.astral.sh/uv/) to manage the `.venv`. Create the
environment with a supported interpreter and install into it with `uv pip`:

```bash
# from the repository root; pins a 3.12 interpreter regardless of the system python
uv venv --python 3.12
source .venv/bin/activate
# Unsloth selects/patches the compatible PyTorch/Transformers/TRL stack for the platform:
uv pip install --python .venv/bin/python unsloth
# Needed only for document distillation and its CLI:
uv pip install --python .venv/bin/python -r requirements-distill.txt
```

`uv venv` without `--python` picks up whatever interpreter `uv` finds first, which may be
outside the supported 3.10–3.12 window (the system python here is 3.14), so always pass
`--python 3.12`. Add new dependencies with
`uv pip install --python .venv/bin/python <package>` rather than bare `pip`, so they land
in `.venv`.

Check the training device before downloading a base model:

```bash
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU")'
nvidia-smi
```

For gated Hugging Face models, authenticate with `hf auth login` or set `HF_TOKEN` in your local environment. Do not commit tokens. The suggested default is `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`; change `--model` to another compatible instruct model. The attached guide also discusses a Qwen2.5-Coder 7B-class run; use it only after a 3B smoke run fits.

## 2. Prepare source chunks

Supported input: `.txt`, `.md`, `.pdf`, `.docx`. PDF extraction is text-based; scanned PDFs need OCR before this step. Chunk metadata records source file, chunk number, and page where available.

```bash
mkdir -p data/raw data/interim data/processed
python scripts/prepare_documents.py \
  --input data/raw \
  --output data/interim/source_chunks.jsonl \
  --chunk-words 1000 \
  --overlap-words 120
```

Inspect sample chunks and confirm page/section boundaries before sending them to a teacher model. Documents can contain proprietary, confidential, personal, licensed, or regulated information. Verify you are authorized to use both the documents and the chosen external teacher endpoint; redact sensitive material or use an approved self-hosted teacher when needed.

## 3. Distill a synthetic dataset

The distiller speaks to an OpenAI-compatible chat-completions API. For Hugging Face Inference Providers, set the Router base URL and token; for a self-hosted vLLM/TGI-compatible endpoint, use its URL and credential. The exact model identifier is provider-specific.

```bash
export OPENAI_BASE_URL="https://router.huggingface.co/v1"
export HF_TOKEN="..."  # keep local; never put this in a committed .env
export OPENAI_API_KEY="$HF_TOKEN"

python scripts/distill_dataset.py \
  --chunks data/interim/source_chunks.jsonl \
  --output data/processed/synthetic.jsonl \
  --model "YOUR_TEACHER_MODEL_ID" \
  --examples-per-chunk 3 \
  --system-prompt-file examples/system_prompt.txt
```

The first run refuses to overwrite an existing output. Use `--resume` to continue an interrupted run (processed chunk IDs are skipped), or `--overwrite` to intentionally replace output. Use `--limit-chunks 2` for a small, cost-controlled API check. The generator is not run by the local test suite.

Each JSONL line contains `messages` (system/user/assistant) plus `source`, `location`, `chunk_id`, and teacher provenance. The assistant answer is instructed to remain within the supplied excerpt and include a short source citation. Review citations and factual accuracy against the source; teacher output is synthetic and is not automatically trustworthy.

## 4. Validate and inspect

```bash
python scripts/validate_dataset.py data/processed/synthetic.jsonl
```

Check duplicate and invalid-row warnings, topic/task diversity, answer quality, citation correctness, and train/validation leakage. Keep a human-reviewed held-out set (or use the trainer’s deterministic 5% split for an initial smoke run). Do not use a validation set as training data.

Accepted formats:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"source":"manual.pdf","location":"page 3"}
```

or Alpaca-style `{"instruction":"...","input":"...","output":"..."}`. One example per line. Conversations must end in an assistant turn.

## 5. Train a QLoRA adapter

```bash
python scripts/train_unsloth.py \
  --train-file data/processed/synthetic.jsonl \
  --model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \
  --output-dir outputs/qwen-coder-3b-lora \
  --max-seq-length 1024 \
  --rank 16 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-steps 60
```

The trainer applies the model tokenizer’s chat template, treats all prior turns as the prompt, and calculates loss on the final assistant completion. It uses a deterministic 5% validation split when there are enough rows; pass `--val-file` for a separately curated validation JSONL or set `--eval-ratio 0` to disable the automatic split. Adapter and tokenizer files are saved under `--output-dir/adapter`; checkpoints go under `--output-dir`.

To test without importing ML dependencies or downloading a model:

```bash
python scripts/train_unsloth.py --train-file data/processed/synthetic.jsonl --preflight-only
```

## 6. Test this repository

```bash
python -m unittest discover -s tests -v
```

The smoke tests exercise document extraction/chunking, JSONL validation, and the trainer’s preflight path. They do not call a teacher API or load/train a model.

## Important notes

- Fine-tuning teaches response style/task behavior; it is not a substitute for retrieval when facts change or exact source traceability is required. For frequently updated or large knowledge bases, consider RAG alongside a small adapter.
- Start with high-quality, deduplicated examples. Do not blindly trust teacher-generated answers or optimize only for training loss.
- Use a license-permitted base model and confirm your synthetic dataset’s source-material rights.
- Unsloth/TRL/Hugging Face APIs evolve. The training script uses the current TRL `SFTConfig` / `processing_class` pattern and Unsloth’s `FastLanguageModel`; pin a tested environment before a long run.

## References

- [Unsloth fine-tuning guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/en/sft_trainer)
- [Hugging Face Hub authentication](https://huggingface.co/docs/huggingface_hub/guides/cli)
