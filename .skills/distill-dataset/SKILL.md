---
name: distill-dataset
description: Distill authorized domain documents into source-grounded instruction/chat JSONL using a larger teacher model for downstream supervised fine-tuning.
---

# Distill a domain dataset from documents

Use this skill to generate candidate SFT examples from supplied knowledge documents. The model-generated output is a draft dataset, not verified ground truth. A human or an independent reviewer must inspect it before fine-tuning.

## Inputs and safeguards

- Source documents and target domain/task/style.
- A teacher model reachable through an OpenAI-compatible Chat Completions endpoint.
- API credentials supplied via environment variables (never commit, print, or place them in prompts/files).
- Permission to process the documents with the chosen teacher service. If documents are confidential, personal, regulated, licensed, or otherwise restricted, use an approved self-hosted teacher or obtain authorization before transmission. Do not assume that a public document is reusable for training.
- User-defined system instruction when the model's target behavior differs from the project default.

Never invent domain-specific facts when the source guide is unavailable. Ask for missing material if it changes the intended examples materially. Exclude secrets and unnecessary personal information. Preserve the original source and generated outputs separately.

## Workflow

1. **Inventory and inspect** source files; identify file types, scan/OCR needs, language, duplicate editions, scope, and usage rights. Exclude irrelevant or unapproved files.
2. **Extract and chunk** text with stable provenance:

```bash
python scripts/prepare_documents.py \
  --input data/raw \
  --output data/interim/source_chunks.jsonl \
  --chunk-words 1000 --overlap-words 120
```

   Supported: `.txt`, `.md`, text-readable `.pdf`, and `.docx`. Scanned PDFs require OCR before extraction. Inspect representative chunks, page labels, tables, formulas, and code to ensure extraction did not corrupt them.
3. **Choose the teacher endpoint and prompt.** A router/provider model ID must be verified against the chosen service. For Hugging Face Router, set `OPENAI_BASE_URL=https://router.huggingface.co/v1`; set `HF_TOKEN` and `OPENAI_API_KEY` locally. For a private OpenAI-compatible endpoint, set its base URL and API key. Avoid sending documents to an unapproved third party.
4. **Generate a small sample first**, inspect the returned questions/answers and provider costs, and only then scale. `--limit-chunks 2` keeps an initial check bounded:

```bash
python scripts/distill_dataset.py \
  --chunks data/interim/source_chunks.jsonl \
  --output data/processed/synthetic.jsonl \
  --model YOUR_TEACHER_MODEL_ID \
  --examples-per-chunk 3 \
  --system-prompt-file examples/system_prompt.txt \
  --limit-chunks 2
```

   The CLI does not call the teacher unless explicitly executed. It defaults to fail-safe output handling: existing output is not replaced; `--resume` skips previously completed chunk IDs; `--overwrite` replaces intentionally.
5. **Review and curate.** Check every answer against its cited source/location, exact numbers/units/formulas, ambiguity, answerability, task diversity, duplicates, leakage, style, and safe abstention. Remove examples that merely copy passages without useful task transformation. When material, use independent verification rather than teacher self-critique alone.
6. **Validate** candidate JSONL before passing it to the fine-tuning skill:

```bash
python scripts/validate_dataset.py data/processed/synthetic.jsonl
```

   Move approved records into explicit train/validation files only after deduplication and split by document/source when possible, so near-identical chunks do not leak across the split. A model-generated citation does not prove the answer is correct.

## Generation contract

Each training record is ChatML-style JSONL:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"source":"manual.pdf","location":"page 4","chunk_id":"...","teacher_model":"..."}
```

The teacher is instructed to create diverse, answerable questions using only the provided excerpt, keep unsupported details out, and return a JSON object with an `examples` array. Its output is parsed and normalized by `scripts/distill_dataset.py`; malformed or empty generations are rejected. The assistant answer carries a concise source citation and provenance is kept in top-level metadata. A stable source identifier is preferred over leaking sensitive local paths.

## Quality and stop criteria

- Stop and ask for clarification if the domain, target behavior, source authorization, or teacher-data policy is unclear and would change the dataset materially.
- Stop the batch if source extraction is corrupted, the teacher repeatedly adds unsupported claims, provenance cannot be established, or the service reports errors/rate limits.
- Keep a small, reviewed validation set untouched during training. Track source-document IDs and hashes to make the generation reproducible.
- Do not over-represent a single chapter or prompt type. Prefer a smaller clean set over many low-quality examples.
