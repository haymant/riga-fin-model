#!/usr/bin/env python3
"""Generate candidate grounded SFT examples from prepared source chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SYSTEM = (
    "You are a careful financial modeling and analysis assistant. Answer only from the supplied source excerpt. "
    "Do not invent facts, numbers, formulas, units, dates, or assumptions. If the excerpt does not support an answer, "
    "do not create that question. Give a concise, useful answer and include a short source citation using the provided "
    "document and location. This is educational/modeling assistance, not personalized investment advice."
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected an object")
            rows.append(row)
    return rows


def extract_examples(content: str) -> list[dict]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("teacher response did not contain a JSON object")
        data = json.loads(text[start : end + 1])
    if isinstance(data, list):
        examples = data
    elif isinstance(data, dict):
        examples = data.get("examples")
    else:
        examples = None
    if not isinstance(examples, list):
        raise ValueError("teacher JSON must contain an 'examples' array")
    return [item for item in examples if isinstance(item, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", required=True, help="Teacher model ID accepted by the configured endpoint")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--examples-per-chunk", type=int, default=3)
    parser.add_argument("--limit-chunks", type=int, help="Process only the first N unprocessed chunks")
    parser.add_argument("--max-context-chars", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true", help="Append output and skip chunk IDs already present")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    if args.examples_per_chunk < 1 or args.max_context_chars < 100:
        parser.error("examples-per-chunk must be positive and max-context-chars at least 100")
    if args.resume and args.overwrite:
        parser.error("choose either --resume or --overwrite")
    if args.output.exists() and not (args.resume or args.overwrite):
        parser.error(f"output already exists: {args.output}; use --resume or --overwrite")
    if not args.chunks.is_file():
        parser.error(f"chunks file not found: {args.chunks}")

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")
    if not api_key:
        parser.error("set OPENAI_API_KEY (or HF_TOKEN) in the environment; never pass secrets on the command line")
    try:
        from openai import OpenAI
    except ImportError:
        parser.error("install the distillation dependencies: pip install -r requirements-distill.txt")

    try:
        chunks = load_jsonl(args.chunks)
        existing = load_jsonl(args.output) if args.resume and args.output.exists() else []
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    done = {row.get("chunk_id") for row in existing if row.get("chunk_id")}
    pending = [r for r in chunks if r.get("chunk_id") not in done]
    if args.limit_chunks is not None:
        if args.limit_chunks < 1:
            parser.error("limit-chunks must be positive")
        pending = pending[: args.limit_chunks]
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip() if args.system_prompt_file else DEFAULT_SYSTEM
    if not system_prompt:
        parser.error("system prompt must not be empty")
    client = OpenAI(api_key=api_key, base_url=base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    written = 0
    with args.output.open(mode, encoding="utf-8") as handle:
        for chunk in pending:
            chunk_id = chunk.get("chunk_id")
            source = str(chunk.get("source", "unknown source"))
            location = str(chunk.get("location", "unspecified location"))
            excerpt = str(chunk.get("text", "")).strip()
            if not excerpt or len(excerpt) > args.max_context_chars:
                if len(excerpt) > args.max_context_chars:
                    print(f"skip {chunk_id}: chunk exceeds max-context-chars; rechunk the document instead of silently truncating", file=sys.stderr)
                continue
            instruction = (
                f"Create up to {args.examples_per_chunk} diverse, answerable instruction/response examples from the excerpt below. "
                "Use only facts supported by the excerpt. Preserve exact numbers, formulas, units, and qualifications. "
                "Do not ask questions that require outside knowledge. Each response must end with a brief citation "
                f"like 'Source: {source} ({location})'. Return JSON only in this schema: "
                '{"examples":[{"instruction":"...","response":"...","task_type":"..."}]}.\n\n'
                f"Document: {source}\nLocation: {location}\n\nSOURCE EXCERPT:\n{excerpt}"
            )
            try:
                result = client.chat.completions.create(
                    model=args.model,
                    temperature=args.temperature,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": instruction}],
                )
                content = result.choices[0].message.content or ""
                generated = extract_examples(content)
            except Exception as exc:  # provider SDK errors differ across compatible endpoints
                print(f"teacher request failed for chunk {chunk_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            chunk_written = 0
            for item in generated:
                question = str(item.get("instruction", item.get("question", ""))).strip()
                answer = str(item.get("response", item.get("answer", ""))).strip()
                if len(question) < 8 or len(answer) < 20:
                    continue
                citation = f"Source: {source} ({location})"
                if citation.lower() not in answer.lower():
                    answer = f"{answer.rstrip()}\n\n{citation}"
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                    "source": source,
                    "location": location,
                    "chunk_id": chunk_id or hashlib.sha256(excerpt.encode()).hexdigest()[:20],
                    "teacher_model": args.model,
                    "task_type": str(item.get("task_type", "domain_qa")),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                chunk_written += 1
                written += 1
            if chunk_written == 0:
                print(f"warning: no usable examples returned for chunk {chunk_id}", file=sys.stderr)
    print(json.dumps({"chunks_considered": len(pending), "examples_written": written, "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
