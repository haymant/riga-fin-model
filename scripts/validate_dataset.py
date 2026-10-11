#!/usr/bin/env python3
"""Validate SFT JSONL structure and report duplicates and basic dataset statistics."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ALLOWED_ROLES = {"system", "user", "assistant"}


def read_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: record must be a JSON object")
            records.append(row)
    return records


def as_messages(record: dict) -> list[dict]:
    if isinstance(record.get("messages"), list):
        return record["messages"]
    instruction = str(record.get("instruction", "")).strip()
    answer = str(record.get("output", record.get("response", ""))).strip()
    if not instruction or not answer:
        return []
    user_content = instruction
    extra_input = str(record.get("input", "")).strip()
    if extra_input:
        user_content += "\n\nInput:\n" + extra_input
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": answer},
    ]


def validate_records(records: list[dict]) -> tuple[list[str], dict]:
    errors: list[str] = []
    hashes: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    total_chars = 0
    for index, record in enumerate(records, 1):
        messages = as_messages(record)
        if not messages:
            errors.append(f"line {index}: expected non-empty messages or instruction/output")
            continue
        normalized = []
        for turn_no, message in enumerate(messages, 1):
            if not isinstance(message, dict):
                errors.append(f"line {index}, turn {turn_no}: message must be an object")
                continue
            role = message.get("role")
            content = message.get("content")
            if role not in ALLOWED_ROLES:
                errors.append(f"line {index}, turn {turn_no}: invalid role {role!r}")
            else:
                roles[role] += 1
            if not isinstance(content, str) or not content.strip():
                errors.append(f"line {index}, turn {turn_no}: content must be a non-empty string")
                continue
            total_chars += len(content)
            normalized.append({"role": role, "content": content.strip()})
        if messages and isinstance(messages[-1], dict) and messages[-1].get("role") != "assistant":
            errors.append(f"line {index}: conversation must end with assistant")
        if not any(isinstance(m, dict) and m.get("role") == "user" for m in messages):
            errors.append(f"line {index}: conversation needs at least one user turn")
        if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages):
            errors.append(f"line {index}: conversation needs at least one assistant turn")
        hashes[hashlib.sha256(json.dumps(normalized, ensure_ascii=False, sort_keys=True).encode()).hexdigest()] += 1
    duplicates = sum(count - 1 for count in hashes.values() if count > 1)
    stats = {
        "examples": len(records),
        "message_roles": dict(roles),
        "average_characters_per_example": round(total_chars / len(records), 1) if records else 0,
        "duplicate_examples": duplicates,
    }
    return errors, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--min-examples", type=int, default=1)
    parser.add_argument("--fail-on-duplicates", action="store_true")
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(f"dataset file not found: {args.dataset}")
    try:
        records = read_records(args.dataset)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    errors, stats = validate_records(records)
    if stats["examples"] < args.min_examples:
        errors.append(f"dataset has {stats['examples']} examples; need at least {args.min_examples}")
    if stats["duplicate_examples"]:
        message = f"found {stats['duplicate_examples']} duplicate example(s)"
        if args.fail_on_duplicates:
            errors.append(message)
        else:
            print(f"warning: {message}", file=sys.stderr)
    print(json.dumps({**stats, "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
