#!/usr/bin/env python3
"""Check the block book: relative links resolve, anchors exist, no PII, tables
are well-formed, and every block kind in the registry is represented.

    python3 scripts/check_book.py
"""
import io
import os
import re
import sys

# Located relative to this file, so a clone anywhere works.
SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(SKILL, "refs")
BOOK = os.path.join(ROOT, "block-book")

errors = []


def md_files(base):
    for dirpath, _, names in os.walk(base):
        for n in names:
            if n.endswith(".md"):
                yield os.path.join(dirpath, n)


def slugify(heading):
    s = heading.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s+", "-", s)


def anchors(text):
    out = set()
    in_fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if m:
            out.add(slugify(m.group(1)))
    return out


files = sorted(md_files(BOOK)) + [os.path.join(ROOT, "ModelAndTradeLayer.md")]
texts = {f: io.open(f, encoding="utf-8").read() for f in files}

# --- links -----------------------------------------------------------------
for f, text in texts.items():
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://", "#")):
            if target.startswith("#"):
                if target[1:] and target[1:] not in anchors(text):
                    errors.append("%s: dead in-page anchor %s" % (f, target))
            continue
        path, _, frag = target.partition("#")
        if not path:
            continue
        full = os.path.normpath(os.path.join(os.path.dirname(f), path))
        if not os.path.exists(full):
            errors.append("%s: missing link target %s" % (f, target))
            continue
        if frag and full.endswith(".md"):
            if frag not in anchors(texts[full]):
                errors.append("%s: anchor #%s not in %s" % (f, frag, os.path.basename(full)))

# --- registry coverage -----------------------------------------------------
import yaml  # noqa: E402

reg = yaml.safe_load(io.open(os.path.join(BOOK, "registry", "blocks.yaml"),
                             encoding="utf-8"))
kinds = [b["kind"] for b in reg["blocks"]]
if len(kinds) != len(set(kinds)):
    errors.append("registry: duplicate kinds")
readme = texts[os.path.join(BOOK, "README.md")]
for k in kinds:
    if "`%s`" % k not in readme:
        errors.append("registry: kind %s missing from README index" % k)

# --- every book_page that is not null exists -------------------------------
for b in reg["blocks"]:
    page = b.get("book_page")
    if page:
        full = os.path.normpath(os.path.join(BOOK, page))
        if not os.path.exists(full):
            errors.append("registry: book_page %s does not exist" % page)

# --- PII -------------------------------------------------------------------
PII = [
    r"\b\d{3}-\d{2}-\d{4}\b",                     # SSN
    r"\(\s*\+?\d{3}[\s-]?\d{3}[\s-]?\d{4}\s*\)",  # phone, parenthesised
    r"\b[A-Z]{2}[A-Z0-9]{9}\d\b",                # ISIN
    r"[\w.+-]+@[\w-]+\.[a-z]{2,}",               # email
    r"(?<![\d.])\+?\d{3}[\s-]?\d{3}[\s-]?\d{4}(?![\d.])",  # phone, bare
]
for f, text in texts.items():
    for pat in PII:
        for m in re.finditer(pat, text):
            errors.append("%s: possible PII %r" % (os.path.basename(f), m.group(0)))

# --- table shape -----------------------------------------------------------
for f, text in texts.items():
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.startswith("|") or i + 1 >= len(lines):
            continue
        nxt = lines[i + 1]
        if not re.match(r"^\|[\s:|-]+\|$", nxt):
            continue
        header = line.count("|")
        sep = nxt.count("|")
        if header != sep:
            errors.append("%s:%d table header has %d pipes, separator %d"
                          % (os.path.basename(f), i + 1, header, sep))
        j = i + 2
        while j < len(lines) and lines[j].startswith("|"):
            if lines[j].count("|") != header:
                errors.append("%s:%d row has %d pipes, header %d"
                              % (os.path.basename(f), j + 1, lines[j].count("|"), header))
            j += 1

# --- report ----------------------------------------------------------------
if errors:
    for e in errors:
        print("FAIL  " + e)
    sys.exit(1)
print("block book OK: %d files, %d registry kinds, links/anchors/tables/PII clean"
      % (len(files), len(kinds)))
