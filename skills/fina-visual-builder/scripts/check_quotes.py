#!/usr/bin/env python3
"""Verify the block book's quoted source excerpts against the real files.

Line-number citations rot. The first audit of this book found 39 of 122 code
citations pointing at the wrong line, including four that were not off by a line
or two but addressed to an entirely different function. Prose that quotes a
comment and cites a line number keeps reading correctly long after the line
number stops being true, so nothing catches it by eye.

This checks the strongest form of citation the book uses: a fenced code block
whose first line is a header naming the file and the line range, followed by a
verbatim excerpt of the source. For each of those, assert the excerpt appears in
the file inside the cited range. That is the subset of citations a reader will
actually go and compare, and it is the subset that must not be wrong.

    python3 scripts/check_quotes.py

Exit status 0 if every quoted excerpt is present at the line it claims.
"""
import io
import os
import re
import sys

# Located relative to this file, so a clone anywhere works.
SKILL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(SKILL, "refs", "block-book")

# Walk up to the repository root (the directory holding `modules/`).
REPO = SKILL
for _ in range(6):
    if os.path.isdir(os.path.join(REPO, "modules")):
        break
    REPO = os.path.dirname(REPO)

SEARCH_ROOTS = tuple(os.path.join(REPO, "modules", m)
                     for m in ("fina-risk", "fina-core", "fina-skills"))
SKIP_DIRS = {".venv", "node_modules", ".git", "build", "cmake-build", "refs",
             ".mypy_cache", ".pytest_cache", "__pycache__", "site-packages"}

# A fenced block whose opening comment is `// path/to/file.ext:12-34` or
# `# path/to/file.ext:12` or a bare `// file.ext:12` (resolved by basename).
HEADER = re.compile(r"^\s*(?://|#)\s*(?:\*?\s*)?"
                    r"(?P<path>[A-Za-z0-9_./-]+\.(?:py|cpp|hpp))"
                    r":(?P<lo>\d+)(?:-(?P<hi>\d+))?\s*(?:\*|/)?\s*$")

# Excerpts in the book are rewrapped and lightly elided, so compare on
# whitespace-normalised, non-empty lines and require every quoted line to match
# one in the cited range, in order.
def norm(line):
    return re.sub(r"\s+", " ", line).strip()


def resolve(name):
    base = os.path.basename(name)
    for root in SEARCH_ROOTS:
        for dirpath, dirnames, names in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if base in names and "fina_risk" in os.path.relpath(dirpath, root):
                return os.path.join(dirpath, base)
    for root in SEARCH_ROOTS:
        for dirpath, dirnames, names in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            if base in names:
                return os.path.join(dirpath, base)
    return None


def check(run, hays, full, lo, hi, rel, header_line, failures):
    """One contiguous run of quoted lines must appear verbatim in the range."""
    joined = " ".join(run)
    if any(joined in h for h in hays):
        return
    # A book excerpt may quote a subset of its run (a comment replaced, a line
    # dropped for length). Accept it if every line but one is present in order.
    for drop in range(len(run)):
        trimmed = run[:drop] + run[drop + 1:]
        if not trimmed:
            continue
        t = " ".join(trimmed)
        if any(t in h for h in hays):
            return
    failures.append("%s:%d  %s quoted block not at %d-%d: %s"
                    % (rel, header_line, os.path.basename(full), lo, hi,
                       " / ".join(run)[:90]))


def main():
    cache = {}
    checked = 0
    failures = []

    for dirpath, _, names in os.walk(BOOK):
        for name in sorted(names):
            if not name.endswith(".md"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, BOOK)
            lines = io.open(path, encoding="utf-8").read().split("\n")

            in_block = False
            block_lang = None
            header = None
            excerpt = []
            header_line = 0
            examined = False
            skip = False

            for i, line in enumerate(lines, 1):
                if line.startswith("```"):
                    if not in_block:
                        in_block = True
                        block_lang = line[3:].strip()
                        header = None
                        excerpt = []
                        header_line = 0
                        examined = False
                        skip = False
                    else:
                        # end of block: run the check
                        if header is not None and excerpt:
                            full = resolve(header.group("path"))
                            if full is None:
                                failures.append(
                                    "%s:%d  cannot find %s"
                                    % (rel, header_line, header.group("path")))
                            else:
                                if full not in cache:
                                    cache[full] = io.open(
                                        full, encoding="utf-8",
                                        errors="replace").read().split("\n")
                                src = cache[full]
                                lo = int(header.group("lo"))
                                hi = int(header.group("hi") or lo)
                                # One whitespace-normalised haystack for the whole
                                # cited range. The book rewraps long lines, so a
                                # quote is a run of consecutive excerpt lines that
                                # must appear contiguously in the haystack. A
                                # second haystack drops the source's own comment
                                # lines, so a quote that omits them still matches.
                                window = src[lo - 1:hi]
                                hay = (" ".join(norm(s) for s in window),
                                       " ".join(norm(s) for s in window
                                                if not s.strip().startswith(
                                                    ("//", "#", "*", "/*"))))
                                checked += 1
                                run = []
                                for q in excerpt + ["<elision>"]:
                                    if q == "<elision>":
                                        if run:
                                            check(run, hay, full, lo, hi, rel,
                                                  header_line, failures)
                                        run = []
                                        continue
                                    run.append(q)
                                if run:
                                    check(run, hay, full, lo, hi, rel,
                                          header_line, failures)
                        in_block = False
                    continue

                if not in_block or skip:
                    continue
                stripped = line.strip()
                if not stripped:
                    continue
                if not examined:
                    examined = True
                    m = None
                    if block_lang in ("", "cpp", "c++", "python", "py"):
                        m = HEADER.match(line)
                    if m:
                        header = m
                        header_line = i
                    else:
                        skip = True
                    continue
                if stripped.startswith(("//", "#", "/*", "*", "*/")):
                    # an inline comment inside the excerpt is not a quoted source line
                    continue
                if stripped.startswith("...") or stripped.startswith("…"):
                    excerpt.append("<elision>")
                    continue
                excerpt.append(norm(line))

    print("quoted excerpts checked: %d" % checked)
    if failures:
        print("FAIL: %d quoted line(s) are not at the line the book claims\n" % len(failures))
        for f in failures:
            print("  " + f)
        return 1
    print("all quoted excerpts are present at the cited lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
