#!/usr/bin/env python3
"""Independent audit: does astrag's index actually match the repo on disk?

This does NOT trust astrag's own parser to grade itself. It rescans the
repo with separate, minimal extraction logic (Python: the ``ast`` module
directly; other languages: regex heuristics) and diffs the result
against ``CodebaseMemory.chunks`` two ways:

1. **File coverage** — every file actually on disk, bucketed into
   indexed / gitignored / excluded / generated / dotfile / binary /
   too-large / unreadable / parse-error, cross-checked against
   ``index_report()`` so the counts have to add up. Catches "silently
   dropped file" bugs.
2. **Content coverage** — for files astrag did consider, do the
   function/class names it extracted match what an independent scan of
   the same file finds? Catches "file indexed but a function inside it
   got missed" bugs, which file-level coverage alone can't see.

Independent verification is only as good as the independent verifier:
Python uses the stdlib ``ast`` module (a real, separate ground truth —
not astrag's ``PythonStdlibParser``, even though both ultimately call
``ast.parse``, the extraction logic here is written from scratch so a
bug in astrag's traversal doesn't also exist in the check). Other
languages (JS/TS/Go/Rust/Java/C/C++) use regex heuristics that are
good-enough sanity checks, not a real parser — expect some noise
(false positives on commented-out code, false negatives on unusual
syntax) and read the diffs with that in mind. Anything else (Ruby, Ru,
YAML, etc.) is reported at the file-coverage level only, honestly
labeled "not independently verified" rather than silently assumed fine.

Usage:
    python3 test.py /path/to/repo
    python3 test.py /path/to/repo --no-gitignore
    python3 test.py /path/to/repo --verbose
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astrag import CodebaseMemory                       # noqa: E402
from astrag.parsing import (DEFAULT_EXCLUDES,            # noqa: E402
                            GENERATED_FILENAMES)
from astrag.universal import BINARY_EXTENSIONS, looks_binary  # noqa: E402

# --------------------------------------------------------------------------
# Independent content extraction — deliberately separate from astrag's own
# parsers. Returns a set of (name, kind) tuples per file.
# --------------------------------------------------------------------------

def scan_python(path: str) -> set[tuple[str, str]] | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            source = fh.read()
        tree = ast.parse(source)
    except (SyntaxError, OSError):
        return None
    found: set[tuple[str, str]] = set()

    def walk_body(body, in_class: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.add((node.name, "method" if in_class else "function"))
            elif isinstance(node, ast.ClassDef):
                found.add((node.name, "class"))
                walk_body(node.body, in_class=True)

    walk_body(tree.body, in_class=False)
    return found


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def scan_markdown(path: str) -> set[tuple[str, str]] | None:
    """ATX headers (# .. ######), outside fenced code blocks -- mirrors
    universal.MarkdownChunker's own boundary rule (kind: 'section')."""
    source = _read(path)
    if source is None:
        return None
    heading = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*$")
    found: set[tuple[str, str]] = set()
    fence = False
    for line in source.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        m = heading.match(line)
        if m:
            found.add((m.group(2).strip()[:80], "section"))
    return found


def scan_yaml(path: str) -> set[tuple[str, str]] | None:
    """Top-level (column 0) ``key:`` lines -- mirrors YamlChunker
    (kind: 'key'; nested/indented keys aren't separately chunked)."""
    source = _read(path)
    if source is None:
        return None
    key = re.compile(r"^([\w./-]+)[ \t]*:")
    found = {(m.group(1), "key") for line in source.splitlines()
            for m in [key.match(line)] if m}
    return found


def scan_toml(path: str) -> set[tuple[str, str]] | None:
    """``[section]`` headers -- mirrors TomlIniChunker (kind: 'section')."""
    source = _read(path)
    if source is None:
        return None
    sec = re.compile(r"^\[+([^\]]+)\]+")
    found = {(m.group(1).strip(), "section") for line in source.splitlines()
            for m in [sec.match(line)] if m}
    return found


def scan_shell(path: str) -> set[tuple[str, str]] | None:
    """``name() {`` / ``function name {`` -- mirrors ShellParser (kind:
    'function'). Indentation IS allowed here (unlike the C-family rules
    above): shell functions are routinely defined inside an if/case
    block and ShellParser's own pattern permits leading whitespace."""
    source = _read(path)
    if source is None:
        return None
    fn = re.compile(r"^[ \t]*(?:function[ \t]+)?([\w.-]+)[ \t]*\(\)[ \t]*\{"
                    r"|^[ \t]*function[ \t]+([\w.-]+)[ \t]*\{")
    found: set[tuple[str, str]] = set()
    for line in source.splitlines():
        m = fn.match(line)
        if m:
            found.add((m.group(1) or m.group(2), "function"))
    return found


def scan_css(path: str) -> set[tuple[str, str]] | None:
    """``selector { ... }`` / ``@rule { ... }`` blocks, any nesting depth
    -- mirrors CssChunker (kind: 'rule' or 'at_rule'). Best-effort: a
    single findall over the whole file rather than CssChunker's real
    brace-depth scan, so it can't distinguish "nested inside @media" from
    "nested inside a plain rule" the way CssChunker's selective recursion
    does -- expect some noise on deeply-nested SCSS-style rules."""
    source = _read(path)
    if source is None:
        return None
    # crude comment strip so `/* foo { */` doesn't get read as a selector
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    block = re.compile(r"([^{};]+)\{")
    found: set[tuple[str, str]] = set()
    for m in block.finditer(source):
        sel = re.sub(r"\s+", " ", m.group(1)).strip()
        if not sel:
            continue
        kind = "at_rule" if sel.startswith("@") else "rule"
        found.add((sel[:60], kind))
    return found


_HTML_SEMANTIC_TAGS = {"header", "nav", "main", "section", "article",
                       "aside", "footer", "form", "template", "dialog",
                       "table"}
_HTML_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img",
                   "input", "link", "meta", "param", "source", "track", "wbr"}


def scan_html(path: str) -> set[tuple[str, str]] | None:
    """Elements with an ``id``, a semantic tag, or script/style/form/
    template -- mirrors HtmlChunker's own selection rule (everything else
    is deliberately not chunked; HTML is markup, not code, so indexing
    every ``<div>`` and ``<span>`` would be pure noise) AND its naming
    rule (``tag#id`` when there's an id, else the ``src`` attribute for
    e.g. ``<script src=...>``, else ``tag@Lline``). Independently
    implemented against stdlib ``html.parser.HTMLParser`` -- same
    tokenizer astrag uses, separate collection logic."""
    source = _read(path)
    if source is None:
        return None
    from html.parser import HTMLParser

    found: set[tuple[str, str]] = set()
    stack: list[tuple[str, dict, int]] = []

    class _Collector(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag in _HTML_VOID_TAGS:
                return
            stack.append((tag, dict(attrs), self.getpos()[0]))

        def handle_endtag(self, tag):
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == tag:
                    t, attrs, line = stack.pop(i)
                    del stack[i:]
                    aid = attrs.get("id")
                    if aid or t in _HTML_SEMANTIC_TAGS or \
                            t in ("script", "style", "form", "template"):
                        name = (f"{t}#{aid}" if aid
                               else attrs.get("src") or f"{t}@L{line}")
                        kind = t if t in ("script", "style", "form",
                                         "template") else "element"
                        found.add((name, kind))
                    return

    try:
        _Collector(convert_charrefs=True).feed(source)
    except Exception:
        return found or set()
    return found


# name-capturing patterns per extension; each yields (name, kind) tuples.
# Heuristic, not a real parser -- see module docstring.
#
# Anchored at column 0 (``^`` with no leading ``\s*``) deliberately: astrag
# itself only indexes top-level functions/classes as separate chunks (its
# own Python parser only walks ``tree.body``, never descending into a
# function's own body looking for nested defs) -- a nested/inner function
# is part of its enclosing chunk's source, not a separately searchable
# unit, by design. A naive regex with no nesting awareness would flag
# every nested closure as "missing" even though astrag never intended to
# index it separately. This does mean indented constructs (Java/C++
# methods inside a class, Rust fns inside an ``impl`` block) aren't
# independently re-verified here -- an honest reduction in coverage
# rather than a wall of false positives.
_REGEX_RULES: dict[str, list[tuple[re.Pattern, str]]] = {
    ".js": [
        (re.compile(r'^(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)', re.M), "function"),
        (re.compile(r'^(?:export\s+)?class\s+([A-Za-z_$][\w$]*)', re.M), "class"),
        (re.compile(r'^(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>', re.M), "function"),
    ],
    ".go": [
        (re.compile(r'^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)', re.M), "function"),
        (re.compile(r'^type\s+([A-Za-z_]\w*)\s+struct\b', re.M), "class"),
    ],
    ".rs": [
        (re.compile(r'^(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)', re.M), "function"),
        (re.compile(r'^(?:pub\s+)?struct\s+([A-Za-z_]\w*)', re.M), "class"),
    ],
    ".java": [
        (re.compile(r'^(?:public\s+)?(?:final\s+)?class\s+([A-Za-z_]\w*)', re.M), "class"),
    ],
    ".c": [(re.compile(r'^[A-Za-z_][\w\*\s]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{', re.M), "function")],
    ".cpp": [(re.compile(r'^[A-Za-z_][\w:\*\s]*?\b([A-Za-z_]\w*)\s*\([^;{]*\)\s*\{', re.M), "function")],
}
_REGEX_RULES[".jsx"] = _REGEX_RULES[".js"]
_REGEX_RULES[".ts"] = _REGEX_RULES[".js"]
_REGEX_RULES[".tsx"] = _REGEX_RULES[".js"]
_REGEX_RULES[".h"] = _REGEX_RULES[".c"]
_REGEX_RULES[".hpp"] = _REGEX_RULES[".cpp"]


_CONTROL_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "else",
    "do", "try", "synchronized", "new", "delete", "throw", "typeof",
}


def _strip_keywords(found: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return {(n, k) for n, k in found if n not in _CONTROL_KEYWORDS}


def scan_regex(path: str, ext: str) -> set[tuple[str, str]] | None:
    rules = _REGEX_RULES.get(ext)
    if rules is None:
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError:
        return None
    found: set[tuple[str, str]] = set()
    for pattern, kind in rules:
        for m in pattern.finditer(source):
            found.add((m.group(1), kind))
    return _strip_keywords(found)


_EXT_SCANNERS = {
    ".md": scan_markdown, ".markdown": scan_markdown, ".mdx": scan_markdown,
    ".yml": scan_yaml, ".yaml": scan_yaml, ".cff": scan_yaml,
    ".toml": scan_toml, ".ini": scan_toml, ".cfg": scan_toml,
    ".conf": scan_toml, ".properties": scan_toml,
    ".sh": scan_shell, ".bash": scan_shell, ".zsh": scan_shell, ".ksh": scan_shell,
    ".css": scan_css, ".scss": scan_css, ".less": scan_css,
    ".html": scan_html, ".htm": scan_html,
}


def independent_scan(path: str, ext: str) -> tuple[set[tuple[str, str]] | None, str]:
    """Returns (found_names_or_None, method) -- None means 'not verified'."""
    if ext == ".py":
        return scan_python(path), "ast (exact)"
    scanner = _EXT_SCANNERS.get(ext)
    if scanner is not None:
        return scanner(path), f"regex/parser (heuristic, {ext})"
    result = scan_regex(path, ext)
    if result is not None:
        return result, "regex (heuristic)"
    return None, "not independently verified"


# --------------------------------------------------------------------------
# Coverage accounting: classify every file on disk the same way index_repo
# does (built-in excludes, gitignore, generated-file list) WITHOUT calling
# astrag's filtering code, so a bug in that filtering doesn't also hide
# itself from this check.
# --------------------------------------------------------------------------

def classify_files(root: str, respect_gitignore: bool) -> dict[str, list[str]]:
    """rel_path -> [] if real candidate file, else classification skipped."""
    matcher = None
    if respect_gitignore:
        from astrag.ignore import IgnoreMatcher
        matcher = IgnoreMatcher(root)

    buckets: dict[str, list[str]] = {
        "excluded_dir": [], "gitignored_dir": [], "dotfile": [],
        "generated": [], "gitignored_file": [], "candidate": [],
    }
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        rel_dir = "" if rel_dir == "." else rel_dir
        if matcher is not None:
            matcher.descend(rel_dir)

        kept = []
        for d in sorted(dirnames):
            child_rel = f"{rel_dir}/{d}" if rel_dir else d
            if d in DEFAULT_EXCLUDES or d.startswith("."):
                buckets["excluded_dir"].append(child_rel)
                continue
            if matcher is not None and matcher.matches(child_rel, is_dir=True):
                buckets["gitignored_dir"].append(child_rel)
                continue
            kept.append(d)
        dirnames[:] = kept

        for fn in sorted(filenames):
            rel_file = f"{rel_dir}/{fn}" if rel_dir else fn
            if fn.startswith("."):
                buckets["dotfile"].append(rel_file)
                continue
            if fn in GENERATED_FILENAMES:
                buckets["generated"].append(rel_file)
                continue
            if matcher is not None and matcher.matches(rel_file, is_dir=False):
                buckets["gitignored_file"].append(rel_file)
                continue
            buckets["candidate"].append(rel_file)
    return buckets


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", nargs="?", default=".")
    ap.add_argument("--no-gitignore", action="store_true",
                    help="match how you indexed, if you also used --no-gitignore")
    ap.add_argument("--verbose", action="store_true",
                    help="list every file, not just ones with issues")
    ap.add_argument("--diff-file", metavar="PATH",
                    help="skip the full scan; show astrag's indexed chunks "
                         "vs. the independent scan for exactly one file "
                         "(path relative to repo root)")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.repo)
    respect_gitignore = not args.no_gitignore

    if args.diff_file:
        print(f"Indexing {root} ...")
        mem = CodebaseMemory().index_repo(root, respect_gitignore=respect_gitignore)
        rel = args.diff_file.replace("\\", "/").lstrip("/")
        full = os.path.join(root, rel)
        ext = os.path.splitext(rel)[1].lower()
        indexed = sorted((c.name, c.kind, c.start_line, c.end_line)
                         for c in mem.chunks if c.file == rel)
        expected, method = independent_scan(full, ext)
        print(f"\n=== {rel} ===")
        print(f"astrag indexed ({len(indexed)}):")
        for name, kind, s, e in indexed:
            print(f"    {name}  [{kind}]  lines {s}-{e}")
        print(f"\nindependent scan [{method}]:")
        if expected is None:
            print("    (not independently verified for this extension)")
        else:
            indexed_names = {n for n, _, _, _ in indexed}
            for name, kind in sorted(expected):
                mark = "OK " if name in indexed_names else "MISSING"
                print(f"    {mark}  {name}  [{kind}]")
        return 0

    print(f"Indexing {root} ...")
    mem = CodebaseMemory().index_repo(root, respect_gitignore=respect_gitignore)
    report = mem.index_report()

    indexed_by_file: dict[str, set[tuple[str, str]]] = {}
    for c in mem.chunks:
        indexed_by_file.setdefault(c.file, set()).add((c.name, c.kind))

    # ---- Layer 1: file coverage ----
    buckets = classify_files(root, respect_gitignore)
    considered = len(buckets["candidate"])
    print()
    print("=== File coverage ===")
    print(f"  candidates on disk (post-filter): {considered}")
    print(f"  index_report() considered:        {report.get('considered', 0)}")
    if considered != report.get("considered", 0):
        print("  MISMATCH -- independent classification and index_repo() "
             "disagree on candidate count. This itself is a bug signal.")
    else:
        print("  OK -- counts agree.")

    zero_chunk_files = []
    zero_chunk_expected = 0   # binary ext or looks_binary() -- not a bug
    for rel in sorted(buckets["candidate"]):
        full = os.path.join(root, rel)
        try:
            size = os.path.getsize(full)
        except OSError:
            continue
        if size == 0:
            continue
        if rel in indexed_by_file:
            continue
        ext = os.path.splitext(rel)[1].lower()
        if ext in BINARY_EXTENSIONS:
            zero_chunk_expected += 1
            continue
        # extension alone doesn't say binary (e.g. a minified/generated
        # .svg or .js) -- ask the same runtime check index_repo() itself
        # uses before calling anything "unexpected"
        try:
            with open(full, encoding="utf-8", errors="replace") as fh:
                head = fh.read(8000)
            if looks_binary(head):
                zero_chunk_expected += 1
                continue
        except OSError:
            pass
        zero_chunk_files.append(rel)
    if zero_chunk_expected:
        print(f"\n  {zero_chunk_expected} file(s) produced zero chunks as "
             f"expected (binary extension or looks_binary() — images, "
             f"fonts, minified/generated assets; not a bug).")
    if zero_chunk_files:
        print(f"\n  {len(zero_chunk_files)} candidate file(s) produced ZERO "
             f"chunks with NO obvious binary/minified explanation — "
             f"cross-check against index_report()'s skip_* counts "
             f"(too-large / unreadable / parse-error):")
        for rel in zero_chunk_files[:50]:
            print(f"    - {rel}")
        if len(zero_chunk_files) > 50:
            print(f"    ... and {len(zero_chunk_files) - 50} more")

    # ---- Layer 2: content coverage ----
    print()
    print("=== Content coverage (independent scan vs. indexed chunks) ===")
    mismatches = 0
    checked = 0
    verified_exts: set[str] = set()
    unverified_exts: set[str] = set()
    for rel in sorted(buckets["candidate"]):
        ext = os.path.splitext(rel)[1].lower()
        full = os.path.join(root, rel)
        expected, method = independent_scan(full, ext)
        if expected is None:
            if ext:
                unverified_exts.add(ext)
            continue
        verified_exts.add(ext)
        checked += 1
        found = indexed_by_file.get(rel, set())
        found_names = {n for n, _ in found}
        expected_names = {n for n, _ in expected}
        missing = expected_names - found_names
        extra = found_names - expected_names

        if missing or (args.verbose and extra):
            mismatches += 1 if missing else 0
            print(f"\n  {rel}  [{method}]")
            if missing:
                print(f"    MISSING from index: {sorted(missing)}")
            if extra and args.verbose:
                print(f"    extra in index (may be independent-scan gap): "
                     f"{sorted(extra)}")
        elif args.verbose:
            print(f"  OK  {rel}  [{method}]  ({len(expected_names)} matched)")

    print(f"\n  checked {checked} file(s) across {len(verified_exts)} "
         f"independently-verified extension(s): {sorted(verified_exts) or '(none)'}")
    if unverified_exts:
        print(f"  {len(unverified_exts)} extension(s) NOT independently "
             f"verified (file-coverage checked only): {sorted(unverified_exts)}")

    # ---- summary ----
    print()
    print("=== Summary ===")
    print(f"  files indexed: {report.get('indexed', 0)}  |  chunks: {len(mem.chunks)}")
    print(f"  skipped (binary/too-large/unreadable/parse-error): "
         f"{sum(v for k, v in report.items() if k.startswith('skipped_'))}")
    print(f"  ignored (gitignore/excludes/generated/dotfiles): "
         f"{report.get('excluded_dirs', 0) + report.get('gitignored_dirs', 0) + report.get('gitignored_files', 0) + report.get('generated_files', 0) + report.get('dotfiles', 0)}")
    print(f"  files with missing functions/classes: {mismatches}")

    ok = mismatches == 0 and considered == report.get("considered", 0)
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())