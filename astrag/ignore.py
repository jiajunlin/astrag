"""Gitignore-style path filtering for the indexer.

Two independent, optional controls decide whether a file is even
*considered* for parsing, before any language-specific logic runs:

1. ``.gitignore`` (nested — each directory's own file layers on top of
   its parents', with negation) — the same files git itself wouldn't
   track are almost never worth indexing: build output, editor caches,
   coverage reports, anything the project already decided isn't source.
2. ``.astragignore`` — identical syntax, for astrag-specific excludes
   that don't belong in ``.gitignore`` (e.g. a ``fixtures/`` or
   ``testdata/`` folder that IS checked into git but has nothing
   indexable in it, or a vendored copy of a dependency you don't want
   cluttering search results).

Neither file existing is required — with nothing present, only the
built-in defaults in ``parsing.py`` (``DEFAULT_EXCLUDES`` directory
names, ``GENERATED_FILENAMES`` lockfiles) apply.

Scope, stated plainly: this is a pragmatic subset of the gitignore
spec, not a byte-for-byte reimplementation of git's own matcher. It
handles comments, blank lines, negation (``!pattern``), directory-only
patterns (trailing ``/``), anchored patterns (leading ``/``),
``*``/``?``/character classes, and ``**`` (any number of path segments,
including zero). It does not handle escaped special characters
(``\\#``, ``\\!``) or interactions with ``.gitattributes``. Good enough
to keep vendor/build noise out of an index; not a guarantee of matching
``git status`` on adversarial patterns.
"""
from __future__ import annotations

import os
import re


def _translate(pattern: str) -> tuple[re.Pattern, bool]:
    """One gitignore pattern line -> (compiled regex, dir_only)."""
    dir_only = pattern.endswith("/")
    pat = pattern.rstrip("/")
    anchored = pat.startswith("/") or "/" in pat[:-1]
    pat = pat.lstrip("/")

    out: list[str] = []
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        if pat[i:i + 3] == "**/":
            out.append("(?:.*/)?")
            i += 3
        elif pat[i:i + 2] == "**":
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "[":
            j = pat.find("]", i)
            if j == -1:
                out.append(re.escape(c))
                i += 1
            else:
                out.append(pat[i:j + 1])
                i = j + 1
        else:
            out.append(re.escape(c))
            i += 1
    body = "".join(out)
    suffix = "(?:/.*)?$"
    prefix = "^" if anchored else "(?:^|.*/)"
    return re.compile(prefix + body + suffix), dir_only


class IgnoreMatcher:
    """Accumulates gitignore-style rules while walking a repo top-down.

    Create once at the repo root. Call ``descend(rel_dir)`` the first
    time you enter a subdirectory (loads that directory's own
    ``.gitignore``/``.astragignore`` on top of the inherited rules —
    nested-gitignore semantics), then ``matches(rel_path, is_dir)`` to
    test any path relative to the repo root.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        # (base_dir_rel, regex, dir_only, negate), in file order so later
        # (more specific / more deeply-nested) rules can override earlier
        # ones, matching git's own precedence.
        self._rules: list[tuple[str, re.Pattern, bool, bool]] = []
        self._loaded_dirs: set[str] = set()
        self._load_dir("")

    def _load_dir(self, rel_dir: str) -> None:
        if rel_dir in self._loaded_dirs:
            return
        self._loaded_dirs.add(rel_dir)
        abs_dir = os.path.join(self.root, rel_dir) if rel_dir else self.root
        for fname in (".gitignore", ".astragignore"):
            fpath = os.path.join(abs_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()
            except OSError:
                continue
            for raw in lines:
                line = raw.rstrip("\n").rstrip("\r")
                if not line or line.lstrip().startswith("#"):
                    continue
                negate = line.startswith("!")
                if negate:
                    line = line[1:]
                if not line:
                    continue
                regex, dir_only = _translate(line)
                self._rules.append((rel_dir, regex, dir_only, negate))

    def descend(self, rel_dir: str) -> None:
        """Load ``rel_dir``'s own ignore files before matching inside it."""
        self._load_dir(rel_dir)

    def matches(self, rel_path: str, is_dir: bool = False) -> bool:
        """True if ``rel_path`` (posix-style, relative to the repo root)
        is ignored under the rules loaded so far."""
        ignored = False
        for base_dir, regex, dir_only, negate in self._rules:
            if dir_only and not is_dir:
                continue
            # a pattern only applies within the subtree it was declared in
            if base_dir and not (rel_path == base_dir or
                                 rel_path.startswith(base_dir + "/")):
                continue
            scoped = rel_path[len(base_dir) + 1:] if base_dir else rel_path
            if regex.match(scoped):
                ignored = not negate
        return ignored