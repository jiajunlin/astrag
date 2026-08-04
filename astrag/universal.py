"""Universal file coverage — every text file in a repo becomes chunks.

Parser tiers (best available wins, routed by ``parser_for``):

1. exact parsers          Python stdlib ``ast`` / tree-sitter (pipeline)
2. brace engine           langs.py HeuristicParser (C-family, JS, Go, …)
3. end-block engine       Ruby, Lua, Elixir, Julia, Crystal, VB, MATLAB
4. format parsers         shell, SQL, Makefile, Dockerfile, Markdown,
                          YAML, TOML/INI, JSON
5. generic fallback       ANY other text file: comment-style sniffing +
                          a universal declaration detector + brace /
                          ``end`` / indentation block strategies — and if
                          a file has no recognisable structure at all it
                          is split into fixed-size ``segment`` chunks.

Nothing textual is ever unsupported; unknown languages just get coarser
chunk boundaries. ``looks_binary``/``BINARY_EXTENSIONS`` keep images,
archives and executables out of the index.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .langs import CssChunker, HtmlChunker, heuristic_parser_for
from .parsing import CodeChunk

_CALL_RX = re.compile(r"\b([A-Za-z_][\w.!?]*)\s*\(")

__all__ = ["parser_for", "looks_binary", "BINARY_EXTENSIONS",
           "EndBlockParser", "GenericCodeChunker"]


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def _mask_lines(lines: list[str], line_comment: tuple = ("#",),
                block_pairs: tuple = (), strings: str = "\"'") -> list[str]:
    """Blank comments/strings per line (length-preserving), tracking
    multi-line block comments across lines."""
    out: list[str] = []
    in_block: str | None = None            # the closing token we await
    for raw in lines:
        buf = list(raw)
        i, n = 0, len(raw)
        while i < n:
            if in_block:
                end = raw.find(in_block, i)
                stop = n if end < 0 else end + len(in_block)
                for j in range(i, stop):
                    buf[j] = " "
                if end < 0:
                    i = n
                else:
                    i = stop
                    in_block = None
                continue
            two = raw[i:i + 2]
            opened = next(((o, c) for o, c in block_pairs
                           if raw.startswith(o, i)), None)
            if opened:
                o, c = opened
                for j in range(i, min(n, i + len(o))):
                    buf[j] = " "
                i += len(o)
                in_block = c
                continue
            if any(raw.startswith(lc, i) for lc in line_comment):
                for j in range(i, n):
                    buf[j] = " "
                break
            ch = raw[i]
            if ch in strings:
                j = i + 1
                while j < n and raw[j] != ch:
                    if raw[j] == "\\":
                        j += 1
                    j += 1
                for k in range(i + 1, min(j, n)):
                    buf[k] = " "
                i = j + 1
                continue
            i += 1
        out.append("".join(buf))
    return out


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _leading_doc(lines: list[str], decl: int, markers: tuple) -> str | None:
    doc: list[str] = []
    i = decl - 1
    while i >= 0:
        s = lines[i].strip()
        if s and any(s.startswith(m) for m in markers):
            doc.append(s.lstrip("".join(set("".join(markers)))).strip())
            i -= 1
        else:
            break
    doc.reverse()
    text = " ".join(d for d in doc if d)
    return text[:600] or None


_DATA_KINDS = frozenset({"key", "section", "target", "stage", "segment",
                         "table", "index", "view", "sequence", "type",
                         "schema", "database", "trigger"})


def _chunk(rel_path, kind, name, qualname, start, end, lines, language,
           signature=None, doc=None, parent=None, calls=(), imports=()):
    if doc is None and kind in _DATA_KINDS:
        # data/doc chunks have no signature worth speaking of — surface
        # the body itself (truncated) so retrieval can see the content
        body_words = " ".join("\n".join(lines[start:end + 1]).split())
        doc = body_words[:400] or None
    return CodeChunk(
        chunk_id=f"{rel_path}::{qualname}", kind=kind, name=name,
        qualname=qualname, file=rel_path, start_line=start + 1,
        end_line=end + 1, signature=(signature or lines[start].strip())[:160],
        docstring=doc, source="\n".join(lines[start:end + 1]),
        language=language, parent=parent, calls=list(calls)[:40],
        imports=list(imports)[:40])


def _calls_in(text: str, deny: frozenset) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for m in _CALL_RX.finditer(text):
        n = m.group(1)
        if n not in deny and n not in seen:
            seen.add(n)
            out.append(n)
    return out[:40]


_DENY = frozenset(
    "if elsif elseif else for while unless until case when do then begin "
    "rescue ensure end return require require_relative import puts print "
    "raise throw new not and or in is def defp defmodule function local "
    "true false nil None sub my use print die exit switch".split())


# --------------------------------------------------------------------------
# tier 3 — end-block languages (def … end)
# --------------------------------------------------------------------------

@dataclass
class EndSpec:
    name: str
    extensions: tuple
    containers: re.Pattern            # class/module … (recurse, class-like)
    functions: re.Pattern             # def/function …
    line_comment: tuple = ("#",)
    block_pairs: tuple = ()
    end_rx: re.Pattern = re.compile(r"^[ \t]*end\b")
    import_rx: re.Pattern | None = None
    block_open_rx: re.Pattern | None = None   # depth counting (flat-indent langs)


RUBY_END = EndSpec(
    name="ruby", extensions=(".rb", ".rake", ".gemspec"),
    containers=re.compile(r"^[ \t]*(?P<kw>class|module)[ \t]+(?P<name>[\w:]+)"),
    functions=re.compile(r"^[ \t]*def[ \t]+(?:self\.)?(?P<name>[\w?!=\[\]<>+\-*\/%]+)"),
    block_pairs=(("=begin", "=end"),),
    import_rx=re.compile(r"^[ \t]*require(?:_relative)?[ \t]+['\"]([^'\"]+)"),
)

LUA_END = EndSpec(
    name="lua", extensions=(".lua",),
    containers=re.compile(r"$^"),                        # none
    functions=re.compile(r"^[ \t]*(?:local[ \t]+)?function[ \t]+"
                         r"(?P<name>[\w.:]+)|^[ \t]*(?:local[ \t]+)?"
                         r"(?P<name2>[\w.]+)[ \t]*=[ \t]*function[ \t]*\("),
    line_comment=("--",), block_pairs=(("--[[", "]]"),),
    import_rx=re.compile(r"require[ \t(]+['\"]([^'\"]+)"),
)

ELIXIR_END = EndSpec(
    name="elixir", extensions=(".ex", ".exs"),
    containers=re.compile(r"^[ \t]*defmodule[ \t]+(?P<name>[\w.]+)"),
    functions=re.compile(r"^[ \t]*def(?:p|macro|macrop)?[ \t]+"
                         r"(?P<name>[\w?!]+)"),
    import_rx=re.compile(r"^[ \t]*(?:alias|import|use)[ \t]+([\w.]+)"),
)

JULIA_END = EndSpec(
    name="julia", extensions=(".jl",),
    containers=re.compile(r"^[ \t]*(?:module)[ \t]+(?P<name>\w+)"),
    functions=re.compile(r"^[ \t]*(?:function|macro)[ \t]+(?P<name>[\w.!]+)"
                         r"|^[ \t]*(?:mutable[ \t]+)?struct[ \t]+(?P<name2>\w+)"),
    import_rx=re.compile(r"^[ \t]*(?:using|import)[ \t]+([\w.,: ]+)"),
    block_open_rx=re.compile(r"^[ \t]*(?:module|function|macro|(?:mutable[ \t]+)?"
                             r"struct|for|while|if|begin|let|try|quote)\b"),
)

CRYSTAL_END = EndSpec(
    name="crystal", extensions=(".cr",),
    containers=RUBY_END.containers, functions=RUBY_END.functions,
    import_rx=re.compile(r"^[ \t]*require[ \t]+\"([^\"]+)\""),
)

VB_END = EndSpec(
    name="vb", extensions=(".vb", ".bas"),
    containers=re.compile(r"^[ \t]*(?:Public[ \t]+|Private[ \t]+|Friend[ \t]+)?"
                          r"(?P<kw>Class|Module|Structure|Interface)[ \t]+"
                          r"(?P<name>\w+)", re.I),
    functions=re.compile(r"^[ \t]*(?:Public[ \t]+|Private[ \t]+|Protected[ \t]+|"
                         r"Friend[ \t]+|Shared[ \t]+|Overrides[ \t]+)*"
                         r"(?:Sub|Function|Property)[ \t]+(?P<name>\w+)", re.I),
    line_comment=("'",),
    end_rx=re.compile(r"^[ \t]*End[ \t]+\w+", re.I),
    import_rx=re.compile(r"^[ \t]*Imports[ \t]+([\w.]+)", re.I),
)

MATLAB_END = EndSpec(
    name="matlab", extensions=(),                       # via .m sniff only
    containers=re.compile(r"^[ \t]*classdef[ \t]+(?P<name>\w+)"),
    functions=re.compile(r"^[ \t]*function[ \t]+(?:[\[\]\w, ]+=[ \t]*)?"
                         r"(?P<name>\w+)"),
    line_comment=("%", "#"), block_pairs=(("%{", "%}"),),
)

FISH_END = EndSpec(
    name="fish", extensions=(".fish",),
    containers=re.compile(r"$^"),
    functions=re.compile(r"^[ \t]*function[ \t]+(?P<name>[\w.-]+)"),
    block_open_rx=re.compile(r"^[ \t]*(?:function|if|while|for|switch|begin)\b"),
)

FORTRAN_END = EndSpec(
    name="fortran", extensions=(".f90", ".f95", ".f03", ".f08", ".f", ".for"),
    containers=re.compile(r"^[ \t]*(?P<kw>module|program)[ \t]+(?P<name>\w+)",
                          re.I),
    functions=re.compile(r"^[ \t]*(?:pure[ \t]+|elemental[ \t]+|recursive[ \t]+|"
                         r"[\w()=, \t]*?)?(?:subroutine|function)[ \t]+"
                         r"(?P<name>\w+)", re.I),
    line_comment=("!",),
    end_rx=re.compile(r"^[ \t]*end\b", re.I),
    import_rx=re.compile(r"^[ \t]*use[ \t]+(\w+)", re.I),
)

END_SPECS = (RUBY_END, LUA_END, ELIXIR_END, JULIA_END, CRYSTAL_END, VB_END,
             FISH_END, FORTRAN_END)


class EndBlockParser:
    """Chunker for ``… end``-delimited languages.

    A declaration's block closes at the first ``end`` whose indentation
    is <= the declaration's (idiomatic nesting is indented deeper), or
    at the next declaration at <= indentation, or at EOF — so a missed
    ``end`` degrades a boundary, never the parse.
    """

    def __init__(self, spec: EndSpec) -> None:
        self.spec = spec
        self.language = spec.name

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        if not raw:
            return []
        masked = _mask_lines(raw, self.spec.line_comment,
                             self.spec.block_pairs)
        imports = []
        if self.spec.import_rx:
            imports = [m.group(1) for l in raw
                       for m in [self.spec.import_rx.match(l)] if m][:40]
        chunks: list[CodeChunk] = []
        self._scan(rel_path, raw, masked, 0, len(raw) - 1, None, False,
                   imports, chunks)
        return chunks

    def _decl(self, line: str):
        m = self.spec.containers.match(line)
        if m and m.group(0).strip():
            name = m.groupdict().get("name")
            if name:
                kind = (m.groupdict().get("kw") or "class").lower()
                return ("container", kind, name)
        m = self.spec.functions.match(line)
        if m:
            name = m.groupdict().get("name") or m.groupdict().get("name2")
            if name:
                kind = "struct" if "struct" in m.group(0) else "function"
                return ("function", kind, name)
        return None

    def _close(self, masked: list[str], decl: int, hi: int) -> int:
        if self.spec.block_open_rx is not None:
            depth = 1
            for j in range(decl + 1, hi + 1):
                if self.spec.end_rx.match(masked[j]):
                    depth -= 1
                    if depth == 0:
                        return j
                elif self.spec.block_open_rx.match(masked[j]):
                    depth += 1
            return hi
        base = _indent(masked[decl])
        for j in range(decl + 1, hi + 1):
            if not masked[j].strip():
                continue
            if self.spec.end_rx.match(masked[j]) and _indent(masked[j]) <= base:
                return j
            if _indent(masked[j]) <= base and self._decl(masked[j]):
                return j - 1
        return hi

    def _scan(self, rel, raw, masked, lo, hi, parent, parent_classlike,
              imports, chunks) -> None:
        i = lo
        while i <= hi:
            d = self._decl(masked[i])
            if not d:
                i += 1
                continue
            role, kind, name = d
            end = self._close(masked, i, hi)
            qual = f"{parent}.{name}" if parent else name
            body = "\n".join(masked[i:end + 1])
            if role == "container":
                chunks.append(_chunk(rel, kind, name, qual, i, end, raw,
                                     self.language, parent=parent,
                                     doc=_leading_doc(raw, i,
                                                      self.spec.line_comment),
                                     imports=imports))
                self._scan(rel, raw, masked, i + 1, end - 1, qual,
                           kind not in ("module",), imports, chunks)
            else:
                k = kind if kind == "struct" else (
                    "method" if parent_classlike else "function")
                chunks.append(_chunk(rel, k, name, qual, i, end, raw,
                                     self.language, parent=parent,
                                     doc=_leading_doc(raw, i,
                                                      self.spec.line_comment),
                                     calls=_calls_in(body, _DENY),
                                     imports=imports))
            i = end + 1


# --------------------------------------------------------------------------
# tier 4 — format parsers
# --------------------------------------------------------------------------

class ShellParser:
    language = "shell"
    extensions = (".sh", ".bash", ".zsh", ".ksh")
    _FN = re.compile(r"^[ \t]*(?:function[ \t]+)?(?P<name>[\w.-]+)[ \t]*"
                     r"\(\)[ \t]*\{|^[ \t]*function[ \t]+(?P<name2>[\w.-]+)"
                     r"[ \t]*\{")

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        masked = _mask_lines(raw, ("#",))
        chunks = []
        i = 0
        while i < len(raw):
            m = self._FN.match(masked[i])
            if not m:
                i += 1
                continue
            name = m.groupdict().get("name") or m.groupdict().get("name2")
            depth = 0
            end = i
            for j in range(i, len(raw)):
                depth += masked[j].count("{") - masked[j].count("}")
                if depth <= 0 and j > i:
                    end = j
                    break
            else:
                end = len(raw) - 1
            body = "\n".join(masked[i:end + 1])
            chunks.append(_chunk(rel_path, "function", name, name, i, end,
                                 raw, self.language,
                                 doc=_leading_doc(raw, i, ("#",)),
                                 calls=_calls_in(body, _DENY)))
            i = end + 1
        return chunks


class SqlParser:
    language = "sql"
    extensions = (".sql", ".ddl", ".psql")
    _CREATE = re.compile(
        r"^[ \t]*create[ \t]+(?:or[ \t]+replace[ \t]+)?(?:unique[ \t]+|"
        r"temp(?:orary)?[ \t]+|materialized[ \t]+)*"
        r"(?P<kw>table|view|function|procedure|trigger|index|sequence|type|"
        r"schema|database)[ \t]+(?:if[ \t]+not[ \t]+exists[ \t]+)?"
        r'(?P<name>[\w."]+)', re.I)

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        # mask comments, strings and $$-quoted bodies for boundary finding
        masked = _mask_lines(raw, ("--",), (("/*", "*/"), ("$$", "$$")), "'")
        chunks = []
        i = 0
        while i < len(raw):
            m = self._CREATE.match(masked[i])
            if not m:
                i += 1
                continue
            end = i
            for j in range(i, len(raw)):
                if ";" in masked[j]:
                    end = j
                    break
            else:
                end = len(raw) - 1
            name = m.group("name").strip('"')
            kind = m.group("kw").lower()
            qual = name
            n = 2
            while any(c.qualname == qual for c in chunks):
                qual = f"{name}#{n}"
                n += 1
            chunks.append(_chunk(rel_path, kind, name, qual, i, end, raw,
                                 self.language,
                                 doc=_leading_doc(raw, i, ("--",))))
            i = end + 1
        return chunks


class MakefileParser:
    language = "make"
    extensions = (".mk",)
    filenames = ("Makefile", "makefile", "GNUmakefile")
    _TARGET = re.compile(r"^(?P<name>[^\s:=#]+)[ \t]*:(?!=)")

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        targets = [(i, m.group("name")) for i, l in enumerate(raw)
                   for m in [self._TARGET.match(l)]
                   if m and not m.group("name").startswith(".")]
        chunks = []
        for n, (i, name) in enumerate(targets):
            end = (targets[n + 1][0] - 1) if n + 1 < len(targets) \
                else len(raw) - 1
            while end > i and not raw[end].strip():
                end -= 1
            chunks.append(_chunk(rel_path, "target", name, name, i, end, raw,
                                 self.language,
                                 doc=_leading_doc(raw, i, ("#",))))
        return chunks


class DockerfileParser:
    language = "dockerfile"
    extensions = ()
    filenames = ("Dockerfile", "Containerfile")
    _FROM = re.compile(r"^FROM[ \t]+(?P<base>\S+)(?:[ \t]+AS[ \t]+"
                       r"(?P<name>\S+))?", re.I)

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        stages = [(i, m) for i, l in enumerate(raw)
                  for m in [self._FROM.match(l)] if m]
        if not stages:
            return [_chunk(rel_path, "stage", "Dockerfile", "Dockerfile",
                           0, len(raw) - 1, raw, self.language)] if raw else []
        chunks = []
        for n, (i, m) in enumerate(stages):
            end = (stages[n + 1][0] - 1) if n + 1 < len(stages) \
                else len(raw) - 1
            name = m.group("name") or f"stage{n + 1}"
            chunks.append(_chunk(rel_path, "stage", name, name, i, end, raw,
                                 self.language,
                                 signature=raw[i].strip()))
        return chunks


class MarkdownChunker:
    language = "markdown"
    extensions = (".md", ".markdown", ".mdx")
    _H = re.compile(r"^(#{1,6})[ \t]+(?P<t>.+?)[ \t]*#*$")

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        heads = []
        fence = False
        for i, l in enumerate(raw):
            if l.lstrip().startswith("```"):
                fence = not fence
            if fence:
                continue
            m = self._H.match(l)
            if m:
                heads.append((i, len(m.group(1)), m.group("t").strip()))
        chunks = []
        stack: list[tuple[int, str]] = []           # (level, qualname)
        for n, (i, lvl, title) in enumerate(heads):
            end = (heads[n + 1][0] - 1) if n + 1 < len(heads) else len(raw) - 1
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            parent = stack[-1][1] if stack else None
            name = title[:80]
            qual = f"{parent} > {name}" if parent else name
            m = 2
            base = qual
            while any(c.qualname == qual for c in chunks):
                qual = f"{base}#{m}"
                m += 1
            chunks.append(_chunk(rel_path, "section", name, qual, i, end, raw,
                                 self.language, parent=parent,
                                 signature="#" * lvl + " " + name))
            stack.append((lvl, qual))
        if not chunks and raw:
            chunks.append(_chunk(rel_path, "section", os.path.basename(rel_path),
                                 os.path.basename(rel_path), 0, len(raw) - 1,
                                 raw, self.language))
        return chunks


class YamlChunker:
    language = "yaml"
    extensions = (".yml", ".yaml", ".cff")
    _KEY = re.compile(r"^(?P<name>[\w./-]+)[ \t]*:")

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        keys = [(i, m.group("name")) for i, l in enumerate(raw)
                for m in [self._KEY.match(l)] if m]
        chunks = []
        for n, (i, name) in enumerate(keys):
            end = (keys[n + 1][0] - 1) if n + 1 < len(keys) else len(raw) - 1
            qual, m = name, 2
            while any(c.qualname == qual for c in chunks):
                qual = f"{name}#{m}"
                m += 1
            chunks.append(_chunk(rel_path, "key", name, qual, i, end, raw,
                                 self.language))
        return chunks


class TomlIniChunker:
    language = "toml"
    extensions = (".toml", ".ini", ".cfg", ".conf", ".properties", ".env")
    _SEC = re.compile(r"^\[+(?P<name>[^\]]+)\]+")

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        secs = [(i, m.group("name").strip()) for i, l in enumerate(raw)
                for m in [self._SEC.match(l)] if m]
        chunks = []
        if secs and secs[0][0] > 0:
            chunks.append(_chunk(rel_path, "section", "(top)", "(top)", 0,
                                 secs[0][0] - 1, raw, self.language))
        for n, (i, name) in enumerate(secs):
            end = (secs[n + 1][0] - 1) if n + 1 < len(secs) else len(raw) - 1
            qual, m = name, 2
            while any(c.qualname == qual for c in chunks):
                qual = f"{name}#{m}"
                m += 1
            chunks.append(_chunk(rel_path, "section", name, qual, i, end, raw,
                                 self.language))
        if not chunks and raw:
            chunks.append(_chunk(rel_path, "section", "(top)", "(top)", 0,
                                 len(raw) - 1, raw, self.language))
        return chunks


class JsonChunker:
    language = "json"
    extensions = (".json", ".jsonc", ".json5", ".ipynb", ".geojson")
    _KEY = re.compile(r'^[ \t]{1,4}"(?P<name>[^"]+)"[ \t]*:')

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        keys = [(i, m.group("name")) for i, l in enumerate(raw)
                for m in [self._KEY.match(l)] if m]
        # only keep keys at the minimal (top) indentation level
        if keys:
            min_ind = min(_indent(raw[i]) for i, _ in keys)
            keys = [(i, n) for i, n in keys if _indent(raw[i]) == min_ind]
        chunks = []
        for n, (i, name) in enumerate(keys):
            end = (keys[n + 1][0] - 1) if n + 1 < len(keys) else len(raw) - 1
            qual, m = name, 2
            while any(c.qualname == qual for c in chunks):
                qual = f"{name}#{m}"
                m += 1
            chunks.append(_chunk(rel_path, "key", name, qual, i, end, raw,
                                 self.language))
        if not chunks and raw:
            chunks.append(_chunk(rel_path, "key", os.path.basename(rel_path),
                                 os.path.basename(rel_path), 0, len(raw) - 1,
                                 raw, self.language))
        return chunks


class CMakeParser:
    language = "cmake"
    extensions = (".cmake",)
    filenames = ("CMakeLists.txt",)
    _FN = re.compile(r"^[ \t]*(?P<kw>function|macro)[ \t]*\([ \t]*"
                     r"(?P<name>\w+)", re.I)
    _END = re.compile(r"^[ \t]*end(?:function|macro)\b", re.I)

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        masked = _mask_lines(raw, ("#",))
        chunks = []
        covered: set[int] = set()
        i = 0
        while i < len(raw):
            m = self._FN.match(masked[i])
            if not m:
                i += 1
                continue
            end = next((j for j in range(i + 1, len(raw))
                        if self._END.match(masked[j])), len(raw) - 1)
            name = m.group("name")
            chunks.append(_chunk(rel_path, m.group("kw").lower(), name, name,
                                 i, end, raw, self.language,
                                 doc=_leading_doc(raw, i, ("#",)),
                                 calls=_calls_in("\n".join(masked[i:end + 1]),
                                                 _DENY)))
            covered.update(range(i, end + 1))
            i = end + 1
        rest = [i for i in range(len(raw))
                if i not in covered and raw[i].strip()]
        if rest:
            chunks.append(_chunk(rel_path, "section", "(top)", "(top)",
                                 rest[0], rest[-1], raw, self.language))
        return chunks


class LatexChunker:
    language = "latex"
    extensions = (".tex", ".ltx")
    _SEC = re.compile(r"^[ \t]*\\(?P<lvl>part|chapter|section|subsection|"
                      r"subsubsection)\*?\{(?P<t>[^}]*)\}")
    _ORDER = {"part": 0, "chapter": 1, "section": 2, "subsection": 3,
              "subsubsection": 4}

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        heads = [(i, self._ORDER[m.group("lvl")], m.group("t").strip())
                 for i, l in enumerate(raw)
                 for m in [self._SEC.match(l)] if m]
        chunks = []
        stack: list[tuple[int, str]] = []
        for n, (i, lvl, title) in enumerate(heads):
            end = (heads[n + 1][0] - 1) if n + 1 < len(heads) else len(raw) - 1
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            parent = stack[-1][1] if stack else None
            name = title[:80] or f"L{i + 1}"
            qual = f"{parent} > {name}" if parent else name
            m2, base = 2, qual
            while any(c.qualname == qual for c in chunks):
                qual = f"{base}#{m2}"
                m2 += 1
            chunks.append(_chunk(rel_path, "section", name, qual, i, end,
                                 raw, self.language, parent=parent))
            stack.append((lvl, qual))
        if not chunks and raw:
            chunks.append(_chunk(rel_path, "section",
                                 os.path.basename(rel_path),
                                 os.path.basename(rel_path), 0,
                                 len(raw) - 1, raw, self.language))
        return chunks


# --------------------------------------------------------------------------
# tier 5 — the generic fallback: any text file at all
# --------------------------------------------------------------------------

class GenericCodeChunker:
    """Best-effort chunker for unrecognised languages.

    Sniffs the comment style, looks for universal declaration keywords
    (``def fn func function sub proc procedure fun macro class module
    struct type interface trait impl object enum`` plus ``name() {`` and
    assignment-to-function forms), and closes blocks by braces, matching
    ``end``, or indentation — whichever the file appears to use. Files
    with no recognisable declarations are split into fixed ``segment``
    chunks so they remain searchable.
    """

    SEGMENT = 40
    _DECL = re.compile(
        r"^[ \t]*(?:(?:public|private|protected|static|export|pub|local|"
        r"global|inline|async|extern)\s+)*"
        r"(?P<kw>def|fn|func|function|sub|proc|procedure|fun|macro|class|"
        r"module|struct|type|interface|trait|impl|object|enum|record|"
        r"contract|task|rule|section)\s+(?P<name>[\w.:!?$-]+)"
        r"|^[ \t]*(?P<name2>[\w.-]+)[ \t]*\(\)[ \t]*\{"
        r"|^[ \t]*(?P<name3>[\w.]+)[ \t]*(?:=|:=|<-)[ \t]*"
        r"(?:function|fn|func|lambda|proc)\b"
        r"|^(?P<name4>[a-z_][\w']*)[ \t]*::(?!:)"          # Haskell sig
        r"|^let[ \t]+(?:rec[ \t]+)?(?P<name5>[\w']+)")     # OCaml / F#

    def __init__(self, language: str = "text") -> None:
        self.language = language

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        raw = source.splitlines()
        if not raw:
            return []
        style = self._comment_style(raw)
        masked = _mask_lines(raw, style, (("/*", "*/"),) if "//" in style else ())
        decls = []
        for i, l in enumerate(masked):
            m = self._DECL.match(l)
            if m:
                gd = m.groupdict()
                name = (gd.get("name") or gd.get("name2") or gd.get("name3")
                        or gd.get("name4") or gd.get("name5"))
                if not name:
                    continue
                kind = (gd.get("kw") or "function").lower()
                if kind in ("def", "fn", "func", "function", "sub", "proc",
                            "procedure", "fun", "macro", "task"):
                    kind = "function"
                decls.append((i, kind, name))
        if not decls:
            return self._segments(rel_path, raw)

        uses_braces = sum(l.count("{") for l in masked) >= max(1, len(decls))
        uses_end = any(re.match(r"^[ \t]*end\b", l) for l in masked)
        chunks = []
        for n, (i, kind, name) in enumerate(decls):
            nxt = decls[n + 1][0] - 1 if n + 1 < len(decls) else len(raw) - 1
            end = self._close(masked, i, nxt, uses_braces, uses_end)
            # Haskell-style bindings: `f :: sig` on one line, equations
            # `f x = …` (same identifier, column 0) plus their indented
            # continuations on the following lines — pull them in
            base = _indent(masked[i])
            while end < nxt:
                s = masked[end + 1]
                if s.strip() and (s.startswith(name)
                                  or _indent(s) > base):
                    end += 1
                else:
                    break
            qual, m = name, 2
            while any(c.qualname == qual for c in chunks):
                qual = f"{name}#{m}"
                m += 1
            body = "\n".join(masked[i:end + 1])
            chunks.append(_chunk(rel_path, kind, name, qual, i, end, raw,
                                 self.language,
                                 doc=_leading_doc(raw, i, style),
                                 calls=_calls_in(body, _DENY)))
        return chunks

    @staticmethod
    def _comment_style(raw: list[str]) -> tuple:
        votes = {"#": 0, "//": 0, "--": 0, ";": 0, "%": 0, "'": 0}
        for l in raw[:400]:
            s = l.lstrip()
            for tok in votes:
                if s.startswith(tok):
                    votes[tok] += 1
        best = max(votes, key=lambda t: votes[t])
        return (best,) if votes[best] else ("#", "//")

    def _close(self, masked, decl, limit, braces, ends) -> int:
        base = _indent(masked[decl])
        if braces and "{" in "".join(masked[decl:min(decl + 3, limit + 1)]):
            depth = 0
            seen = False
            for j in range(decl, limit + 1):
                depth += masked[j].count("{") - masked[j].count("}")
                if "{" in masked[j]:
                    seen = True
                if seen and depth <= 0:
                    return j
            return limit
        if ends:
            for j in range(decl + 1, limit + 1):
                if re.match(r"^[ \t]*end\b", masked[j]) \
                        and _indent(masked[j]) <= base:
                    return j
        # indentation block
        end = decl
        for j in range(decl + 1, limit + 1):
            if not masked[j].strip():
                continue
            if _indent(masked[j]) <= base:
                break
            end = j
        return max(end, decl)

    def _segments(self, rel_path, raw) -> list[CodeChunk]:
        chunks = []
        for start in range(0, len(raw), self.SEGMENT):
            end = min(start + self.SEGMENT - 1, len(raw) - 1)
            name = f"L{start + 1}-{end + 1}"
            chunks.append(_chunk(rel_path, "segment", name, name, start, end,
                                 raw, self.language))
        return chunks


# --------------------------------------------------------------------------
# .m disambiguation (Objective-C vs MATLAB)
# --------------------------------------------------------------------------

class MSniffParser:
    language = "objc"

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        head = source[:4000]
        if "#import" in head or "@interface" in head \
                or "@implementation" in head:
            return heuristic_parser_for(".mm").parse_file(rel_path, source)
        return EndBlockParser(MATLAB_END).parse_file(rel_path, source)


# --------------------------------------------------------------------------
# binary detection + master router
# --------------------------------------------------------------------------

BINARY_EXTENSIONS = frozenset("""
.png .jpg .jpeg .gif .bmp .ico .icns .webp .tiff .heic .svgz
.pdf .zip .gz .bz2 .xz .zst .7z .rar .tar .tgz .whl .deb .rpm .dmg .iso
.exe .dll .so .dylib .a .o .obj .lib .class .jar .war .pyc .pyd .wasm
.mp3 .mp4 .wav .flac .ogg .avi .mov .mkv .webm
.woff .woff2 .ttf .otf .eot
.db .sqlite .sqlite3 .parquet .pickle .pkl .npy .npz .pb .onnx .bin .dat
.min.js .min.css .map .lock .sum
""".split())

MAX_FILE_BYTES = 1_500_000


def looks_binary(source: str) -> bool:
    head = source[:8000]
    if "\x00" in head:
        return True
    lines = head.splitlines() or [""]
    return max(len(l) for l in lines) > 4000        # minified / generated


_FILENAME_PARSERS = {}
for _cls in (MakefileParser, DockerfileParser, CMakeParser):
    for _fn in _cls.filenames:
        _FILENAME_PARSERS[_fn] = _cls

_EXT_PARSERS = {}
for _cls in (ShellParser, SqlParser, MakefileParser, MarkdownChunker,
             YamlChunker, TomlIniChunker, JsonChunker, CMakeParser,
             LatexChunker):
    for _e in _cls.extensions:
        _EXT_PARSERS[_e] = _cls

_cache: dict[str, object] = {}


def parser_for(rel_path: str):
    """Master router: a parser instance for *any* text file path.

    Never returns None — the generic fallback covers everything.
    (Python is handled upstream by the pipeline's exact parsers.)
    """
    base = os.path.basename(rel_path)
    ext = os.path.splitext(base)[1]
    key = base if base in _FILENAME_PARSERS else (ext or base)
    if key in _cache:
        return _cache[key]

    inst = None
    if base in _FILENAME_PARSERS:
        inst = _FILENAME_PARSERS[base]()
    elif ext == ".m":
        inst = MSniffParser()
    elif ext in (".xml", ".svg", ".xhtml", ".vue", ".svelte", ".astro",
                 ".erb", ".ejs", ".njk", ".hbs", ".mustache", ".liquid",
                 ".jinja", ".jinja2", ".j2", ".twig"):
        inst = HtmlChunker()
    else:
        inst = heuristic_parser_for(ext)            # brace / html / css
        if inst is None:
            for spec in END_SPECS:
                if ext in spec.extensions:
                    inst = EndBlockParser(spec)
                    break
        if inst is None and ext in _EXT_PARSERS:
            inst = _EXT_PARSERS[ext]()
        if inst is None:
            inst = GenericCodeChunker(language=ext.lstrip(".") or "text")
    _cache[key] = inst
    return inst


KNOWN_EXTENSIONS = tuple(sorted(
    set(_EXT_PARSERS) | {e for s in END_SPECS for e in s.extensions}
    | {".m", ".xml", ".svg", ".xhtml", ".vue", ".svelte", ".astro", ".erb",
       ".ejs", ".njk", ".hbs", ".mustache", ".liquid", ".jinja", ".jinja2",
       ".j2", ".twig"}))