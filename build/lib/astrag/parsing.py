"""Layer 1 — parsing: turn source files into *structural* chunks.

Instead of chunking by characters, code is split into logical units
(functions, methods, classes). Metadata (signature, docstring, calls,
imports, location) is extracted separately from the implementation, so
the retrieval layer can work with cheap "signature cards" and fetch full
bodies only on demand.

Two parsers are provided:

* ``PythonStdlibParser`` — reference implementation on Python's built-in
  ``ast`` module. Zero dependencies, rich extraction (docstrings,
  decorators, call names, imports).
* ``TreeSitterParser`` — optional multi-language adapter, used
  automatically when ``tree_sitter_language_pack`` (or the older
  ``tree_sitter_languages``) is installed. Best-effort extraction for
  JS/TS/Go/Rust/Java/…; it never breaks anything when absent.
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Code-aware tokenisation (shared by retrieval + compression layers)
# --------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def split_identifier(token: str) -> list[str]:
    """``parseHTTPResponse`` / ``parse_http_response`` -> [parse, http, response]."""
    return [p.lower() for p in _CAMEL.sub(" ", token.replace("_", " ")).split() if p]


def code_tokens(text: str) -> list[str]:
    """Tokenise code/text: identifiers plus their camelCase/snake_case parts."""
    out: list[str] = []
    for m in _IDENT.finditer(text or ""):
        tok = m.group(0)
        out.append(tok.lower())
        parts = split_identifier(tok)
        if len(parts) > 1:
            out.extend(parts)
    return out


def approx_tokens(text: str) -> int:
    """Cheap LLM-token estimate (~4 chars/token). Good enough for budgeting."""
    return max(1, (len(text) + 3) // 4)


# --------------------------------------------------------------------------
# The structural unit everything else operates on
# --------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """One logical unit of code (function / method / class) plus metadata."""

    chunk_id: str            # stable id: "<relpath>::<qualname>"
    kind: str                # "function" | "method" | "class"
    name: str
    qualname: str
    file: str
    start_line: int
    end_line: int
    signature: str           # e.g. "def retry_with_backoff(fn, retries=3) -> Any:"
    docstring: str | None
    source: str              # full implementation (only sent to the LLM on demand)
    language: str = "python"
    parent: str | None = None            # enclosing class for methods
    decorators: list = field(default_factory=list)
    calls: list = field(default_factory=list)     # simple names this chunk calls
    imports: list = field(default_factory=list)   # module-level imports of its file

    # ---- views ----
    def doc_summary(self) -> str:
        lines = (self.docstring or "").strip().splitlines()
        return lines[0].strip() if lines else ""

    def card(self) -> str:
        """Compact 'signature card' — what stage-2 retrieval shows the LLM."""
        out = [f"[{self.chunk_id}]  ({self.kind})", f"    {self.signature}"]
        if self.doc_summary():
            out.append(f'    """{self.doc_summary()}"""')
        out.append(f"    @ {self.file}:{self.start_line}-{self.end_line}")
        return "\n".join(out)

    def search_text(self) -> str:
        """Text the retrieval indexes are built from (never the full body)."""
        return "\n".join(
            [self.qualname, self.signature, self.docstring or "",
             " ".join(self.calls), self.file]
        )


# --------------------------------------------------------------------------
# Reference parser: Python stdlib ast
# --------------------------------------------------------------------------

class PythonStdlibParser:
    """Structural parser for Python built on the standard ``ast`` module."""

    language = "python"
    extensions = (".py",)

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        lines = source.splitlines()
        imports = self._imports(tree)
        chunks: list[CodeChunk] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunks.append(self._function(node, rel_path, lines, imports, parent=None))
            elif isinstance(node, ast.ClassDef):
                chunks.extend(self._class(node, rel_path, lines, imports))
        return chunks

    # ---- helpers ----
    @staticmethod
    def _imports(tree: ast.Module) -> list[str]:
        out: list[str] = []
        for n in tree.body:
            if isinstance(n, ast.Import):
                out += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mod = n.module or ""
                out += [f"{mod}.{a.name}" if mod else a.name for a in n.names]
        return out

    @staticmethod
    def _span(node, lines: list[str]) -> tuple[int, int, str]:
        start = node.lineno
        for d in getattr(node, "decorator_list", []):
            start = min(start, d.lineno)
        end = node.end_lineno
        return start, end, "\n".join(lines[start - 1:end])

    @staticmethod
    def _signature(node) -> str:
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        args = ast.unparse(node.args)
        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        return f"{prefix} {node.name}({args}){ret}:"

    @staticmethod
    def _calls(node) -> list[str]:
        """Call references with scope hints kept: bare names stay bare,
        ``self.m()``/``cls.m()`` become ``self.m``, and one-level module
        attribute calls become ``alias.f`` so the graph can resolve them
        against imports instead of string-matching every ``f``."""
        names: list[str] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                if isinstance(f, ast.Name):
                    names.append(f.id)
                elif isinstance(f, ast.Attribute):
                    base = f.value
                    if isinstance(base, ast.Name):
                        prefix = "self" if base.id in ("self", "cls") else base.id
                        names.append(f"{prefix}.{f.attr}")
                    else:
                        names.append(f.attr)
        seen: set[str] = set()
        return [n for n in names if not (n in seen or seen.add(n))]

    def _function(self, node, rel_path, lines, imports, parent) -> CodeChunk:
        start, end, src = self._span(node, lines)
        qual = f"{parent}.{node.name}" if parent else node.name
        return CodeChunk(
            chunk_id=f"{rel_path}::{qual}",
            kind="method" if parent else "function",
            name=node.name, qualname=qual, file=rel_path,
            start_line=start, end_line=end,
            signature=self._signature(node),
            docstring=ast.get_docstring(node),
            source=src, parent=parent,
            decorators=[ast.unparse(d) for d in node.decorator_list],
            calls=self._calls(node), imports=list(imports),
        )

    def _class(self, node, rel_path, lines, imports) -> list[CodeChunk]:
        start, end, src = self._span(node, lines)
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        sig = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
        out = [CodeChunk(
            chunk_id=f"{rel_path}::{node.name}", kind="class",
            name=node.name, qualname=node.name, file=rel_path,
            start_line=start, end_line=end, signature=sig,
            docstring=ast.get_docstring(node), source=src,
            decorators=[ast.unparse(d) for d in node.decorator_list],
            calls=[], imports=list(imports),
        )]
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(self._function(sub, rel_path, lines, imports, parent=node.name))
        return out


# --------------------------------------------------------------------------
# Optional multi-language parser: tree-sitter
# --------------------------------------------------------------------------

class TreeSitterParser:
    """Best-effort tree-sitter adapter for non-Python languages.

    Requires ``pip install tree-sitter-language-pack`` (or the older
    ``tree_sitter_languages``). When unavailable, ``for_language`` returns
    ``None`` and the pipeline simply skips those files. For Python the
    stdlib parser is preferred because its extraction is richer.
    """

    EXT_LANG = {
        ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java",
        ".rb": "ruby", ".c": "c", ".cpp": "cpp", ".py": "python",
    }
    NODE_KINDS = {
        "python": {"function_definition": "function", "class_definition": "class"},
        "javascript": {"function_declaration": "function",
                       "method_definition": "method",
                       "class_declaration": "class"},
        "typescript": {"function_declaration": "function",
                       "method_definition": "method",
                       "class_declaration": "class"},
        "go": {"function_declaration": "function",
               "method_declaration": "method"},
        "rust": {"function_item": "function", "struct_item": "class",
                 "impl_item": "class"},
        "java": {"method_declaration": "method", "class_declaration": "class"},
        "ruby": {"method": "function", "class": "class"},
        "c": {"function_definition": "function"},
        "cpp": {"function_definition": "function", "class_specifier": "class"},
    }
    _cache: dict = {}

    def __init__(self, language: str, parser) -> None:
        self.language = language
        self._parser = parser

    @classmethod
    def for_language(cls, language: str):
        if language in cls._cache:
            return cls._cache[language]
        parser = None
        for modname in ("tree_sitter_language_pack", "tree_sitter_languages"):
            try:
                mod = __import__(modname)
                parser = mod.get_parser(language)
                break
            except Exception:
                continue
        inst = cls(language, parser) if parser is not None else None
        cls._cache[language] = inst
        return inst

    @classmethod
    def available(cls, language: str) -> bool:
        return cls.for_language(language) is not None

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        if self._parser is None:
            return []
        tree = self._parser.parse(source.encode("utf-8"))
        src_lines = source.splitlines()
        kinds = self.NODE_KINDS.get(self.language, {})
        out: list[CodeChunk] = []
        stack = [(tree.root_node, None)]
        while stack:
            node, parent = stack.pop()
            kind = kinds.get(node.type)
            child_parent = parent
            if kind:
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode("utf-8", "replace") if name_node else None
                if name:
                    start = node.start_point[0] + 1
                    end = node.end_point[0] + 1
                    qual = f"{parent}.{name}" if parent and kind != "class" else name
                    out.append(CodeChunk(
                        chunk_id=f"{rel_path}::{qual}",
                        kind="method" if (parent and kind == "function") else kind,
                        name=name, qualname=qual, file=rel_path,
                        start_line=start, end_line=end,
                        signature=src_lines[start - 1].strip(),
                        docstring=None,
                        source="\n".join(src_lines[start - 1:end]),
                        language=self.language, parent=parent,
                    ))
                    if kind == "class":
                        child_parent = name
            for child in node.children:
                stack.append((child, child_parent))
        return out


# --------------------------------------------------------------------------
# Repo walking
# --------------------------------------------------------------------------

DEFAULT_EXCLUDES = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".pytest_cache", ".tox", ".astrag",
}


def iter_source_files(root: str, extensions: tuple[str, ...] | None,
                      excludes: set[str] = DEFAULT_EXCLUDES):
    """Yield ``(relative_path, absolute_path)`` for matching source files.

    ``extensions=None`` walks *every* file (binary/oversize filtering is
    the caller's job)."""
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in excludes and not d.startswith("."))
        for fn in sorted(filenames):
            if extensions is None or os.path.splitext(fn)[1] in extensions:
                full = os.path.join(dirpath, fn)
                yield os.path.relpath(full, root).replace(os.sep, "/"), full
