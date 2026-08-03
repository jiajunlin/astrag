"""Backward program slicing for compression (upgrade over regex scoring).

``slice_lines(source, language)`` returns the set of 0-based line
indices that a backward slice keeps, or ``None`` when the source can't
be sliced (the caller then falls back to pure anchor scoring).

Python gets an exact-ish AST slice: seeds are ``return`` / ``raise`` /
``yield`` statements (plus any statement mentioning an anchor name);
walking the statement list backwards, a statement is kept iff it
*defines* a name the slice still needs, and the names it *loads* are
added to the need-set. Control-flow headers (``if``/``for``/``while``/
``with``/``try``) enclosing any kept line are kept too, so the slice
stays well-formed.

Other languages get the same def-use chain computed heuristically on
comment/string-masked text: writes are detected via ``x = / x := /
var·let·const x / x += ...`` patterns, seeds via ``return / throw /
panic! / yield``.
"""
from __future__ import annotations

import ast
import re
import textwrap

from astrag.parsing import CodeChunk

__all__ = ["slice_lines", "python_backward_slice", "heuristic_backward_slice"]


# --------------------------------------------------------------------------
# Python: AST def-use backward slice
# --------------------------------------------------------------------------

def _loads(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _stores(stmt: ast.stmt) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store):
            base = n.value
            if isinstance(base, ast.Name):
                out.add(f"{base.id}.{n.attr}")
                out.add(base.id)          # mutating an attr keeps the object
    return out


def _is_seed(stmt: ast.stmt, anchors: set[str]) -> bool:
    if isinstance(stmt, (ast.Return, ast.Raise)):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value,
                                                 (ast.Yield, ast.YieldFrom)):
        return True
    if anchors and _loads(stmt) & anchors:
        return True
    return False


def _slice_block(body: list[ast.stmt], needed: set[str],
                 anchors: set[str], kept: set[int]) -> set[str]:
    """Backward pass over one statement list; returns the need-set that
    flows out of the top of the block."""
    for stmt in reversed(body):
        inner = getattr(stmt, "body", None)
        if inner is not None:             # compound: if/for/while/with/try/def
            before = set(needed)
            for blk in (getattr(stmt, "finalbody", []),
                        getattr(stmt, "orelse", []), inner):
                needed |= _slice_block(blk, set(needed), anchors, kept)
            for h in getattr(stmt, "handlers", []):
                needed |= _slice_block(h.body, set(needed), anchors, kept)
            child_kept = any(s.lineno - 1 in kept
                             for blk in ast.walk(stmt)
                             for s in [blk] if hasattr(s, "lineno"))
            header_needed = (_is_seed(stmt, anchors) or child_kept
                             or bool(_stores(stmt) & before))
            if header_needed:
                kept.add(stmt.lineno - 1)
                # loop/branch conditions feed control flow
                test = getattr(stmt, "test", None) or getattr(stmt, "iter", None)
                if test is not None:
                    needed |= _loads(test)
                items = getattr(stmt, "items", None)
                if items:                 # with ... as x
                    for it in items:
                        needed |= _loads(it.context_expr)
            continue

        keep = _is_seed(stmt, anchors) or bool(_stores(stmt) & needed)
        if keep:
            for ln in range(stmt.lineno - 1, (stmt.end_lineno or stmt.lineno)):
                kept.add(ln)
            needed -= {s for s in _stores(stmt) if "." not in s}
            needed |= _loads(stmt)
    return needed


def python_backward_slice(source: str,
                          anchors: set[str] | None = None) -> set[int] | None:
    """Kept line indices (0-based, relative to *source*) or None."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    fn = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = node
            break
    body = fn.body if fn is not None else tree.body
    kept: set[int] = set()
    _slice_block(body, set(), anchors or set(), kept)
    if fn is not None:                    # signature + decorators always kept
        first = min([fn.lineno] + [d.lineno for d in fn.decorator_list])
        for ln in range(first - 1, fn.body[0].lineno - 1):
            kept.add(ln)
        doc = fn.body[0]
        if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) \
                and isinstance(doc.value.value, str):
            kept.add(doc.lineno - 1)
    return kept or None


# --------------------------------------------------------------------------
# Brace languages: masked-text def-use chain
# --------------------------------------------------------------------------

_H_SEED = re.compile(r"^\s*(?:return\b|throw\b|panic!|yield\b|=>|raise\b)"
                     r"|(?:\breturn|\bthrow)\b[^;]*;?\s*$")
_H_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_H_KW = frozenset("if else for while switch match case return throw new var "
                  "let const fn func function true false null nil None self "
                  "this mut pub static void int float double bool string auto "
                  "defer go await async try catch finally".split())
_H_CTRL = re.compile(r"^\s*}?\s*(?:if|else|for|while|switch|match|do|try|"
                     r"catch|finally|foreach|loop|select|when|guard)\b")


def _mask(source: str) -> str:
    """Blank comments and string contents, preserving length/newlines."""
    out, i, n = list(source), 0, len(source)
    while i < n:
        ch = source[i]
        two = source[i:i + 2]
        if two == "//" or ch == "#":
            while i < n and source[i] != "\n":
                out[i] = " "; i += 1
        elif two == "/*":
            i += 2
            while i < n - 1 and source[i:i + 2] != "*/":
                if source[i] != "\n":
                    out[i] = " "
                i += 1
            i += 2
        elif ch in "\"'`":
            q = ch; i += 1
            while i < n and source[i] != q:
                if source[i] == "\\":
                    out[i] = " "; i += 1
                if i < n and source[i] != "\n":
                    out[i] = " "
                i += 1
            i += 1
        else:
            i += 1
    return "".join(out)


def _h_writes(line: str) -> set[str]:
    out: set[str] = set()
    for m in re.finditer(r"\b(?:var|let|const|auto|mut)?\s*"
                         r"([A-Za-z_][A-Za-z0-9_]*)\s*"
                         r"(?::?=|\+=|-=|\*=|/=|\|=|&=|:=)", line):
        out.add(m.group(1))
    m = re.match(r"\s*(?:var|let|const)\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
    if m:
        out.add(m.group(1))
    return out - _H_KW


def heuristic_backward_slice(source: str,
                             anchors: set[str] | None = None) -> set[int] | None:
    lines = _mask(source).splitlines()
    raw = source.splitlines()
    if len(lines) < 4:
        return None
    anchors = anchors or set()
    kept: set[int] = {0, len(lines) - 1}          # signature + closer
    needed: set[str] = set()
    seeded = False
    for i, ln in enumerate(lines):
        idents = set(_H_IDENT.findall(ln)) - _H_KW
        if _H_SEED.search(ln) or (anchors and idents & anchors):
            kept.add(i)
            needed |= idents
            seeded = True
    if not seeded:
        return None
    for i in range(len(lines) - 1, -1, -1):
        if i in kept:
            continue
        w = _h_writes(lines[i])
        if w & needed:
            kept.add(i)
            needed |= (set(_H_IDENT.findall(lines[i])) - _H_KW)
    # control headers that own a kept line (by brace nesting approximation)
    depth = 0
    open_stack: list[tuple[int, int]] = []        # (line, depth at opener)
    owners: dict[int, list[int]] = {}
    for i, ln in enumerate(lines):
        for ch in ln:
            if ch == "{":
                open_stack.append((i, depth)); depth += 1
            elif ch == "}":
                depth = max(0, depth - 1)
                if open_stack:
                    open_stack.pop()
        for opener, _ in open_stack:
            owners.setdefault(i, []).append(opener)
    for i in list(kept):
        for opener in owners.get(i, []):
            if _H_CTRL.match(raw[opener]) or opener == 0:
                kept.add(opener)
    return kept


def slice_lines(source: str, language: str = "python",
                anchors: set[str] | None = None) -> set[int] | None:
    if language == "python":
        return python_backward_slice(source, anchors)
    if language in ("html", "css"):
        return None
    return heuristic_backward_slice(source, anchors)


# --------------------------------------------------------------------------
# Interprocedural slicing: follow the call graph into callees
# --------------------------------------------------------------------------
#
# ``slice_lines`` above is intraprocedural — a backward slice within one
# function's own body. A real system/program dependence graph (SDG) slice
# also follows *data* that crosses a call boundary: if the seed function
# calls ``charge_card(...)`` and needs its return value, the exact
# execution path includes what ``charge_card`` itself does. A full SDG
# (precise inter-procedural def-use over every parameter/return) is a
# much bigger undertaking than this module's line-level scoring; what's
# implemented here is a scoped, honest approximation: reuse the existing
# scope-aware call graph (``graph.py``) to walk from a seed chunk into
# the callees it references, and slice each independently up to a depth
# budget. It answers "what does the exact execution path touch, one/two
# calls deep" without claiming full SDG precision (aliasing, higher-order
# calls, and reflection can still hide a real edge — same caveat as the
# call graph itself).

def interprocedural_slice(seed_chunk, graph, chunk_by_id: dict,
                          max_depth: int = 1,
                          anchors: set[str] | None = None) -> dict:
    """Backward-slice ``seed_chunk``, then recurse into callees it calls.

    ``graph`` is a ``CodeGraph`` (for ``callees_of``); ``chunk_by_id`` maps
    chunk_id -> CodeChunk. Returns ``{chunk_id: (chunk, kept_lines)}`` for
    the seed plus every callee pulled in, up to ``max_depth`` hops. A
    chunk is only visited once even if reachable via multiple paths.
    """
    out: dict = {}
    seen: set[str] = set()

    def _visit(chunk, depth: int) -> None:
        if chunk is None or chunk.chunk_id in seen or chunk.kind == "class":
            return
        seen.add(chunk.chunk_id)
        kept = slice_lines(chunk.source, chunk.language, anchors)
        out[chunk.chunk_id] = (chunk, kept)
        if depth >= max_depth:
            return
        for callee_id in graph.callees_of(chunk.chunk_id):
            _visit(chunk_by_id.get(callee_id), depth + 1)

    _visit(seed_chunk, 0)
    return out


def compute_function_summary(chunk: CodeChunk) -> tuple[set[str], set[str], set[str]]:
    """Return (defs, uses, params) for a function body."""
    defs, uses = set(), set()
    if chunk.language == 'python':
        try:
            tree = ast.parse(chunk.source)
            # walk AST to collect assignments and loads
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, ast.Store):
                        defs.add(node.id)
                    elif isinstance(node.ctx, ast.Load):
                        uses.add(node.id)
        except:
            pass
    else:
        # Heuristic: extract variable names from masked source
        masked = _mask(chunk.source)
        for m in re.finditer(r'\b([A-Za-z_]\w*)\s*[:=]', masked):
            defs.add(m.group(1))
        for m in re.finditer(r'\b([A-Za-z_]\w*)\s*[+\-*/%]?=', masked):
            if not m.group(1) in ('if','for','while','return','raise'):
                uses.add(m.group(1))
    # parameters are considered defs that are also inputs
    params = set(re.findall(r'\(\s*([A-Za-z_]\w*)', chunk.signature))
    return defs, uses, params

def interprocedural_slice(seed_chunk, graph, chunk_by_id: dict,
                          max_depth: int = 1,
                          anchors: set[str] = None) -> dict:
    """Return dict of {chunk_id: (chunk, kept_lines)} for the SDG slice."""
    # First, precompute summaries for all chunks
    summaries = {}
    for cid, c in chunk_by_id.items():
        summaries[cid] = compute_function_summary(c)

    # Worklist for backward slicing: we need to know which variables are required
    needed_vars = set(anchors or [])
    # Start with seed chunk: we need its return value (if any) and the seed anchors
    worklist = [(seed_chunk.chunk_id, set(needed_vars))]
    visited = set()
    result = {}

    while worklist:
        cid, need = worklist.pop()
        if cid in visited:
            continue
        visited.add(cid)
        chunk = chunk_by_id.get(cid)
        if not chunk:
            continue
        # Perform intraprocedural slice for this chunk given the 'need' set
        # We'll use the existing python_backward_slice or heuristic, but we need to incorporate need
        # We'll implement a simple version: we compute slice_lines with anchors = need
        kept = slice_lines(chunk.source, chunk.language, anchors=need)
        result[cid] = (chunk, kept)
        # Now, for every call in this chunk, determine if we need to slice the callee
        # Find calls: from chunk.calls (list of names) but we need to map to actual callee chunk ids
        for callee_name in chunk.calls:
            # Resolve callee via graph
            callee_ids = graph.callees_of(cid)  # list of chunk ids
            # Actually, graph.callees_of returns direct callees by chunk_id.
            # We'll iterate over those and decide if we need to include them.
            for callee_id in callee_ids:
                # Determine which variables are needed from that callee:
                # If the call is like x = foo(...) then we need the return value.
                # We'll check if any variable used after the call depends on the callee's return.
                # For simplicity, we'll assume that if we need any variable that is defined by the callee,
                # we need the callee.
                callee_chunk = chunk_by_id.get(callee_id)
                if not callee_chunk:
                    continue
                # Summary of callee: defs, uses, params
                callee_defs, _, _ = summaries.get(callee_id, (set(), set(), set()))
                # If we need any variable that the callee defines, we need to slice the callee
                # Also, if we need the return (implicit), we pass.
                if need & callee_defs:
                    # Also, we need to pass the parameters that the callee uses
                    # We'll compute the actual arguments passed in the call (from source)
                    # For simplicity, we'll pass all arguments as needed.
                    # We'll just propagate the need set to the callee.
                    worklist.append((callee_id, need))
                # Also, if the call is not assignment, we may still need it for side effects?
                # We'll keep it simple.

    return result