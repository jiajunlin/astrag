"""Layer 1b — structural knowledge graph over parsed chunks.

Nodes are chunk ids (plus ``file:<path>`` container nodes); edges are:

* ``contains`` — file -> top-level def/class, class -> method
* ``calls``    — function/method -> function/method it invokes

Call resolution is *scope-aware* rather than global string matching:

1. ``self.m`` / ``cls.m``  -> method ``m`` on the caller's own class.
2. ``alias.f``             -> ``f`` in the module that ``alias`` was
                              imported as (module name matched against
                              file paths).
3. bare ``f``              -> same-file definition first; then a
                              ``from mod import f`` target; then a
                              cross-file match **only if unique** in the
                              whole repo. Ambiguous shared names (every
                              class having ``connect()``/``execute()``)
                              no longer create false edges.

``personalized_pagerank`` ranks chunks by structural importance in the
call graph, personalised by a preference vector (e.g. retrieval scores):

    PR(v) = (1 - d) * e_v + d * sum_{u in In(v)} PR(u) / Out(u)
"""
from __future__ import annotations

import builtins
from collections import defaultdict

from .parsing import CodeChunk

_BUILTINS = set(dir(builtins))


def _module_names(rel_path: str) -> list[str]:
    """'a/b/utils.py' -> ['a.b.utils', 'b.utils', 'utils'] (suffix chain)."""
    stem = rel_path.rsplit(".", 1)[0].replace("/", ".").replace("\\", ".")
    parts = stem.split(".")
    return [".".join(parts[i:]) for i in range(len(parts))]


class CodeGraph:
    def __init__(self) -> None:
        self.chunk_ids: set[str] = set()
        self.files: set[str] = set()
        self.edges: list[tuple[str, str, str]] = []
        self._out: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._in: dict[str, list[tuple[str, str]]] = defaultdict(list)

    # ---- construction ----
    @classmethod
    def from_chunks(cls, chunks: list[CodeChunk]) -> "CodeGraph":
        g = cls()
        by_name: dict[str, list[CodeChunk]] = defaultdict(list)
        by_file_name: dict[tuple[str, str], list[CodeChunk]] = defaultdict(list)
        by_file_qual: dict[tuple[str, str], CodeChunk] = {}
        module_files: dict[str, str] = {}          # module name -> rel_path
        for c in chunks:
            g.chunk_ids.add(c.chunk_id)
            g.files.add(c.file)
            by_name[c.name].append(c)
            by_file_name[(c.file, c.name)].append(c)
            by_file_qual[(c.file, c.qualname)] = c
            for m in _module_names(c.file):
                module_files.setdefault(m, c.file)

        def add_targets(src: CodeChunk, targets: list[CodeChunk]) -> None:
            for t in targets:
                if t.chunk_id != src.chunk_id:
                    g._add(src.chunk_id, "calls", t.chunk_id)

        for c in chunks:
            container = f"{c.file}::{c.parent}" if c.parent else f"file:{c.file}"
            g._add(container, "contains", c.chunk_id)
            if c.kind == "class":
                continue

            # from-import map: exposed name -> module that provides it
            from_imports: dict[str, str] = {}
            import_aliases: set[str] = set()
            for imp in c.imports:
                if "." in imp:
                    mod, _, name = imp.rpartition(".")
                    from_imports[name] = mod
                import_aliases.add(imp.split(".")[0])
                import_aliases.add(imp)

            for ref in c.calls:
                if "." in ref:
                    prefix, name = ref.split(".", 1)
                    if name in _BUILTINS and prefix not in ("self",):
                        continue
                    if prefix == "self" and c.parent:
                        # 1. method on the caller's own class
                        t = by_file_qual.get((c.file, f"{c.parent}.{name}"))
                        if t:
                            add_targets(c, [t])
                        continue
                    # 2. alias.f -> function f in module `prefix`
                    target_file = module_files.get(prefix)
                    if target_file is None and prefix in import_aliases:
                        for m, f in module_files.items():
                            if m.endswith(prefix):
                                target_file = f
                                break
                    if target_file:
                        add_targets(c, by_file_name.get((target_file, name), []))
                    continue

                name = ref
                if name in _BUILTINS:
                    continue
                # 3a. same-file definition
                local = by_file_name.get((c.file, name))
                if local:
                    add_targets(c, local)
                    continue
                # 3b. explicit `from mod import name`
                mod = from_imports.get(name)
                if mod:
                    for m in (mod, mod.split(".")[-1]):
                        f = module_files.get(m)
                        if f:
                            add_targets(c, by_file_name.get((f, name), []))
                            break
                    if mod:
                        continue
                # 3c. cross-file only when globally unambiguous
                candidates = by_name.get(name, [])
                if len(candidates) == 1:
                    add_targets(c, candidates)
        return g

    def _add(self, src: str, kind: str, dst: str) -> None:
        self.edges.append((src, kind, dst))
        self._out[src].append((kind, dst))
        self._in[dst].append((kind, src))

    # ---- queries ----
    def callees_of(self, chunk_id: str) -> list[str]:
        return sorted({d for k, d in self._out.get(chunk_id, []) if k == "calls"})

    def callers_of(self, chunk_id: str) -> list[str]:
        return sorted({s for k, s in self._in.get(chunk_id, []) if k == "calls"})

    def neighbors(self, chunk_id: str, kinds: tuple = ("calls",)) -> list[str]:
        """Undirected 1-hop neighbourhood over the given edge kind(s).

        Defaults to the static call graph only (``calls``) — this is what
        retrieval's graph-hop expansion uses, and it deliberately excludes
        ``contains`` edges: a class and its own methods, or a file and its
        top-level definitions, are structurally adjacent but not a call
        relationship, so including them here would leak score to unrelated
        siblings just for sitting in the same file/class. Pass e.g.
        ``("calls", "api_calls", "ffi_calls", "rpc_calls")`` to also expand
        across cross-service edges from ``crosslang.py``.
        """
        out: set[str] = set()
        for k, d in self._out.get(chunk_id, []):
            if k in kinds:
                out.add(d)
        for k, s in self._in.get(chunk_id, []):
            if k in kinds:
                out.add(s)
        return sorted(out)

    def _out_of_kind(self, chunk_id: str, kind: str) -> list[str]:
        return sorted({d for k, d in self._out.get(chunk_id, []) if k == kind})

    def _in_of_kind(self, chunk_id: str, kind: str) -> list[str]:
        return sorted({s for k, s in self._in.get(chunk_id, []) if k == kind})

    def api_callees_of(self, chunk_id: str) -> list[str]:
        """Backend route handlers this chunk calls over HTTP (see
        ``crosslang.py``); empty unless ``trace_api_calls`` was enabled."""
        return self._out_of_kind(chunk_id, "api_calls")

    def api_callers_of(self, chunk_id: str) -> list[str]:
        """Frontend call sites that reach this chunk over HTTP."""
        return self._in_of_kind(chunk_id, "api_calls")

    def ffi_callees_of(self, chunk_id: str) -> list[str]:
        """Foreign-function targets this chunk calls across a language
        boundary (Rust/C ``extern "C"``, Python ``ctypes``/``cffi``)."""
        return self._out_of_kind(chunk_id, "ffi_calls")

    def ffi_callers_of(self, chunk_id: str) -> list[str]:
        return self._in_of_kind(chunk_id, "ffi_calls")

    def rpc_callees_of(self, chunk_id: str) -> list[str]:
        """RPC service methods this chunk calls (gRPC client stub calls
        matched to servicer implementations by service.method name)."""
        return self._out_of_kind(chunk_id, "rpc_calls")

    def rpc_callers_of(self, chunk_id: str) -> list[str]:
        return self._in_of_kind(chunk_id, "rpc_calls")

    def cross_service_neighbors(self, chunk_id: str) -> list[str]:
        """Union of api_calls/ffi_calls/rpc_calls neighbours — the unified
        cross-language meta-graph edges from ``crosslang.py``."""
        return sorted(set(self.api_callees_of(chunk_id)) |
                      set(self.api_callers_of(chunk_id)) |
                      set(self.ffi_callees_of(chunk_id)) |
                      set(self.ffi_callers_of(chunk_id)) |
                      set(self.rpc_callees_of(chunk_id)) |
                      set(self.rpc_callers_of(chunk_id)))

    # ---- personalized pagerank ----
    def personalized_pagerank(self, preference: dict[str, float],
                              damping: float = 0.85, max_iter: int = 50,
                              tol: float = 1e-9) -> dict[str, float]:
        """PPR over the ``calls`` graph, rooted at *preference* (e.g. the
        stage-1/2 retrieval scores). Importance flows along call edges, so
        widely-used utilities reachable from the query hits rank up."""
        total = sum(v for v in preference.values() if v > 0)
        if total <= 0:
            return {}
        e = {n: max(v, 0.0) / total for n, v in preference.items()
             if n in self.chunk_ids and v > 0}
        if not e:
            return {}

        out_deg = {n: len(set(self.callees_of(n))) for n in self.chunk_ids}
        pr = dict(e)
        for _ in range(max_iter):
            nxt = {n: (1 - damping) * e.get(n, 0.0) for n in pr}
            dangling = 0.0
            for u, score in pr.items():
                deg = out_deg.get(u, 0)
                if deg == 0:
                    dangling += score
                    continue
                share = damping * score / deg
                for v in self.callees_of(u):
                    nxt[v] = nxt.get(v, 0.0) + share
            if dangling:                       # dangling mass -> preference
                for n, ev in e.items():
                    nxt[n] = nxt.get(n, 0.0) + damping * dangling * ev
            delta = sum(abs(nxt.get(n, 0.0) - pr.get(n, 0.0))
                        for n in set(nxt) | set(pr))
            pr = nxt
            if delta < tol:
                break
        return pr

    def stats(self) -> dict:
        calls = sum(1 for _, k, _ in self.edges if k == "calls")
        return {
            "files": len(self.files),
            "chunks": len(self.chunk_ids),
            "contains_edges": len(self.edges) - calls,
            "call_edges": calls,
        }

    def to_json(self) -> dict:
        nodes = sorted(self.chunk_ids | {f"file:{f}" for f in self.files})
        return {"nodes": nodes, "edges": [list(e) for e in self.edges]}