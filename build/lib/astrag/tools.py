"""Fine-grained tool-call surface.

Stage 2 hands the model *signatures only*. When the model decides it
needs exact logic, it issues one of these tool calls instead of having
whole files pushed into context. ``dispatch`` routes a
``(name, arguments)`` pair — the shape produced by LLM tool-use APIs —
to the right method, and ``anthropic_tool_schemas`` exports JSON-schema
definitions ready for the Anthropic Messages API ``tools`` parameter.
"""
from __future__ import annotations

from astrag.parsing import code_tokens


class CodebaseTools:
    def __init__(self, memory) -> None:
        self.memory = memory

    # ---- tools ----
    def search_code(self, query: str, k: int = 10) -> list[dict]:
        """Semantic search; returns signature cards, never bodies."""
        cards = self.memory.retriever.stage2_cards(query, files=None, k=k)
        return [c.to_dict() for c in cards]

    def find_existing_implementations(self, description: str,
                                      k: int = 5) -> list[dict]:
        """Replication check: run *before* writing any new function."""
        cards = self.memory.retriever.find_existing(description, k=k)
        top = cards[0].score if cards else 0.0
        return [dict(c.to_dict(),
                     strong_match=bool(top and c.score >= 0.5 * top))
                for c in cards]

    def get_function_body(self, chunk_id: str) -> dict:
        """Fetch the exact implementation of one chunk, on demand."""
        c = self.memory.chunk(chunk_id)
        return {"chunk_id": c.chunk_id, "file": c.file,
                "lines": [c.start_line, c.end_line], "source": c.source}

    def get_callees(self, chunk_id: str) -> dict:
        return {"chunk_id": chunk_id,
                "calls": self.memory.graph.callees_of(chunk_id)}

    def get_callers(self, chunk_id: str) -> dict:
        return {"chunk_id": chunk_id,
                "called_by": self.memory.graph.callers_of(chunk_id)}

    def get_graph_stats(self) -> dict:
        """Live counts of the dependency graph: files, chunks, edges."""
        return self.memory.graph.stats()

    def get_index_report(self) -> dict:
        """What indexing did and didn't cover, and why: files considered
        vs. actually indexed, plus a breakdown of every skip reason
        (gitignored, a lockfile/generated file, binary, oversize,
        unreadable, or a parse error). Call this when results seem to be
        missing files, instead of guessing."""
        return self.memory.index_report()

    def get_central_functions(self, k: int = 10) -> list[dict]:
        """The most structurally central chunks (Personalized PageRank,
        uniform preference over every chunk) — the utilities most of the
        codebase ends up depending on, independent of any query."""
        graph = self.memory.graph
        uniform = {cid: 1.0 for cid in graph.chunk_ids}
        scores = graph.personalized_pagerank(uniform)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out = []
        for chunk_id, score in ranked:
            try:
                c = self.memory.chunk(chunk_id)
            except Exception:
                continue
            out.append({"chunk_id": chunk_id, "score": round(score, 6),
                        "signature": c.signature, "file": c.file})
        return out

    def get_dependency_graph(self, chunk_id: str, depth: int = 2) -> dict:
        """Live BFS subgraph around one chunk, out to ``depth`` hops both
        directions, over the full unified meta-graph — static ``calls``
        plus any cross-service edges from ``crosslang.py`` (``api_calls``,
        ``ffi_calls``, ``rpc_calls``) — for inspecting the blast radius of
        a change without re-running the whole indexer."""
        graph = self.memory.graph
        edge_getters = [
            ("calls", graph.callees_of, graph.callers_of),
            ("api_calls", graph.api_callees_of, graph.api_callers_of),
            ("ffi_calls", graph.ffi_callees_of, graph.ffi_callers_of),
            ("rpc_calls", graph.rpc_callees_of, graph.rpc_callers_of),
        ]
        nodes = {chunk_id}
        edges: list[tuple[str, str, str]] = []
        frontier = {chunk_id}
        for _ in range(max(0, depth)):
            nxt: set[str] = set()
            for n in frontier:
                for kind, callees_of, callers_of in edge_getters:
                    for d in callees_of(n):
                        edges.append((n, kind, d)); nxt.add(d)
                    for s in callers_of(n):
                        edges.append((s, kind, n)); nxt.add(s)
            nxt -= nodes
            nodes |= nxt
            frontier = nxt
            if not frontier:
                break
        seen = set()
        dedup = []
        for e in edges:
            if e not in seen:
                seen.add(e); dedup.append(list(e))
        return {"root": chunk_id, "nodes": sorted(nodes), "edges": dedup}

    def get_api_trace(self, chunk_id: str) -> dict:
        """Cross-language HTTP callers/callees for a chunk (see
        ``crosslang.py``) — empty unless the index was built with
        ``trace_api_calls=True``."""
        graph = self.memory.graph
        return {"chunk_id": chunk_id,
                "calls_backend": graph.api_callees_of(chunk_id),
                "called_from_frontend": graph.api_callers_of(chunk_id)}

    def build_context(self, query: str, token_budget: int = 1500) -> dict:
        """Assemble a full compressed context block for a task in one call."""
        ctx = self.memory.build_context(query, token_budget=token_budget)
        return {"context": ctx.text, "stats": ctx.stats_line()}

    def find_chunks_by_name(self, name: str) -> list[dict]:
        """Find all chunks whose name or qualname contains the given string."""
        results = []
        name_lower = name.lower()
        for c in self.memory.chunks:
            if name_lower in c.name.lower() or name_lower in c.qualname.lower():
                results.append({
                    "chunk_id": c.chunk_id,
                    "kind": c.kind,
                    "file": c.file,
                    "name": c.name,
                    "qualname": c.qualname,
                    "signature": c.signature,
                    "language": c.language,
                })
        return results

    def get_slice(self, chunk_id: str, query: str = "", max_depth: int = 1) -> dict:
        """Return the SDG-lite slice for a chunk, including callees as
        needed (see ``slicing.interprocedural_slice``)."""
        from .compression import _render_kept
        from .slicing import interprocedural_slice
        chunk = self.memory.chunk(chunk_id)
        anchors = set(code_tokens(query)) if query else set()
        slice_data = interprocedural_slice(
            seed_chunk=chunk,
            graph=self.memory.graph,
            chunk_by_id=self.memory.retriever.by_id,
            max_depth=max_depth,
            anchors=anchors
        )
        result = []
        for cid, (c, kept_lines) in slice_data.items():
            result.append({
                "chunk_id": cid,
                "file": c.file,
                "lines": sorted(kept_lines) if kept_lines else [],
                "source": (_render_kept(c.source, kept_lines) if kept_lines
                          else c.source),
            })
        return {"slice": result}

    def predict_impact(self, chunk_id: str, new_code: str = None) -> dict:
        """Analyze impact of changing a chunk: its direct and transitive
        callers/callees over the static call graph, plus any cross-service
        callers (frontend/RPC-client/FFI-caller) that would also need to
        change — those are often the ones missed when eyeballing a diff,
        since they live in a different file, language, or process."""
        graph = self.memory.graph
        callees = graph.callees_of(chunk_id)
        callers = graph.callers_of(chunk_id)
        cross_service = graph.cross_service_neighbors(chunk_id)
        return {
            "chunk_id": chunk_id,
            "callers": callers,
            "callees": callees,
            "transitive_callers": self._transitive_closure(callers, graph, direction="in"),
            "transitive_callees": self._transitive_closure(callees, graph, direction="out"),
            "cross_service_dependents": cross_service,
        }

    def _transitive_closure(self, seeds, graph, direction="out"):
        visited = set(seeds)
        frontier = set(seeds)
        while frontier:
            nxt = set()
            for cid in frontier:
                if direction == "out":
                    neighbors = graph.callees_of(cid)
                else:
                    neighbors = graph.callers_of(cid)
                for nb in neighbors:
                    if nb not in visited:
                        visited.add(nb)
                        nxt.add(nb)
            frontier = nxt
        return list(visited)

    # ---- plumbing ----
    def dispatch(self, name: str, arguments: dict | None):
        """Route a tool-use block (name + input dict) to the right method."""
        if name.startswith("_") or not hasattr(self, name):
            raise ValueError(f"unknown tool {name!r}")
        return getattr(self, name)(**(arguments or {}))

    @staticmethod
    def anthropic_tool_schemas() -> list[dict]:
        def obj(props: dict, required: list[str]) -> dict:
            return {"type": "object", "properties": props, "required": required}

        s_str = {"type": "string"}
        s_int = {"type": "integer"}
        return [
            {"name": "search_code",
             "description": ("Semantic search over the indexed codebase. "
                             "Returns signature cards (chunk_id, signature, "
                             "docstring summary, file:lines) — no bodies."),
             "input_schema": obj({"query": s_str, "k": s_int}, ["query"])},
            {"name": "find_existing_implementations",
             "description": ("Replication check. Call this BEFORE writing any "
                             "new function: lists existing functions that may "
                             "already implement the described behaviour, so "
                             "they can be called instead of rewritten."),
             "input_schema": obj({"description": s_str, "k": s_int},
                                 ["description"])},
            {"name": "get_function_body",
             "description": ("Fetch the exact source of one function/method/"
                             "class by chunk_id (use ids returned by "
                             "search_code)."),
             "input_schema": obj({"chunk_id": s_str}, ["chunk_id"])},
            {"name": "get_callees",
             "description": "Functions this chunk calls (static call graph).",
             "input_schema": obj({"chunk_id": s_str}, ["chunk_id"])},
            {"name": "get_callers",
             "description": "Functions that call this chunk (static call graph).",
             "input_schema": obj({"chunk_id": s_str}, ["chunk_id"])},
            {"name": "get_graph_stats",
             "description": ("Live counts of the dependency graph: indexed "
                             "files, chunks, contains/call edges."),
             "input_schema": obj({}, [])},
            {"name": "get_index_report",
             "description": ("What indexing skipped and why (gitignored, "
                             "a lockfile, binary, oversize, unreadable, "
                             "parse error) — use when results seem to be "
                             "missing files."),
             "input_schema": obj({}, [])},
            {"name": "get_central_functions",
             "description": ("The most structurally central chunks in the "
                             "codebase (Personalized PageRank over the call "
                             "graph) — widely-depended-on utilities, "
                             "independent of any specific query."),
             "input_schema": obj({"k": s_int}, [])},
            {"name": "get_dependency_graph",
             "description": ("Live BFS subgraph of call/API edges around one "
                             "chunk out to a given depth — the blast radius "
                             "of changing it, for real-time inspection."),
             "input_schema": obj({"chunk_id": s_str, "depth": s_int},
                                 ["chunk_id"])},
            {"name": "get_api_trace",
             "description": ("Cross-language HTTP callers/callees for a "
                             "chunk — e.g. which frontend fetch() calls "
                             "reach this backend route handler, or which "
                             "backend routes a frontend call reaches. Empty "
                             "unless the index was built with "
                             "trace_api_calls=True."),
             "input_schema": obj({"chunk_id": s_str}, ["chunk_id"])},
            {"name": "build_context",
             "description": ("Build a complete, token-budgeted context block "
                             "for a coding task: relevant files, signature "
                             "cards, a replication check, and compressed "
                             "implementations. Ideal first call for any task."),
             "input_schema": obj({"query": s_str, "token_budget": s_int},
                                 ["query"])},
            {"name": "find_chunks_by_name",
             "description": "Search for chunks whose name or qualname contains a substring.",
             "input_schema": obj({"name": s_str}, ["name"])},
            {"name": "get_slice",
             "description": ("Return the program slice (SDG) for a chunk, "
                             "including callees up to a given depth."),
             "input_schema": obj({"chunk_id": s_str, "query": s_str, "max_depth": s_int},
                                 ["chunk_id"])},
            {"name": "predict_impact",
             "description": ("Analyze the impact of changing a chunk: "
                             "list callers, callees, and transitive dependencies."),
             "input_schema": obj({"chunk_id": s_str}, ["chunk_id"])},
        ]