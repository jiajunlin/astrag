"""Orchestration: index -> retrieve -> compress -> assembled prompt context.

``CodebaseMemory`` is the one object most callers need:

    mem = CodebaseMemory().index_repo("path/to/repo")
    ctx = mem.build_context("add retry to the payments client",
                            token_budget=1500)
    # ctx.text is ready to prepend to an LLM prompt;
    # CodebaseTools(mem) provides the on-demand body fetches.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from .compression import (CompressionResult, SemanticAnchorCompressor,
                          knapsack_pack, make_surprisal_score_fn,
                          sliced_text)
from .graph import CodeGraph
from .langs import SUPPORTED_EXTENSIONS, heuristic_parser_for
from .universal import (BINARY_EXTENSIONS, MAX_FILE_BYTES, looks_binary,
                        parser_for)
from .parsing import (CodeChunk, PythonStdlibParser, TreeSitterParser,
                      approx_tokens, iter_source_files)
from .retrieval import RetrievalResult, RetrievedCard, TwoStageRetriever

REPLICATION_CHECK_INSTRUCTION = (
    "Before implementing anything new, check the API surface above for an "
    "existing function that already performs the task; if one exists, call "
    "it instead of rewriting the logic. To inspect exact logic, request it "
    'with the tool call get_function_body(chunk_id="<id in brackets above>") '
    "rather than asking for whole files. Other tools: search_code, "
    "find_existing_implementations, get_callers, get_callees."
)


def default_encoder(model_name: str = "all-MiniLM-L6-v2"):
    """Create a sentence‑transformer encoder function."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return lambda texts: model.encode(texts, convert_to_numpy=True).tolist()
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed; "
            "install with: pip install sentence-transformers"
        )


@dataclass
class FetchedBody:
    chunk_id: str
    file: str
    start_line: int
    end_line: int
    compression: CompressionResult
    language: str = "python"
    level: str = "compressed"          # full | sliced | compressed | signature


@dataclass
class BuiltContext:
    query: str
    text: str
    files: list = field(default_factory=list)
    cards: list = field(default_factory=list)
    bodies: list = field(default_factory=list)
    budget: int = 0

    def stats_line(self) -> str:
        orig = sum(b.compression.original_tokens for b in self.bodies)
        comp = sum(b.compression.compressed_tokens for b in self.bodies)
        ratio = orig / comp if comp else 1.0
        return (f"context ≈{approx_tokens(self.text)} tok (budget {self.budget}) | "
                f"files={len(self.files)} cards={len(self.cards)} "
                f"bodies={len(self.bodies)} | body compression "
                f"{orig}→{comp} tok ({ratio:.1f}x)")


class CodebaseMemory:
    """End-to-end pipeline: Layer 1 parse+graph, Layer 2 retrieve, Layer 3 compress."""

    def __init__(self, encode_fn=None, dense_model: str | None = None,
                 compressor: SemanticAnchorCompressor | None = None,
                 rerank_fn=None, ppr_weight: float = 0.25,
                 graph_boost: float = 0.1,
                 use_surprisal: bool = True,
                 trace_api_calls: bool = False) -> None:
        if encode_fn is None and dense_model is not None:
            encode_fn = default_encoder(dense_model)
        self.encode_fn = encode_fn
        self.graph_boost = graph_boost
        self.compressor = compressor or SemanticAnchorCompressor()
        # a user-supplied score_fn is never overwritten by the auto one
        self.compressor._auto_score = self.compressor.score_fn is None
        self.rerank_fn = rerank_fn        # cross-encoder hook: (q, texts)->scores
        self.ppr_weight = ppr_weight      # blend of Personalized PageRank (fallback)
        self.use_surprisal = use_surprisal
        # heuristic frontend-call <-> backend-route edges (crosslang.py);
        # off by default since route-pattern scanning adds indexing cost
        self.trace_api_calls = trace_api_calls
        self.root: str | None = None
        self.chunks: list[CodeChunk] = []
        self.graph: CodeGraph | None = None
        self.retriever: TwoStageRetriever | None = None

    # ------------------------------------------------------------------
    # Layer 1 — parse the repo into structural chunks + graph
    # ------------------------------------------------------------------
    def index_repo(self, root, prefer_tree_sitter: bool = False,
                  cache_path: str | None = None) -> "CodebaseMemory":
        """Parse every supported source file under ``root``.

        Parser choice per file: Python -> stdlib ``ast`` (richest);
        other languages -> tree-sitter when installed (exact grammars),
        else the built-in heuristic parsers (C, C++, C#, Java, JS/JSX,
        TS/TSX, Go, Rust, PHP, Swift, Kotlin, HTML, CSS).

        ``cache_path``, if given, enables incremental re-indexing: an
        sqlite cache (``cache.py``) keyed on ``(mtime, size)`` with a
        content-hash fallback, so a re-run only re-parses files that
        actually changed since the cache was last written. Files removed
        from the repo are pruned from the cache automatically.
        """
        self.root = os.path.abspath(str(root))
        py = PythonStdlibParser()
        chunks: list[CodeChunk] = []
        cache = None
        if cache_path:
            from .cache import IndexCache
            cache = IndexCache(cache_path)
        try:
            for rel, full in iter_source_files(self.root, extensions=None):
                base = os.path.basename(rel)
                ext = os.path.splitext(base)[1].lower()
                if ext in BINARY_EXTENSIONS or base.endswith((".min.js",
                                                              ".min.css")):
                    continue
                try:
                    st = os.stat(full)
                    if st.st_size > MAX_FILE_BYTES:
                        continue
                except OSError:
                    continue

                _box: list[str] = []       # lazy-read memo, avoids double I/O

                def _read(full=full, box=_box) -> str:
                    if not box:
                        with open(full, encoding="utf-8",
                                  errors="replace") as fh:
                            box.append(fh.read())
                    return box[0]

                if cache is not None:
                    cached = cache.lookup(rel, st.st_mtime, st.st_size, _read)
                    if cached is not None:
                        chunks += cached
                        continue

                try:
                    source = _read()
                except OSError:
                    continue
                if looks_binary(source):
                    continue
                if ext == ".py" and not prefer_tree_sitter:
                    file_chunks = py.parse_file(rel, source)
                else:
                    lang = TreeSitterParser.EXT_LANG.get(ext)
                    ts = TreeSitterParser.for_language(lang) if lang else None
                    if ts is not None:
                        file_chunks = ts.parse_file(rel, source)
                    elif ext == ".py":
                        file_chunks = py.parse_file(rel, source)
                    else:
                        try:
                            file_chunks = parser_for(rel).parse_file(rel, source)
                        except Exception:
                            continue      # never let one odd file kill indexing
                chunks += file_chunks
                if cache is not None:
                    cache.store(rel, st.st_mtime, st.st_size, source,
                               file_chunks)
            if cache is not None:
                cache.prune()
        finally:
            if cache is not None:
                cache.close()
        self.chunks = chunks
        self._rebuild()
        return self

    def _rebuild(self) -> None:
        self.graph = CodeGraph.from_chunks(self.chunks)
        if self.trace_api_calls:
            from .crosslang import annotate_graph
            annotate_graph(self.graph, self.chunks)
        self.retriever = TwoStageRetriever(self.chunks,
                                           encode_fn=self.encode_fn,
                                           graph=self.graph,
                                           rerank_fn=self.rerank_fn,
                                           graph_boost=self.graph_boost)
        # corpus-surprisal line scorer (perplexity stand-in) — fitted on
        # the repo itself; skipped if the user plugged in a real score_fn
        if self.use_surprisal and getattr(self.compressor,
                                          "_auto_score", True):
            self.compressor.score_fn = make_surprisal_score_fn(
                [c.source for c in self.chunks])
            self.compressor._auto_score = True

    def stats(self) -> dict:
        kinds: dict[str, int] = {}
        for c in self.chunks:
            kinds[c.kind] = kinds.get(c.kind, 0) + 1
        out = {"files": len({c.file for c in self.chunks}),
               "chunks": len(self.chunks), **kinds}
        if self.graph:
            out["call_edges"] = self.graph.stats()["call_edges"]
        return out

    def chunk(self, chunk_id: str) -> CodeChunk:
        try:
            return self.retriever.by_id[chunk_id]
        except (KeyError, AttributeError):
            raise ValueError(
                f"unknown chunk_id {chunk_id!r}; call search_code first")

    # ------------------------------------------------------------------
    # Layer 2 — retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, k_files: int = 4,
                 k_chunks: int = 10, graph_boost: float | None = None,
                 ppr_weight: float | None = None) -> RetrievalResult:
        result = self.retriever.retrieve(query, k_files=k_files,
                                         k_chunks=k_chunks * 2,
                                         graph_boost=graph_boost,
                                         ppr_weight=ppr_weight)
        # if the caller didn't pin ppr_weight, use what stage2 resolved
        # (its own auto-tune or self.ppr_weight) instead of re-deriving
        # a second, disconnected default here
        effective_pw = ppr_weight if ppr_weight is not None else result.ppr_weight
        result.cards = self._blend_ppr(result.cards, effective_pw)[:k_chunks]
        return result

    def _blend_ppr(self, cards: list[RetrievedCard], ppr_weight: float | None = None) -> list[RetrievedCard]:
        """final = (1-w)·norm(retrieval) + w·norm(PPR rooted at the hits).

        PPR pushes structurally central chunks (widely-called utilities
        reachable from the query hits) above isolated text matches."""
        w = ppr_weight if ppr_weight is not None else self.ppr_weight
        if not cards or not self.graph or w <= 0:
            return cards
        pref = {c.chunk_id: c.score for c in cards}
        ppr = self.graph.personalized_pagerank(pref)
        if not ppr:
            return cards
        max_s = max(c.score for c in cards) or 1.0
        max_p = max(ppr.values()) or 1.0
        for c in cards:
            c.score = round(max_s * ((1 - w) * (c.score / max_s)
                                     + w * (ppr.get(c.chunk_id, 0.0) / max_p)), 3)
        return sorted(cards, key=lambda c: (-c.score, c.chunk_id))

    def replication_check(self, description: str, k: int = 5):
        return self.retriever.find_existing(description, k=k)

    # ------------------------------------------------------------------
    # Layer 3 + assembly — the compressed prompt context
    # ------------------------------------------------------------------
    def build_context(self, query: str, token_budget: int = 1800,
                      k_files: int = 4, k_chunks: int = 10,
                      fetch_bodies: int = 2,
                      graph_boost: float | None = None,
                      ppr_weight: float | None = None) -> BuiltContext:
        result = self.retrieve(query, k_files=k_files, k_chunks=k_chunks,
                               graph_boost=graph_boost, ppr_weight=ppr_weight)
        duplicates = self.replication_check(query, k=3)

        # measure the fixed sections first (headers, stage-1 list,
        # replication check, instructions), then split what's left
        overhead = approx_tokens(self._assemble(
            query, result.files, [], duplicates, [], token_budget))
        remaining = max(120, token_budget - overhead)
        card_budget = int(remaining * 0.40)

        # signature cards, trimmed to their budget share
        kept_cards: list[RetrievedCard] = []
        used = 0
        for card in result.cards:
            cost = approx_tokens(card.render())
            if kept_cards and used + cost > card_budget:
                break
            kept_cards.append(card)
            used += cost
        body_budget = max(80, remaining - used)

        # ---- multi-choice knapsack packing of implementations ----
        # each candidate body offers discrete levels — full / AST-sliced /
        # anchor-compressed / signature-only — and the DP picks the
        # value-maximal combination that fits body_budget exactly once.
        bodies: list[FetchedBody] = []
        top = [c for c in result.cards
               if c.kind in ("function", "method")][:max(0, fetch_bodies * 2)]
        if top and fetch_bodies > 0:
            variant_lists = []
            for c in top:
                chunk = self.chunk(c.chunk_id)
                rel = max(c.score, 0.01)
                src = chunk.source
                full_cost = approx_tokens(src)
                variants = [("full", full_cost, rel * 1.0, src)]
                sl = sliced_text(src, query, language=chunk.language,
                                 extra_anchors=(chunk.name,))
                if sl is not None and approx_tokens(sl) < full_cost:
                    variants.append(("sliced", approx_tokens(sl),
                                     rel * 0.8, sl))
                sac = self.compressor.compress(
                    src, query, budget_tokens=max(60, full_cost // 2),
                    extra_anchors=(chunk.name,), language=chunk.language)
                if sac.compressed_tokens < full_cost:
                    variants.append(("compressed", sac.compressed_tokens,
                                     rel * 0.65, sac.text))
                sig = (chunk.signature or chunk.name) + \
                    (f"\n    # {chunk.doc_summary()}" if chunk.doc_summary() else "")
                variants.append(("signature", approx_tokens(sig),
                                 rel * 0.3, sig))
                variant_lists.append(variants)

            picked = knapsack_pack(variant_lists, budget=body_budget)
            picked = picked[:max(0, fetch_bodies)] or (
                [(0, "signature", variant_lists[0][-1][3])] if variant_lists else [])
            for idx, level, text_payload in picked:
                c = top[idx]
                chunk = self.chunk(c.chunk_id)
                orig = approx_tokens(chunk.source)
                comp = approx_tokens(text_payload)
                res = CompressionResult(
                    text=text_payload, original_tokens=orig,
                    compressed_tokens=comp,
                    kept_lines=len(text_payload.splitlines()),
                    total_lines=len(chunk.source.splitlines()))
                bodies.append(FetchedBody(
                    c.chunk_id, c.file, c.start_line, c.end_line, res,
                    language=chunk.language, level=level))

        text = self._assemble(query, result.files, kept_cards,
                              duplicates, bodies, token_budget)
        return BuiltContext(query=query, text=text, files=result.files,
                            cards=kept_cards, bodies=bodies,
                            budget=token_budget)

    @staticmethod
    def _assemble(query, files, cards, duplicates, bodies, budget) -> str:
        parts = [f"# Codebase context (AST-RAG, budget ≈{budget} tokens)",
                 f"# Task: {query}", "",
                 "## Stage 1 — relevant files"]
        parts += [f"- {f}  (score {s:.2f})" for f, s in files]
        parts += ["", "## Stage 2 — API surface (signatures & docstrings only)"]
        parts += [c.render() for c in cards]
        if duplicates:
            parts += ["", "## Replication check — possible existing implementations"]
            parts += [f"- {c.signature}   [{c.chunk_id}]  (score {c.score:.2f})"
                      for c in duplicates]
        if bodies:
            parts += ["", "## Fetched implementations (semantic-anchor compressed)"]
            for b in bodies:
                r = b.compression
                parts.append(f"### {b.chunk_id}  [{b.level}]  "
                             f"({b.file}:{b.start_line}-{b.end_line}, "
                             f"{r.original_tokens}→{r.compressed_tokens} tok, "
                             f"{r.ratio:.1f}x)")
                parts.append(f"```{b.language}\n" + r.text + "\n```")
        parts += ["", "## Instructions", REPLICATION_CHECK_INSTRUCTION]
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        data = {"version": 1, "root": self.root,
                "trace_api_calls": self.trace_api_calls,
                "chunks": [asdict(c) for c in self.chunks]}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)

    @classmethod
    def load(cls, path: str, encode_fn=None, graph_boost: float = 0.1,
             ppr_weight: float = 0.25) -> "CodebaseMemory":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        mem = cls(encode_fn=encode_fn,
                  trace_api_calls=data.get("trace_api_calls", False),
                  graph_boost=graph_boost,
                  ppr_weight=ppr_weight)
        mem.root = data.get("root")
        mem.chunks = [CodeChunk(**c) for c in data["chunks"]]
        mem._rebuild()
        return mem