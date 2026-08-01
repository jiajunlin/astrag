"""Layer 2 — two-stage AST RAG.

Stage 1 (coarse):  rank *files* against the query.
Stage 2 (fine):    within those files, rank chunk **signature cards**
                   (signature + docstring summary + location — never
                   full bodies).

Exact implementations are pulled lazily through the tool layer
(``CodebaseTools.get_function_body``), which is how the context window
stays small.

The default scorer is a dependency-free Okapi BM25 over code-aware
tokens (identifiers split on camelCase/snake_case), so everything works
offline and deterministically. To use dense embeddings instead, pass
``encode_fn=lambda texts: [...vectors...]`` (e.g. a sentence-transformers
model or a hosted embedding API) — both stages switch to cosine search.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .parsing import CodeChunk, code_tokens


# --------------------------------------------------------------------------
# Scorers
# --------------------------------------------------------------------------

class BM25Index:
    """Small, dependency-free Okapi BM25 index (the offline default)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._postings: dict[str, dict[int, int]] = defaultdict(dict)
        self._doc_len: list[int] = []
        self._avgdl = 1.0
        self._idf: dict[str, float] = {}

    def fit(self, docs: list[list[str]]) -> "BM25Index":
        for i, toks in enumerate(docs):
            self._doc_len.append(max(1, len(toks)))
            for term, tf in Counter(toks).items():
                self._postings[term][i] = tf
        n = len(docs) or 1
        self._avgdl = sum(self._doc_len) / max(1, len(self._doc_len))
        for term, posting in self._postings.items():
            df = len(posting)
            self._idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        return self

    def scores(self, query_tokens: list[str]) -> dict[int, float]:
        acc: dict[int, float] = defaultdict(float)
        for term in query_tokens:
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf[term]
            for doc, tf in posting.items():
                dl = self._doc_len[doc]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                acc[doc] += idf * tf * (self.k1 + 1) / denom
        return acc

    def top(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        s = self.scores(query_tokens)
        return sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


class DenseIndex:
    """Cosine index over vectors from a user-supplied ``encode_fn``."""

    def __init__(self, encode_fn) -> None:
        self.encode_fn = encode_fn
        self._vecs: list[list[float]] = []

    @staticmethod
    def _norm(v):
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def fit(self, texts: list[str]) -> "DenseIndex":
        self._vecs = [self._norm(v) for v in self.encode_fn(texts)]
        return self

    def top(self, query_text: str, k: int) -> list[tuple[int, float]]:
        q = self._norm(self.encode_fn([query_text])[0])
        scored = [(i, sum(a * b for a, b in zip(q, v)))
                  for i, v in enumerate(self._vecs)]
        return sorted(scored, key=lambda kv: (-kv[1], kv[0]))[:k]


# --------------------------------------------------------------------------
# Retrieval results
# --------------------------------------------------------------------------

@dataclass
class RetrievedCard:
    chunk_id: str
    score: float
    kind: str
    file: str
    start_line: int
    end_line: int
    signature: str
    doc_summary: str

    def render(self) -> str:
        out = [f"[{self.chunk_id}]  ({self.kind}, score {self.score:.2f})",
               f"    {self.signature}"]
        if self.doc_summary:
            out.append(f'    """{self.doc_summary}"""')
        out.append(f"    @ {self.file}:{self.start_line}-{self.end_line}")
        return "\n".join(out)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    query: str
    files: list[tuple[str, float]]   # stage 1
    cards: list[RetrievedCard]       # stage 2


# --------------------------------------------------------------------------
# The two-stage retriever
# --------------------------------------------------------------------------

class TwoStageRetriever:
    """Hybrid two-stage retriever.

    Stage 2 fuses multiple rankers with **Reciprocal Rank Fusion**:

        RRF(d) = sum_m 1 / (k + rank_m(d)),  k = 60

    Rankers: a *symbol* BM25 (names + signatures — exact identifier
    matching), a *document* BM25 (full search text), and, when
    ``encode_fn`` is given, a dense cosine index. Fused candidates are
    then optionally expanded 1 hop along the call graph (``graph=``)
    and re-scored by a cross-encoder-style ``rerank_fn`` on the top
    candidates.

    ``rerank_fn(query, texts) -> list[float]`` is the hook for e.g.
    bge-reranker / CodeBERT rerankers; it sees ``(query, signature +
    doc)`` pairs for the fused top-``rerank_top`` chunks.
    """

    RRF_K = 60

    def __init__(self, chunks: list[CodeChunk], encode_fn=None,
                 graph=None, rerank_fn=None, rerank_top: int = 20,
                 expand_hops: int = 1, expand_decay: float = 0.35, graph_boost: float = 0.0) -> None:
        self.chunks = list(chunks)
        self.graph_boost = graph_boost
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.files = sorted({c.file for c in self.chunks})
        self.graph = graph
        self.rerank_fn = rerank_fn
        self.rerank_top = rerank_top
        self.expand_hops = expand_hops
        self.expand_decay = expand_decay
        
        by_file: dict[str, list[str]] = defaultdict(list)
        by_file_sym: dict[str, list[str]] = defaultdict(list)
        for c in self.chunks:
            by_file[c.file].append(c.search_text())
            by_file_sym[c.file].append(self._symbol_text(c))
        file_texts = ["\n".join(by_file[f]) for f in self.files]
        file_syms = ["\n".join(by_file_sym[f]) for f in self.files]
        chunk_texts = [c.search_text() for c in self.chunks]
        chunk_syms = [self._symbol_text(c) for c in self.chunks]

        self._file_doc = BM25Index().fit([code_tokens(t) for t in file_texts])
        self._file_sym = BM25Index().fit([code_tokens(t) for t in file_syms])
        self._chunk_doc = BM25Index().fit([code_tokens(t) for t in chunk_texts])
        self._chunk_sym = BM25Index().fit([code_tokens(t) for t in chunk_syms])
        self._file_dense = self._chunk_dense = None
        if encode_fn is not None:
            self._file_dense = DenseIndex(encode_fn).fit(file_texts)
            self._chunk_dense = DenseIndex(encode_fn).fit(chunk_texts)

    # ---- dynamic weight computation ----
    def _compute_dynamic_weights(self, query: str, fused_scores: dict[int, float]) -> tuple[float, float]:
        """Return (graph_boost, ppr_weight) tuned for this query.

        Uses query specificity (code-identifier density) and score spread
        to adjust the importance of graph centrality.
        """
        # 1. Query specificity: fraction of tokens that are code identifiers
        tokens = code_tokens(query)
        # Build a set of all known names from the corpus
        all_names = set()
        for c in self.chunks:
            all_names.update(code_tokens(c.name))
            all_names.update(code_tokens(c.qualname))
        code_ratio = sum(1 for t in tokens if t in all_names) / max(1, len(tokens))
        # code_ratio ~ 0 for natural language, ~1 for code-heavy queries
        # Map to graph_boost: lower for high code_ratio (0.1), higher for low code_ratio (0.4)
        graph_boost = 0.1 + (0.4 - 0.1) * (1 - code_ratio)

        # 2. Score spread: coefficient of variation of the top 10 fused scores
        scores = list(fused_scores.values())
        if len(scores) > 1:
            top = sorted(scores, reverse=True)[:10]
            mean = sum(top) / len(top)
            std = (sum((x - mean) ** 2 for x in top) / len(top)) ** 0.5
            cv = std / max(mean, 1e-6)
            # cv high => scores spread out => retrieval confident => lower boost
            # cv low => scores close => uncertain => higher boost
            cv_clipped = max(0.1, min(1.0, cv))
            adjustment = 0.8 + 0.4 * (1 - cv_clipped)  # 0.8 if cv=1, 1.2 if cv=0.1
            graph_boost *= adjustment
        # Clamp to [0.05, 0.5]
        graph_boost = max(0.05, min(0.5, graph_boost))

        # ppr_weight: we set it slightly lower than graph_boost (0.8x) to avoid over‑blending
        ppr_weight = graph_boost * 0.8
        return graph_boost, ppr_weight

    @staticmethod
    def _symbol_text(c: CodeChunk) -> str:
        return " ".join([c.name, c.qualname.replace(".", " "),
                         c.signature or "", c.kind, c.file, c.language])

    # ---- reciprocal rank fusion ----
    @classmethod
    def rrf_fuse(cls, rankings: list[list[int]]) -> dict[int, float]:
        """Fuse rank lists (best first) without score normalisation."""
        fused: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, doc in enumerate(ranking, start=1):
                fused[doc] += 1.0 / (cls.RRF_K + rank)
        return dict(fused)

    def _fused(self, query: str, doc_ix, sym_ix, dense_ix, n: int) -> dict[int, float]:
        toks = code_tokens(query)
        rankings = [[i for i, _ in doc_ix.top(toks, n)],
                    [i for i, _ in sym_ix.top(toks, n)]]
        if dense_ix is not None:
            rankings.append([i for i, _ in dense_ix.top(query, n)])
        return self.rrf_fuse(rankings)

    # ---- stage 1: coarse (files) ----
    def stage1_files(self, query: str, k: int = 4) -> list[tuple[str, float]]:
        fused = self._fused(query, self._file_doc, self._file_sym,
                            self._file_dense, len(self.files))
        top = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.files[i], round(100 * s, 3)) for i, s in top]

    # ---- stage 2: fine (signature cards, never bodies) ----
    def stage2_cards(self, query: str, files=None, k: int = 10,
                     kinds=None, graph_boost: float | None = None,
                     ppr_weight: float | None = None) -> list[RetrievedCard]:
        fused = self._fused(query, self._chunk_doc, self._chunk_sym,
                            self._chunk_dense, len(self.chunks))

        # Compute dynamic weights if either is not provided
        if graph_boost is None or ppr_weight is None:
            gb, pw = self._compute_dynamic_weights(query, fused)
            if graph_boost is None:
                graph_boost = gb
            if ppr_weight is None:
                ppr_weight = pw

        # Apply graph expansion and reranking (these don't use graph_boost)
        fused = self._expand_graph(fused)
        fused = self._rerank(query, fused)
        # Use the (possibly dynamic) graph_boost
        fused = self._boost_with_graph(fused, graph_boost)
        scores = {self.chunks[i].chunk_id: s for i, s in fused.items()}
        scores = self._expand_graph(scores)
        scores = self._rerank(query, scores)

        allowed = set(files) if files else None
        out: list[RetrievedCard] = []
        for cid, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            c = self.by_id[cid]
            if allowed is not None and c.file not in allowed:
                continue
            if kinds and c.kind not in kinds:
                continue
            out.append(RetrievedCard(
                chunk_id=c.chunk_id, score=round(100 * score, 3), kind=c.kind,
                file=c.file, start_line=c.start_line, end_line=c.end_line,
                signature=c.signature, doc_summary=c.doc_summary(),
            ))
            if len(out) >= k:
                break
        return out

    # ---- graph neighbourhood expansion ----
    def _expand_graph(self, scores: dict[str, float]) -> dict[str, float]:
        if self.graph is None or self.expand_hops <= 0:
            return scores
        out = dict(scores)
        frontier = dict(scores)
        for _ in range(self.expand_hops):
            nxt: dict[str, float] = {}
            for cid, s in frontier.items():
                if s <= 0:
                    continue
                for nb in self.graph.neighbors(cid):
                    if nb not in self.by_id:
                        continue
                    bonus = self.expand_decay * s
                    if bonus > nxt.get(nb, 0.0):
                        nxt[nb] = bonus
            for nb, bonus in nxt.items():
                out[nb] = out.get(nb, 0.0) + bonus
            frontier = nxt
        return out

    # ---- cross-encoder style reranking ----
    def _rerank(self, query: str, scores: dict[str, float]) -> dict[str, float]:
        if self.rerank_fn is None or not scores:
            return scores
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:self.rerank_top]
        texts = []
        for cid, _ in ranked:
            c = self.by_id[cid]
            texts.append(f"{c.signature or c.name}\n{c.doc_summary() or ''}")
        try:
            new = self.rerank_fn(query, texts)
        except Exception:
            return scores
        out = dict(scores)
        base = max(scores.values()) or 1.0
        lo, hi = min(new), max(new)
        span = (hi - lo) or 1.0
        for (cid, _), ns in zip(ranked, new):
            out[cid] = base * (1.0 + (ns - lo) / span)   # reranked block on top
        return out

    # ---- graph boost (structural centrality) ----
    def _boost_with_graph(self, scores: dict[str, float], graph_boost: float | None = None) -> dict[str, float]:
        boost = graph_boost if graph_boost is not None else self.graph_boost
        if self.graph is None or boost <= 0:
            return scores
        # Compute uniform PageRank for all nodes
        uniform = {cid: 1.0 for cid in self.graph.chunk_ids}
        ppr = self.graph.personalized_pagerank(uniform)
        max_ppr = max(ppr.values()) if ppr else 1.0
        # Blend: new_score = (1 - boost)*old + boost * normalized_ppr
        out = {}
        for cid, s in scores.items():
            ppr_score = ppr.get(cid, 0.0) / max_ppr
            out[cid] = (1 - boost) * s + boost * ppr_score
        return out

    def retrieve(self, query: str, k_files: int = 4,
                 k_chunks: int = 10, graph_boost: float | None = None,
                 ppr_weight: float | None = None) -> RetrievalResult:
        files = self.stage1_files(query, k=k_files)
        cards = self.stage2_cards(query, files=[f for f, _ in files], k=k_chunks,
                                  graph_boost=graph_boost, ppr_weight=ppr_weight)
        return RetrievalResult(query=query, files=files, cards=cards)

    # ---- replication check ----
    def find_existing(self, description: str, k: int = 5) -> list[RetrievedCard]:
        """Existing functions/methods that may already implement *description*.

        Prompt pattern this enables: "Before implementing X, check the
        retrieved AST graph for any existing function that performs X.
        If found, call that function instead of rewriting the logic."
        """
        return self.stage2_cards(description, files=None, k=k,
                                 kinds=("function", "method"))