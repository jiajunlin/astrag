"""Layer 3 — semantic-anchor context compression (training-free).

This is an inference-time approximation of anchor-token compression
(SAC) and LongCodeZip-style budgeted selection — no learned weights, so
it runs anywhere:

1. Build an **anchor set**: query terms + symbol names that must survive
   (function name, parameters, callees).
2. Score every line by structural role (``def``/``return``/``raise``…),
   anchor overlap, and noise penalties (comments, docstring bodies,
   blank lines).
3. Greedily keep the highest-value lines under a token budget,
   preserving original order and inserting ``# … n line(s) elided …``
   markers where code was dropped.

The actual learned/perplexity-based ranking used by SAC / LongCodeZip
can be plugged in via ``score_fn(line) -> float`` (e.g. negative
per-line perplexity from a small local LM); it is simply added to the
heuristic score.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .parsing import approx_tokens, code_tokens
from .slicing import slice_lines

_DECL_KW = (r"def|class|function|fn|func|fun|struct|enum|union|interface|"
            r"trait|impl|namespace|module|record|object|extension|protocol|init")
_STRUCTURAL = [
    (re.compile(r"^\s*(?:export\s+|pub(?:\([^)]*\))?\s+|public\s+|private\s+|"
                r"protected\s+|static\s+|async\s+|abstract\s+|final\s+|"
                r"override\s+|declare\s+|default\s+|const\s+|open\s+)*"
                r"(?:" + _DECL_KW + r")\b"), 4.0),
    (re.compile(r"^\s*(return|yield|raise|throw|panic!?)\b|=>"), 2.5),
    (re.compile(r"^\s*(if|elif|else|for|foreach|while|do|switch|match|case|"
                r"try|except|catch|finally|with|guard|when|defer|select)\b"), 1.0),
    (re.compile(r"^\s*(@\w|\[[A-Z]|#\[)"), 1.0),      # decorators / attributes
    (re.compile(r"^\s*[})\];]*\s*$"), -0.8),          # bare closers: cheap noise
]
_FORCE_KEEP = re.compile(
    r"^\s*(?:export\s+|pub(?:\([^)]*\))?\s+|public\s+|private\s+|protected\s+|"
    r"internal\s+|static\s+|async\s+|abstract\s+|final\s+|sealed\s+|partial\s+|"
    r"virtual\s+|override\s+|declare\s+|default\s+|open\s+|data\s+|unsafe\s+|"
    r"const\s+)*(?:" + _DECL_KW + r")\s")
_COMMENT = re.compile(r"^\s*(#(?!\[)|//|/\*|\*(?:\s|/|$)|<!--)")
_DOC_COMMENT = re.compile(r"^\s*(///|//!|/\*\*)")
_TRIPLE = re.compile(r'"""|\'\'\'')
_ANCHOR_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "with",
    "is", "are", "it", "this", "that", "be", "as", "at", "by", "from",
    "add", "use", "make", "new", "my", "our",
}


@dataclass
class CompressionResult:
    text: str
    original_tokens: int
    compressed_tokens: int
    kept_lines: int = 0
    total_lines: int = 0

    @property
    def ratio(self) -> float:
        return self.original_tokens / max(1, self.compressed_tokens)


class SemanticAnchorCompressor:
    def __init__(self, score_fn=None, anchor_weight: float = 1.2,
                 comment_penalty: float = -1.2,
                 slice_bonus: float = 1.5) -> None:
        self.score_fn = score_fn
        self.anchor_weight = anchor_weight
        self.comment_penalty = comment_penalty
        self.slice_bonus = slice_bonus

    # ---- scoring ----
    def _line_scores(self, lines: list[str], anchors: set[str]) -> list[float]:
        scores: list[float] = []
        in_doc = False
        doc_first = False
        for line in lines:
            stripped = line.strip()
            s = 0.0
            if not stripped:
                scores.append(-10.0)
                continue
            for pattern, weight in _STRUCTURAL:
                if pattern.search(line):
                    s += weight
            if _DOC_COMMENT.match(line):
                s += 0.8
            elif _COMMENT.match(line):
                s += self.comment_penalty
            # docstring handling: keep the summary line, elide the body
            delims = len(_TRIPLE.findall(stripped))
            if in_doc:
                s += 0.9 if doc_first else -0.4
                doc_first = False
            elif delims:
                s += 1.0
            if delims % 2 == 1:
                in_doc = not in_doc
                doc_first = in_doc
            # anchor overlap — the core "semantic anchor" signal
            overlap = len(set(code_tokens(line)) & anchors)
            s += self.anchor_weight * min(overlap, 4)
            if self.score_fn is not None:
                s += float(self.score_fn(line))
            scores.append(s)
        return scores

    # ---- compression ----
    def compress(self, code: str, query: str = "", budget_tokens: int = 400,
                 extra_anchors: tuple = (), language: str = "python",
                 use_slicing: bool = True) -> CompressionResult:
        original = approx_tokens(code)
        lines = code.splitlines()
        if original <= budget_tokens or len(lines) <= 3:
            return CompressionResult(code, original, original,
                                     kept_lines=len(lines),
                                     total_lines=len(lines))

        anchors = set(code_tokens(query))
        for a in extra_anchors:
            anchors.update(code_tokens(a))
        anchors -= _ANCHOR_STOPWORDS
        scores = self._line_scores(lines, anchors)

        # backward program slice: lines on the def-use chain to the
        # returns/raises (given the anchors) outrank regex heuristics
        if use_slicing:
            sliced = slice_lines(code, language, anchors)
            if sliced:
                for i in sliced:
                    if i < len(scores):
                        scores[i] += self.slice_bonus

        kept: set[int] = set()
        used = 0
        # a chunk always starts at its declaration: keep the leading
        # signature block regardless of language (until the params close
        # and the line ends in `{`/`:`/`=>` — capped at 5 lines)
        balance = 0
        for i in range(min(5, len(lines))):
            balance += lines[i].count("(") - lines[i].count(")")
            kept.add(i)
            used += approx_tokens(lines[i])
            tail = lines[i].rstrip()
            if balance <= 0 and (tail.endswith(("{", ":", "=>"))
                                 or i == 0 and len(lines) > 1
                                 and not tail.endswith(",")):
                break
        # keyword-matched declarations elsewhere in the snippet survive too
        for i, line in enumerate(lines):
            if not _FORCE_KEEP.match(line):
                continue
            j, balance = i, 0
            while j < len(lines):
                balance += lines[j].count("(") - lines[j].count(")")
                if j not in kept:
                    kept.add(j)
                    used += approx_tokens(lines[j])
                if balance <= 0:
                    break
                j += 1
        # greedy budgeted selection of the remaining highest-value lines
        # (reserving ~1/8 of the budget for the elision markers added later)
        select_budget = max(16, budget_tokens - max(4, budget_tokens // 8))
        for i in sorted(range(len(lines)), key=lambda j: (-scores[j], j)):
            if i in kept or not lines[i].strip():
                continue
            cost = approx_tokens(lines[i])
            if used + cost > select_budget:
                continue
            kept.add(i)
            used += cost

        # reassemble in order with elision markers
        out: list[str] = []
        last = -1
        for i in sorted(kept):
            gap = sum(1 for j in range(last + 1, i) if lines[j].strip())
            if gap:
                indent = re.match(r"\s*", lines[i]).group(0)
                out.append(f"{indent}# … {gap} line(s) elided …")
            out.append(lines[i])
            last = i
        trailing = sum(1 for j in range(last + 1, len(lines)) if lines[j].strip())
        if trailing:
            out.append(f"# … {trailing} line(s) elided …")

        text = "\n".join(out)
        compressed = approx_tokens(text)
        if compressed >= original:      # markers outweighed savings — keep original
            return CompressionResult(code, original, original,
                                     kept_lines=len(lines),
                                     total_lines=len(lines))
        return CompressionResult(text, original, compressed,
                                 kept_lines=len(kept), total_lines=len(lines))

    def compress_many(self, items: list[tuple[str, float, str]], query: str,
                      total_budget: int, floor: int = 80) -> list[CompressionResult]:
        """Compress several snippets under one shared budget.

        ``items`` is ``(label, relevance_weight, code)``; more relevant
        snippets get proportionally more of the budget (LongCodeZip's
        coarse-then-fine idea: whole low-value chunks shrink hardest).
        """
        if not items:
            return []
        weights = [max(w, 1e-6) for _, w, _ in items]
        total_w = sum(weights)
        sizes = [approx_tokens(code) for _, _, code in items]
        shares = [max(floor, int(total_budget * w / total_w)) for w in weights]
        # a snippet never needs more budget than its own size; hand the
        # surplus to snippets that are still over-budget
        surplus = sum(max(0, s - n) for s, n in zip(shares, sizes))
        shares = [min(s, n) for s, n in zip(shares, sizes)]
        needy = [i for i in range(len(items)) if shares[i] < sizes[i]]
        if surplus and needy:
            needy_w = sum(weights[i] for i in needy)
            for i in needy:
                extra = int(surplus * weights[i] / needy_w)
                shares[i] = min(sizes[i], shares[i] + extra)
        results = []
        for (label, _, code), share in zip(items, shares):
            results.append(self.compress(code, query=query,
                                         budget_tokens=share,
                                         extra_anchors=(label,)))
        return results


# --------------------------------------------------------------------------
# Surprisal scoring (training-free stand-in for SLM perplexity)
# --------------------------------------------------------------------------

def make_surprisal_score_fn(texts: list[str], weight: float = 0.6):
    """Build a ``score_fn(line) -> float`` from corpus token surprisal.

    A unigram model with add-one smoothing is fit over the repo's own
    code tokens; a line's score is its mean per-token surprisal
    ``-log p(t)`` centred on the corpus mean. Boilerplate made of
    ubiquitous tokens scores below zero, distinctive logic above — the
    same signal an SLM's conditional log-likelihood provides, without a
    model. To use a real SLM (e.g. Qwen2.5-Coder-0.5B), pass its
    negative per-line perplexity as ``score_fn`` instead; the interface
    is identical.
    """
    from collections import Counter

    counts: Counter = Counter()
    for t in texts:
        counts.update(code_tokens(t))
    total = sum(counts.values()) or 1
    vocab = len(counts) or 1

    def surprisal(tok: str) -> float:
        return -math.log((counts.get(tok, 0) + 1) / (total + vocab))

    mean = (sum(surprisal(t) * c for t, c in counts.items()) / total
            if counts else 0.0)

    def score_fn(line: str) -> float:
        toks = code_tokens(line)
        if not toks:
            return 0.0
        s = sum(surprisal(t) for t in toks) / len(toks)
        return weight * (s - mean)

    return score_fn


# --------------------------------------------------------------------------
# Slice-only rendering (a discrete compression level for the knapsack)
# --------------------------------------------------------------------------

def sliced_text(code: str, query: str = "", language: str = "python",
                extra_anchors: tuple = ()) -> str | None:
    """Render the backward slice of *code* with elision markers, or None."""
    anchors = set(code_tokens(query))
    for a in extra_anchors:
        anchors.update(code_tokens(a))
    kept = slice_lines(code, language, anchors - _ANCHOR_STOPWORDS)
    if not kept:
        return None
    lines = code.splitlines()
    out: list[str] = []
    last = -1
    for i in sorted(k for k in kept if k < len(lines)):
        gap = sum(1 for j in range(last + 1, i) if lines[j].strip())
        if gap:
            indent = re.match(r"\s*", lines[i]).group(0)
            out.append(f"{indent}# … {gap} line(s) elided …")
        out.append(lines[i])
        last = i
    trailing = sum(1 for j in range(last + 1, len(lines)) if lines[j].strip())
    if trailing:
        out.append(f"# … {trailing} line(s) elided …")
    return "\n".join(out)


def _render_kept(code: str, kept: set) -> str:
    lines = code.splitlines()
    out: list[str] = []
    last = -1
    for i in sorted(k for k in kept if k < len(lines)):
        gap = sum(1 for j in range(last + 1, i) if lines[j].strip())
        if gap:
            indent = re.match(r"\s*", lines[i]).group(0)
            out.append(f"{indent}# … {gap} line(s) elided …")
        out.append(lines[i])
        last = i
    trailing = sum(1 for j in range(last + 1, len(lines)) if lines[j].strip())
    if trailing:
        out.append(f"# … {trailing} line(s) elided …")
    return "\n".join(out)


def interprocedural_sliced_text(seed_chunk, graph, chunk_by_id: dict,
                                query: str = "", max_depth: int = 1) -> str:
    """Render an SDG-lite slice: the seed chunk's backward slice, plus a
    backward slice of every callee it invokes (see
    ``slicing.interprocedural_slice``), one section per chunk.

    This is what a "show me the exact execution path" retrieval answers
    when the path crosses a function boundary — the seed alone only shows
    that it *calls* something; this also shows what that something does.
    """
    from .slicing import interprocedural_slice

    anchors = set(code_tokens(query)) - _ANCHOR_STOPWORDS
    visited = interprocedural_slice(seed_chunk, graph, chunk_by_id,
                                    max_depth=max_depth, anchors=anchors)
    sections = []
    for chunk_id, (chunk, kept) in visited.items():
        body = (_render_kept(chunk.source, kept) if kept
               else chunk.source)
        tag = "seed" if chunk_id == seed_chunk.chunk_id else "callee"
        sections.append(f"# [{tag}] {chunk_id}\n{body}")
    return "\n\n".join(sections)


# --------------------------------------------------------------------------
# Multi-choice 0/1 knapsack packing (optimal context assembly)
# --------------------------------------------------------------------------

def knapsack_pack(items: list[list[tuple[str, int, float, object]]],
                  budget: int, granularity: int = 4) -> list[tuple[int, str, object]]:
    """Multi-choice 0/1 knapsack over discrete compression levels.

    ``items[i]`` is the variant list for snippet *i*: tuples of
    ``(level_name, token_cost, value, payload)``; at most one variant
    per snippet is chosen. Solved exactly by dynamic programming on a
    token grid of *granularity* tokens:

        maximize   Σ v_i·x_i    s.t.   Σ c_i·x_i ≤ B,  x_i ∈ {0,1}

    Returns ``[(item_index, level_name, payload), ...]`` in item order.
    """
    cells = max(1, budget // granularity)
    dp = [0.0] * (cells + 1)                      # value per used-cell count
    choice: list[list[int]] = []                  # choice[i][b] = variant idx or -1

    for variants in items:
        row = [-1] * (cells + 1)
        new_dp = dp[:]
        for vi, (_, cost, value, _) in enumerate(variants):
            ccells = max(1, (cost + granularity - 1) // granularity)
            if value <= 0 or ccells > cells:
                continue
            for b in range(cells, ccells - 1, -1):
                cand = dp[b - ccells] + value
                if cand > new_dp[b]:
                    new_dp[b] = cand
                    row[b] = vi
        dp = new_dp
        choice.append(row)

    # backtrack from the best cell
    best_b = max(range(cells + 1), key=lambda b: dp[b])
    picked: list[tuple[int, str, object]] = []
    b = best_b
    for i in range(len(items) - 1, -1, -1):
        vi = choice[i][b]
        # verify this cell's value actually came through item i's choice:
        # recompute dp without item i is expensive; the standard MCKP
        # backtrack stores the row per item, so row[b] == -1 means skip.
        if vi >= 0:
            name, cost, value, payload = items[i][vi]
            ccells = max(1, (cost + granularity - 1) // granularity)
            picked.append((i, name, payload))
            b -= ccells
    picked.reverse()
    return picked
