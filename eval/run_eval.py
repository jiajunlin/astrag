#!/usr/bin/env python3
"""Retrieval quality benchmark: hit-rate@k, MRR, precision@1.

Runs the labeled query set in ``queries_sample_repo.json`` (natural
language task descriptions, each hand-labeled with the chunk_id(s) that
correctly answer it) against ``search_code`` (stage-2 signature-card
retrieval) over the bundled sample repo, and reports standard IR
retrieval metrics:

* **hit-rate@k** — fraction of queries where *some* relevant chunk_id
  appears in the top k results. The core "did retrieval find it" metric.
* **MRR** (Mean Reciprocal Reciprocal Rank) — mean of 1/rank of the first
  relevant hit (0 if none in the candidate pool). Rewards ranking the
  right answer *first*, not just somewhere in the list.
* **precision@1** — fraction of queries where the *top* result is
  relevant. The strictest metric: what an agent sees if it only looks
  at the first hit.

Honest scope note: this measures *retrieval* quality (did the right
signature card get surfaced), not generation quality. CodeBLEU compares
a *generated* code string against a reference and doesn't apply here —
there's no generated code in a retrieval-only pipeline. If a full
code-generation eval is ever added on top of astrag, CodeBLEU (or
simpler exact/AST-match) would be the right metric for that stage, not
this one.

Usage:
    python3 demo/eval/run_eval.py
    python3 demo/eval/run_eval.py --k 5 --queries demo/eval/queries_sample_repo.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from astrag import CodebaseMemory, CodebaseTools  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def run(queries_path: str, repo_path: str, k: int) -> int:
    with open(queries_path, encoding="utf-8") as fh:
        queries = json.load(fh)

    mem = CodebaseMemory().index_repo(repo_path)
    tools = CodebaseTools(mem)

    hits_at_k = 0
    prec_at_1 = 0
    reciprocal_ranks = []
    rows = []

    for item in queries:
        query = item["query"]
        relevant = set(item["relevant"])
        cards = tools.search_code(query, k=k)
        ranked_ids = [c["chunk_id"] for c in cards]

        rank = next((i + 1 for i, cid in enumerate(ranked_ids)
                    if cid in relevant), None)
        hit = rank is not None
        top1 = bool(ranked_ids) and ranked_ids[0] in relevant

        hits_at_k += int(hit)
        prec_at_1 += int(top1)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        rows.append({
            "query": query,
            "expected": sorted(relevant),
            "top_result": ranked_ids[0] if ranked_ids else None,
            "rank_of_first_relevant": rank,
            "hit": hit,
        })

    n = len(queries)
    hit_rate = hits_at_k / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    precision1 = prec_at_1 / n if n else 0.0

    for r in rows:
        mark = "✓" if r["hit"] else "✗"
        print(f"  {mark}  {r['query']!r}")
        print(f"        expected: {r['expected']}")
        print(f"        top hit:  {r['top_result']!r}  "
             f"(rank of first relevant: {r['rank_of_first_relevant']})")

    print()
    print(f"queries        : {n}")
    print(f"hit-rate@{k}    : {hit_rate:.2f}  ({hits_at_k}/{n})")
    print(f"MRR            : {mrr:.3f}")
    print(f"precision@1    : {precision1:.2f}  ({prec_at_1}/{n})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries",
                    default=os.path.join(HERE, "queries_sample_repo.json"))
    ap.add_argument("--repo",
                    default=os.path.join(HERE, "..", "sample_repo"))
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)
    return run(args.queries, args.repo, args.k)


if __name__ == "__main__":
    raise SystemExit(main())
