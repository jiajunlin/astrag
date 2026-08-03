"""CLI:  python -m astrag {index,query,body,schemas} ..."""
from __future__ import annotations

import argparse
import json
import sys

from .pipeline import CodebaseMemory
from .tools import CodebaseTools


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="astrag",
        description="AST-based two-stage code RAG with semantic-anchor compression")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index", help="parse a repo and save the index")
    p.add_argument("root")
    p.add_argument("-o", "--out", default=".astrag.json")
    p.add_argument("--cache", help="sqlite cache path for incremental "
                                   "re-indexing (skips unchanged files)")
    p.add_argument("--trace-api", action="store_true",
                   help="trace frontend<->backend HTTP route edges "
                        "(crosslang.py)")
    p.add_argument("--dense", action="store_true",
                   help="enable dense embeddings (requires sentence-transformers)")
    p.add_argument("--dense-model", default="all-MiniLM-L6-v2",
                   help="model name for dense encoder")
    p.add_argument("--no-gitignore", action="store_true",
                   help="index files that .gitignore/.astragignore would "
                        "normally exclude")
    p.add_argument("--report", action="store_true",
                   help="print the full skip breakdown (why each file "
                        "wasn't indexed), not just the totals")

    p = sub.add_parser("query", help="build a compressed prompt context")
    p.add_argument("index")
    p.add_argument("query")
    p.add_argument("--budget", type=int, default=1800)
    p.add_argument("--files", type=int, default=4)
    p.add_argument("--chunks", type=int, default=10)
    p.add_argument("--bodies", type=int, default=2)
    p.add_argument("--graph-boost", type=float, default=None,
                   help="weight of structural centrality (0..1); if omitted, auto-tuned")
    p.add_argument("--ppr-weight", type=float, default=None,
                   help="weight of PPR blending (0..1); if omitted, auto-tuned")

    p = sub.add_parser("body", help="print the exact source of one chunk")
    p.add_argument("index")
    p.add_argument("chunk_id")

    p = sub.add_parser("find-chunks", help="list chunks matching a name substring")
    p.add_argument("index")
    p.add_argument("name")

    sub.add_parser("schemas", help="print Anthropic tool-use JSON schemas")

    p = sub.add_parser("mcp", help="serve the tools over MCP stdio "
                                   "(for IBM Bob, Claude Desktop, …)")
    p.add_argument("target", help="repo directory to index, or a saved "
                                  ".astrag.json index")
    p.add_argument("--save", help="also save the index to this path")
    p.add_argument("--cache", help="sqlite cache path for incremental "
                                   "re-indexing when indexing a directory")
    p.add_argument("--trace-api", action="store_true",
                   help="trace frontend<->backend HTTP route edges")
    p.add_argument("--dense", action="store_true",
                   help="enable dense embeddings (requires sentence-transformers)")
    p.add_argument("--dense-model", default="all-MiniLM-L6-v2",
                   help="model name for dense encoder")
    p.add_argument("--no-gitignore", action="store_true",
                   help="index files that .gitignore/.astragignore would "
                        "normally exclude")

    p = sub.add_parser("graph-export", help="export the dependency graph "
                    "for a graph database or GraphViz")
    p.add_argument("index", help="a saved .astrag.json index, or a repo "
                             "directory to index on the fly")
    p.add_argument("-o", "--out", required=True,
                   help="output path; format is chosen by extension "
                        "(.cypher or .graphml or .dot)")
    p.add_argument("--trace-api", action="store_true",
                   help="include frontend<->backend HTTP route edges")

    p = sub.add_parser("bob-init", help="write .bob/mcp.json + .bob/rules "
                                        "into a repo for IBM Bob")
    p.add_argument("repo", help="path to the project Bob will open")

    args = parser.parse_args(argv)

    if args.cmd == "index":
        mem = CodebaseMemory(trace_api_calls=args.trace_api,
                             dense_model=args.dense_model if args.dense else None)
        mem.index_repo(args.root, cache_path=args.cache,
                       respect_gitignore=not args.no_gitignore)
        mem.save(args.out)
        print(f"indexed {args.root}: {mem.stats()} -> {args.out}")
        if args.report:
            print(f"index report: {mem.index_report()}")
    elif args.cmd == "query":
        mem = CodebaseMemory.load(args.index,
                                   graph_boost=args.graph_boost if args.graph_boost is not None else 0.1,
                                   ppr_weight=args.ppr_weight if args.ppr_weight is not None else 0.25)
        ctx = mem.build_context(args.query, token_budget=args.budget,
                                k_files=args.files, k_chunks=args.chunks,
                                fetch_bodies=args.bodies,
                                graph_boost=args.graph_boost,
                                ppr_weight=args.ppr_weight)
        print(ctx.text)
        print(f"\n[stats] {ctx.stats_line()}", file=sys.stderr)
    elif args.cmd == "body":
        mem = CodebaseMemory.load(args.index)
        print(mem.chunk(args.chunk_id).source)
    elif args.cmd == "find-chunks":
        mem = CodebaseMemory.load(args.index)
        results = CodebaseTools(mem).find_chunks_by_name(args.name)
        print(json.dumps(results, indent=2))
    elif args.cmd == "schemas":
        print(json.dumps(CodebaseTools.anthropic_tool_schemas(), indent=2))
    elif args.cmd == "mcp":
        from .mcp_server import serve
        if args.target.endswith(".json"):
            mem = CodebaseMemory.load(args.target)
        else:
            mem = CodebaseMemory(trace_api_calls=args.trace_api,
                                 dense_model=args.dense_model if args.dense else None)
            mem.index_repo(args.target, cache_path=args.cache,
                           respect_gitignore=not args.no_gitignore)
        if args.save:
            mem.save(args.save)
        serve(mem)
    elif args.cmd == "graph-export":
        from .graph_export import export_cypher, export_graphml, export_dot
        if args.index.endswith(".json"):
            mem = CodebaseMemory.load(args.index)
        else:
            mem = CodebaseMemory(trace_api_calls=args.trace_api)
            mem.index_repo(args.index)
        if args.out.endswith(".graphml"):
            export_graphml(mem.graph, args.out)
        elif args.out.endswith(".dot"):
            export_dot(mem.graph, args.out)
        else:
            export_cypher(mem.graph, args.out)
        print(f"exported {mem.graph.stats()} -> {args.out}")
    elif args.cmd == "bob-init":
        for path in _bob_init(args.repo):
            print(f"wrote {path}")
    return 0


_BOB_RULE = """\
# astrag: codebase memory rules

This project exposes an `astrag` MCP server (see `.bob/mcp.json`) that
holds an AST-level index of the codebase. Follow these rules:

1. **Replication check first.** Before writing any new function, call
   `find_existing_implementations` with a one-line description of the
   behaviour. If a strong match exists, call it instead of rewriting it.
2. **Fetch bodies by chunk_id, not by file.** Use `search_code` to find
   the relevant chunk, then `get_function_body(chunk_id=...)` for exact
   source. Do not paste whole files into context.
3. **Kick off tasks with `build_context`.** For any non-trivial task,
   call `build_context(query=<task>, token_budget=1500)` once to get the
   relevant files, API signature cards, replication check, and compressed
   implementations in a single block.
4. Use `get_callers` / `get_callees` before changing a function's
   signature or behaviour to see the blast radius.
"""


def _bob_init(repo: str) -> list[str]:
    """Write .bob/mcp.json (merging if present) + a workspace rule."""
    import os

    repo = os.path.abspath(repo)
    bob_dir = os.path.join(repo, ".bob")
    rules_dir = os.path.join(bob_dir, "rules")
    os.makedirs(rules_dir, exist_ok=True)

    # PYTHONPATH must point at the directory *containing* the astrag package
    pkg_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    mcp_path = os.path.join(bob_dir, "mcp.json")
    config = {"mcpServers": {}}
    if os.path.exists(mcp_path):
        try:
            with open(mcp_path, encoding="utf-8") as fh:
                config = json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass
        config.setdefault("mcpServers", {})
    config["mcpServers"]["astrag"] = {
        "command": "python3",
        "args": ["-m", "astrag", "mcp", "."],
        "cwd": repo,
        "env": {"PYTHONPATH": pkg_parent},
        "alwaysAllow": ["search_code", "find_existing_implementations",
                        "get_function_body", "get_callees", "get_callers",
                        "build_context"],
        "disabled": False,
    }
    with open(mcp_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")

    rule_path = os.path.join(rules_dir, "astrag-replication-check.md")
    with open(rule_path, "w", encoding="utf-8") as fh:
        fh.write(_BOB_RULE)
    return [mcp_path, rule_path]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:          # e.g. `python -m astrag ... | head`
        raise SystemExit(0)