"""MCP (Model Context Protocol) server over STDIO — zero dependencies.

Exposes the astrag tool surface (search_code, find_existing_implementations,
get_function_body, get_callees, get_callers, build_context) to any MCP
client speaking newline-delimited JSON-RPC 2.0 on stdin/stdout — e.g.
IBM Bob, Claude Desktop, or Claude Code.

Run directly:

    python -m astrag mcp /path/to/repo            # index at startup
    python -m astrag mcp /path/to/.astrag.json    # or load a saved index

For IBM Bob, ``python -m astrag bob-init /path/to/repo`` writes the
project-level ``.bob/mcp.json`` (STDIO transport) and a workspace rule
that teaches Bob the replication-check workflow.
"""
from __future__ import annotations

import json
import sys

from . import __version__
from .tools import CodebaseTools

PROTOCOL_FALLBACK = "2025-06-18"


def _mcp_tool_schemas() -> list[dict]:
    """Anthropic-style schemas -> MCP shape (input_schema -> inputSchema)."""
    out = []
    for t in CodebaseTools.anthropic_tool_schemas():
        out.append({"name": t["name"], "description": t["description"],
                    "inputSchema": t["input_schema"]})
    return out


def serve(memory, stdin=None, stdout=None) -> None:
    """Blocking request loop: one JSON-RPC message per line."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    tools = CodebaseTools(memory)

    def send(payload: dict) -> None:
        stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stdout.flush()

    def result(mid, res) -> None:
        send({"jsonrpc": "2.0", "id": mid, "result": res})

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params") or {}

        if method == "initialize":
            result(mid, {
                "protocolVersion": params.get("protocolVersion",
                                              PROTOCOL_FALLBACK),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "astrag", "version": __version__},
            })
        elif method.startswith("notifications/"):
            continue                       # no response to notifications
        elif method == "ping":
            result(mid, {})
        elif method == "tools/list":
            result(mid, {"tools": _mcp_tool_schemas()})
        elif method == "tools/call":
            try:
                out = tools.dispatch(params.get("name", ""),
                                     params.get("arguments") or {})
                text = json.dumps(out, ensure_ascii=False, indent=2)
                result(mid, {"content": [{"type": "text", "text": text}],
                             "isError": False})
            except Exception as exc:       # tool errors flow back as content
                result(mid, {"content": [{"type": "text",
                                          "text": f"error: {exc}"}],
                             "isError": True})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601,
                            "message": f"method not found: {method}"}})
