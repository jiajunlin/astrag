"""Export ``CodeGraph`` to formats a real graph database can ingest.

``graph.py``'s in-memory ``CodeGraph`` is deliberately simple: it's fast
for a single repo and needs no server. Past that scale — a graph shared
across an org's repos, traversed by many services at once — the right
move is a real graph database (Neo4j, NebulaGraph, ...), not rewriting
``CodeGraph`` on top of a heavier in-process library. These exporters
write the graph out in formats those systems already import natively,
with zero new required dependencies:

* ``export_cypher``  -> a ``.cypher`` script of ``CREATE`` statements,
  loadable via ``cypher-shell < out.cypher`` (Neo4j) or any Cypher-
  compatible engine.
* ``export_graphml`` -> standard GraphML XML, importable by Neo4j
  (``apoc.import.graphml``), NebulaGraph Studio, Gephi, and most graph
  tooling.
* ``export_networkx`` -> an in-memory ``networkx.DiGraph``, for anyone
  who wants PyTorch Geometric-style GNN workflows on top; ``networkx``
  is an optional import (``pip install networkx``) and this function is
  simply unavailable without it — nothing else in the package depends
  on it.

None of this changes how astrag itself traverses the graph; it's purely
an escape hatch for scaling storage/traversal beyond one process.
"""
from __future__ import annotations

from xml.sax.saxutils import escape

from .graph import CodeGraph


def _node_label(chunk_id: str) -> str:
    return "File" if chunk_id.startswith("file:") else "Chunk"


def export_cypher(graph: CodeGraph, path: str) -> None:
    """Write CREATE statements for every node and edge to ``path``."""
    nodes = sorted(graph.chunk_ids | {f"file:{f}" for f in graph.files})
    with open(path, "w", encoding="utf-8") as fh:
        for n in nodes:
            label = _node_label(n)
            name = n[len("file:"):] if label == "File" else n
            fh.write(f'CREATE (:{label} {{id: "{_cy_escape(n)}", '
                    f'name: "{_cy_escape(name)}"}});\n')
        fh.write("\n")
        for src, kind, dst in graph.edges:
            rel = kind.upper()
            fh.write(
                f'MATCH (a {{id: "{_cy_escape(src)}"}}), '
                f'(b {{id: "{_cy_escape(dst)}"}}) '
                f'CREATE (a)-[:{rel}]->(b);\n')


def _cy_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def export_graphml(graph: CodeGraph, path: str) -> None:
    """Write the graph as GraphML XML to ``path``."""
    nodes = sorted(graph.chunk_ids | {f"file:{f}" for f in graph.files})
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
        '  <key id="kind" for="edge" attr.name="kind" attr.type="string"/>',
        '  <graph id="astrag" edgedefault="directed">',
    ]
    for n in nodes:
        label = _node_label(n)
        lines.append(f'    <node id="{escape(n)}">')
        lines.append(f'      <data key="label">{escape(label)}</data>')
        lines.append('    </node>')
    for i, (src, kind, dst) in enumerate(graph.edges):
        lines.append(f'    <edge id="e{i}" source="{escape(src)}" '
                    f'target="{escape(dst)}">')
        lines.append(f'      <data key="kind">{escape(kind)}</data>')
        lines.append('    </edge>')
    lines.append('  </graph>')
    lines.append('</graphml>')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def export_networkx(graph: CodeGraph):
    """Return a ``networkx.DiGraph`` (requires ``pip install networkx``)."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "export_networkx requires networkx: pip install networkx"
        ) from exc
    g = nx.DiGraph()
    for n in graph.chunk_ids | {f"file:{f}" for f in graph.files}:
        g.add_node(n, label=_node_label(n))
    for src, kind, dst in graph.edges:
        g.add_edge(src, dst, kind=kind)
    return g


def export_dot(graph: CodeGraph, path: str) -> None:
    """Export graph to GraphViz DOT format."""
    lines = ['digraph CodeGraph {', '  rankdir=LR;']
    for n in sorted(graph.chunk_ids | {f"file:{f}" for f in graph.files}):
        label = _node_label(n)
        lines.append(f'  "{n}" [label="{n}", shape={ "box" if label=="File" else "ellipse" }];')
    for src, kind, dst in graph.edges:
        lines.append(f'  "{src}" -> "{dst}" [label="{kind}"];')
    lines.append('}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))