"""Cross-language call tracing via HTTP route matching.

``graph.py``'s call graph resolves calls *within* one language's own
scoping rules (imports, `self`, same-file names) — it has no notion of
"this TypeScript `fetch()` call is actually the same request this Rust
handler answers." That link only exists at the level of the URL path
and HTTP method both sides agree on, so that's what this module matches
on, instead of pretending to trace it through each language's own call
semantics (which don't cross the network boundary at all).

Scope, stated plainly: this is regex pattern-matching over literal
route strings for common frameworks across ~8 languages/ecosystems. It
will miss routes built from string concatenation, config-driven
prefixes, or template variables it doesn't recognise — the same class
of limitation the brace-language engine in ``langs.py`` already has for
syntax. It is not a network trace and not a type checker; it is a
best-effort index of "these two chunks mention the same path," which is
still far more useful than nothing when jumping from a frontend
component to the backend service that answers it.

Usage::

    edges = build_api_edges(chunks)          # [(caller_id, "api_calls", handler_id), ...]
    for src, _, dst in edges:
        graph._add(src, "api_calls", dst)    # fold into an existing CodeGraph
"""
from __future__ import annotations
import re
from .parsing import CodeChunk
from pathlib import Path

# ---- FFI: extern declarations in C/C++ and Python ctypes/cffi calls ----
_FFI_PATTERNS = [
    # C/C++ extern "C" { ... } or extern "C" function prototypes
    re.compile(r'extern\s*"C"\s*\{([^}]*)\}', re.S),
    re.compile(r'extern\s*"C"\s+([^;{]+)\s*[({;]'),
    # Python ctypes / cffi: lib.func(), dll.func()
    re.compile(r'\b(?:ctypes\.)?(?:CDLL|WinDLL|LibraryLoader)\([^)]*\)\s*\.\s*([A-Za-z_]\w*)'),
    re.compile(r'\b(?:cffi\.)?FFI\(\)\.(?:dlopen|def_extern)\([^)]*\)\s*\.\s*([A-Za-z_]\w*)'),
]

def extract_ffi_targets(source: str) -> set[str]:
    """Return set of C function names that appear to be called via FFI."""
    found = set()
    for pat in _FFI_PATTERNS:
        for m in pat.finditer(source):
            if len(m.groups()) == 1 and m.group(1):
                # For extern blocks, we extract function names inside
                if '{' in m.group(0):
                    body = m.group(1)
                    for name in re.findall(r'\b([A-Za-z_]\w*)\s*\(', body):
                        found.add(name)
                else:
                    found.add(m.group(1).strip().split()[-1])
    return found

def build_ffi_edges(chunks: list[CodeChunk]) -> list[tuple[str, str, str]]:
    """Create edges from chunks that call FFI functions to the C/C++ function definitions."""
    # First, collect all function declarations that are marked extern in C/C++ chunks
    ffi_defined = set()
    for c in chunks:
        if c.language in ('c', 'cpp', 'objc'):
            # Look for function prototypes with extern linkage
            if re.search(r'\bextern\s*"C"\s+\w+\s+\*?\s*([A-Za-z_]\w*)\s*\(', c.source):
                ffi_defined.add(c.name)
    edges = []
    for c in chunks:
        targets = extract_ffi_targets(c.source)
        for t in targets:
            if t in ffi_defined:
                edges.append((c.chunk_id, "ffi_calls", t))  # edge to the C function by name; we'll resolve later
    return edges

# ---- RPC: gRPC (protobuf) and GraphQL ----
_PROTO_SERVICE = re.compile(r'service\s+(\w+)\s*\{([^}]*)\}', re.S)
_PROTO_RPC = re.compile(r'rpc\s+(\w+)\s*\([^)]*\)\s*returns\s*\([^)]*\)')
_GRAPHQL_QUERY = re.compile(r'query\s+(\w+)\s*\([^)]*\)\s*\{([^}]*)\}')
_GRAPHQL_MUTATION = re.compile(r'mutation\s+(\w+)\s*\([^)]*\)\s*\{([^}]*)\}')

def build_rpc_edges(chunks: list[CodeChunk]) -> list[tuple[str, str, str]]:
    """Link client stub calls (in any language) to service definitions (proto/GraphQL)."""
    # Collect all service/method definitions from proto and GraphQL files
    service_methods = {}  # service_name -> set of method names
    for c in chunks:
        if c.language == 'proto':
            for sm in _PROTO_SERVICE.finditer(c.source):
                service = sm.group(1)
                body = sm.group(2)
                methods = set(_PROTO_RPC.findall(body))
                service_methods[service] = methods
        elif c.language == 'graphql':
            # Assume GraphQL schema: type Query { ... }
            # we can just record all field names as methods
            pass  # We'll parse later if needed
    # For now, we just link client calls that match service.method pattern
    edges = []
    for c in chunks:
        # Look for calls like service.Method(...) or client.Method(...)
        for m in re.finditer(r'\b(\w+)\.(\w+)\s*\(', c.source):
            service, method = m.group(1), m.group(2)
            if service in service_methods and method in service_methods[service]:
                edges.append((c.chunk_id, "rpc_calls", f"{service}.{method}"))
    return edges

# ---- frontend: HTTP client calls with a literal path -----------------

_FRONTEND_PATTERNS = [
    # fetch('/api/cart'), fetch(`/api/cart/${id}`)
    re.compile(r"""\bfetch\(\s*[`'"]([^`'"]+)[`'"]"""),
    # axios.get('/x'), axios.post('/x'), this.http.get<T>('/x') (Angular)
    re.compile(r"""\b(?:axios|http)\.(?:get|post|put|patch|delete)"""
              r"""(?:<[^>]*>)?\(\s*[`'"]([^`'"]+)[`'"]"""),
    # $.ajax({url: '/x', ...})
    re.compile(r"""\$\.ajax\(\s*\{[^}]*?url\s*:\s*[`'"]([^`'"]+)[`'"]""",
              re.S),
]

# ---- backend: route registrations, per ecosystem ----------------------

_BACKEND_PATTERNS = [
    # Flask/FastAPI: @app.route("/x"), @app.get("/x"), @router.post("/x")
    re.compile(r"""@(?:app|router)\.(?:route|get|post|put|patch|delete)"""
              r"""\(\s*[rf]?["']([^"']+)["']"""),
    # Express/Koa: app.get('/x', ...), router.post("/x", ...)
    re.compile(r"""\b(?:app|router)\.(?:get|post|put|patch|delete)"""
              r"""\(\s*[`'"]([^`'"]+)[`'"]"""),
    # Go net/http / gorilla mux: http.HandleFunc("/x", ...), r.HandleFunc("/x", ...)
    re.compile(r"""\.HandleFunc\(\s*"([^"]+)\""""),
    # Rust actix-web / warp-style attributes: #[get("/x")], #[post("/x")]
    re.compile(r"""#\[(?:get|post|put|patch|delete)\(\s*"([^"]+)\""""),
    # ASP.NET Core attributes: [HttpGet("x")], [Route("api/x")]
    re.compile(r"""\[(?:Http(?:Get|Post|Put|Patch|Delete)|Route)"""
              r"""\(\s*"([^"]+)\""""),
    # Spring: @GetMapping("/x"), @RequestMapping("/x")
    re.compile(r"""@(?:GetMapping|PostMapping|PutMapping|PatchMapping"""
              r"""|DeleteMapping|RequestMapping)"""
              r"""\(\s*"([^"]+)\""""),
]

_PARAM = re.compile(r"""(:[A-Za-z_]\w*|\{[A-Za-z_]\w*\}|<[A-Za-z_]\w*>|\$\{[^}]+\})""")


def normalize_path(path: str) -> str:
    """Strip host/query, lower-case, collapse path params to ``*``."""
    path = path.split("?", 1)[0]
    path = re.sub(r"^https?://[^/]+", "", path)
    path = _PARAM.sub("*", path)
    path = path.strip("/").lower()
    return path


def _extract(text: str, patterns: list) -> set[str]:
    found: set[str] = set()
    for pat in patterns:
        for m in pat.finditer(text):
            p = normalize_path(m.group(1))
            if p:
                found.add(p)
    return found


def build_api_edges(chunks: list[CodeChunk]) -> list[tuple[str, str, str]]:
    """Return ``(frontend_chunk_id, "api_calls", backend_chunk_id)`` edges
    for every literal path that appears in both a frontend HTTP call and a
    backend route registration somewhere in the given chunks."""
    frontend_calls: dict[str, set[str]] = {}
    backend_routes: dict[str, set[str]] = {}
    for c in chunks:
        fp = _extract(c.source, _FRONTEND_PATTERNS)
        if fp:
            frontend_calls[c.chunk_id] = fp
        # Route attributes/decorators (Rust #[...], Python @app.route, Java
        # @GetMapping, ASP.NET [HttpGet(...)]) sit *above* a chunk's own
        # start line, so a parser that (correctly) starts the chunk body at
        # the declaration keyword won't include them in ``c.source`` — check
        # the parsed ``decorators`` list too, not just the body text.
        bp = _extract(c.source, _BACKEND_PATTERNS)
        bp |= _extract("\n".join(c.decorators or []), _BACKEND_PATTERNS)
        if bp:
            backend_routes[c.chunk_id] = bp

    edges: list[tuple[str, str, str]] = []
    for fe_id, paths in frontend_calls.items():
        for be_id, routes in backend_routes.items():
            if paths & routes:
                edges.append((fe_id, "api_calls", be_id))
    return edges


def annotate_graph(graph, chunks: list[CodeChunk]) -> int:
    """Add ``api_calls`` edges (see ``build_api_edges``) into an existing
    ``CodeGraph`` in place. Returns the number of edges added."""
    edges = build_api_edges(chunks)
    edges.extend(build_ffi_edges(chunks))
    edges.extend(build_rpc_edges(chunks))
    for src, kind, dst in edges:
        graph._add(src, kind, dst)
    return len(edges)
