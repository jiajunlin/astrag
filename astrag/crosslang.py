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

# ---- FFI: extern declarations in C/C++ and Python ctypes/cffi calls ----
#
# Known scope limitation: these patterns match a load-and-call written in
# one contiguous piece of text (e.g. ``ctypes.CDLL(path).add(a, b)`` inline
# in a call expression). The far more common real-world pattern —
# ``lib = ctypes.CDLL(path)`` at module scope, called later as ``lib.add(...)``
# from inside a function — splits across two statements that usually aren't
# even in the same chunk (the binding is module-level; ``CodeChunk.source``
# is just one function's body). Resolving that would mean tracking module-
# level variable bindings across the whole file, which is a real feature
# but a bigger one than this pass — documented here rather than silently
# left to fail on the common case.
_FFI_PATTERNS = [
    # C/C++ extern "C" { ... } block: pull every function name inside
    re.compile(r'extern\s*"C"\s*\{([^}]*)\}', re.S),
    # single extern "C" prototype: name is the identifier right before '('
    # (not ".split()[-1]" of the whole prototype text, which grabs a
    # parameter-list token like "b)" instead of the function name)
    re.compile(r'extern\s*"C"\s+[\w\s\*]*?\b([A-Za-z_]\w*)\s*\('),
    # Python ctypes / cffi: lib.func(), dll.func()
    re.compile(r'\b(?:ctypes\.)?(?:CDLL|WinDLL|LibraryLoader)\([^)]*\)\s*\.\s*([A-Za-z_]\w*)'),
    re.compile(r'\b(?:cffi\.)?FFI\(\)\.(?:dlopen|def_extern)\([^)]*\)\s*\.\s*([A-Za-z_]\w*)'),
]
_FFI_BLOCK = _FFI_PATTERNS[0]


def extract_ffi_targets(source: str) -> set[str]:
    """Return the set of function names that appear to be called via FFI."""
    found: set[str] = set()
    for pat in _FFI_PATTERNS:
        for m in pat.finditer(source):
            if pat is _FFI_BLOCK:
                for name in re.findall(r'\b([A-Za-z_]\w*)\s*\(', m.group(1)):
                    found.add(name)
            elif m.group(1):
                found.add(m.group(1))
    return found


def build_ffi_edges(chunks: list[CodeChunk]) -> list[tuple[str, str, str]]:
    """Edges from chunks that call an FFI function to the C/C++ chunk that
    actually defines it (resolved to a real chunk_id, not a bare name —
    edge endpoints elsewhere in the graph are always chunk_ids, and a
    dangling name-string endpoint wouldn't resolve via ``get_function_body``
    or show up correctly in graph_export)."""
    # name -> chunk_id, for C/C++ functions declared with extern "C" linkage
    ffi_defined: dict[str, str] = {}
    for c in chunks:
        if c.language in ('c', 'cpp', 'objc'):
            if re.search(r'\bextern\s*"C"\s+[\w\s\*]*?\b' + re.escape(c.name)
                        + r'\s*\(', c.source):
                ffi_defined[c.name] = c.chunk_id

    edges: list[tuple[str, str, str]] = []
    for c in chunks:
        for name in extract_ffi_targets(c.source):
            target_id = ffi_defined.get(name)
            if target_id and target_id != c.chunk_id:
                edges.append((c.chunk_id, "ffi_calls", target_id))
    return edges

# ---- RPC: gRPC (protobuf) ----
_PROTO_SERVICE = re.compile(r'service\s+(\w+)\s*\{([^}]*)\}', re.S)
_PROTO_RPC = re.compile(r'rpc\s+(\w+)\s*\([^)]*\)\s*returns\s*\([^)]*\)')


def build_rpc_edges(chunks: list[CodeChunk]) -> list[tuple[str, str, str]]:
    """Link client stub calls to the servicer method that implements them.

    Proto ``service X { rpc Method(...) }`` declares the contract; the
    implementation is normally a class whose name contains the service
    name (e.g. a generated ``XServicer`` base, or a hand-written service
    class) with a method matching the rpc name. Client call sites rarely
    name their stub/channel variable after the literal service name (e.g.
    ``stub.SubmitOrder(...)``, not ``OrderService.SubmitOrder(...)``), so
    matching is done on the *method* name across all known services, not
    on the qualifier — the method names in one proto file are usually
    distinctive enough that this doesn't produce much cross-service noise.
    Only edges that resolve to a real implementing chunk are kept; an
    unresolvable rpc name is skipped rather than pointed at a synthetic
    ``"Service.Method"`` string that isn't a real graph node.
    """
    method_to_service: dict[str, str] = {}
    for c in chunks:
        if c.language == 'proto':
            for sm in _PROTO_SERVICE.finditer(c.source):
                service = sm.group(1)
                for method in _PROTO_RPC.findall(sm.group(2)):
                    method_to_service[method] = service

    if not method_to_service:
        return []

    # servicer implementation: a method whose name is a known rpc method,
    # defined on a class whose name contains the service name (handles both
    # generated `XServicer` bases and hand-written `XServiceImpl`-style names)
    impl_by_method: dict[str, str] = {}
    for c in chunks:
        if c.kind != 'method' or not c.parent or c.name not in method_to_service:
            continue
        service = method_to_service[c.name]
        if service.lower() in c.parent.lower():
            impl_by_method[c.name] = c.chunk_id

    edges: list[tuple[str, str, str]] = []
    for c in chunks:
        for m in re.finditer(r'\b\w+\.(\w+)\s*\(', c.source):
            method = m.group(1)
            target_id = impl_by_method.get(method)
            if target_id and target_id != c.chunk_id and method in method_to_service:
                edges.append((c.chunk_id, "rpc_calls", target_id))
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