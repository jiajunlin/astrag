"""Layer 1 (multi-language) — heuristic parsers, zero dependencies.

Accuracy tiers for non-Python files:

1. **tree-sitter** (optional, exact grammars) — used automatically when
   ``tree-sitter-language-pack`` is installed; see ``TreeSitterParser``.
2. **Heuristic brace engine** (this module) — masks strings/comments,
   tracks brace depth, and matches declaration patterns per language.
   Handles C, C++, C#, Java, JavaScript/JSX, TypeScript/TSX, Go, Rust,
   PHP, Swift and Kotlin well enough for retrieval: chunk boundaries,
   signatures, doc comments, call names and imports.
3. **HTML / CSS chunkers** — structural chunking by element / rule.

Heuristics are documented over-approximations: exotic syntax (macros,
heavy templates, generated code) may mis-chunk, but a slightly wrong
boundary only costs retrieval precision — nothing is executed.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from .parsing import CodeChunk, code_tokens

# --------------------------------------------------------------------------
# masking: blank out comments & string literals (newlines preserved)
# --------------------------------------------------------------------------

def mask_source(src: str, line_comments: tuple, block_comments: tuple,
                string_delims: tuple, template_delim: str | None = None) -> str:
    out = list(src)
    i, n = 0, len(src)

    def blank(a: int, b: int) -> None:
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        hit = False
        for tok in line_comments:
            if src.startswith(tok, i):
                j = src.find("\n", i)
                j = n if j == -1 else j
                blank(i, j)
                i = j
                hit = True
                break
        if hit:
            continue
        for op, cl in block_comments:
            if src.startswith(op, i):
                j = src.find(cl, i + len(op))
                j = n if j == -1 else j + len(cl)
                blank(i, j)
                i = j
                hit = True
                break
        if hit:
            continue
        ch = src[i]
        if template_delim and ch == template_delim:
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == template_delim:
                    break
                j += 1
            blank(i, j + 1)
            i = j + 1
            continue
        if ch in string_delims:
            verbatim = i > 0 and src[i - 1] == "@"        # C# @"..."
            j = i + 1
            while j < n:
                c = src[j]
                if c == "\\" and not verbatim:
                    j += 2
                    continue
                if c == ch:
                    break
                if c == "\n" and not verbatim:            # unterminated
                    break
                j += 1
            blank(i, j + 1)
            i = j + 1
            continue
        i += 1
    return "".join(out)


def _depths(masked: str) -> list[int]:
    """depth *before* each character, from { } nesting."""
    depths = [0] * (len(masked) + 1)
    d = 0
    for i, ch in enumerate(masked):
        depths[i] = d
        if ch == "{":
            d += 1
        elif ch == "}":
            d = max(0, d - 1)
    depths[len(masked)] = d
    return depths


def _match(masked: str, pos: int, open_ch: str, close_ch: str) -> int:
    """Index of the bracket matching ``masked[pos]`` (or -1)."""
    depth = 0
    for i in range(pos, len(masked)):
        c = masked[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


# --------------------------------------------------------------------------
# language specs
# --------------------------------------------------------------------------

@dataclass
class Pattern:
    kind: str            # chunk kind: function / class / struct / …
    rx: re.Pattern       # matched on the *masked* source, re.M
    style: str           # func | block | block_or_stmt | stmt | arrow
    container: bool = False   # recurse into body with member patterns


@dataclass
class LangSpec:
    name: str
    extensions: tuple
    decl_patterns: tuple
    member_patterns: tuple = ()
    line_comments: tuple = ("//",)
    block_comments: tuple = (("/*", "*/"),)
    string_delims: tuple = ('"', "'")
    template_delim: str | None = None
    import_patterns: tuple = ()
    call_deny: frozenset = frozenset()


_COMMON_DENY = frozenset(
    "if else for while switch do return case break continue goto new delete "
    "sizeof typeof nameof throw throws try catch finally assert defer go "
    "select await yield in of instanceof not and or match when guard "
    "public private protected static void int this super require include "
    "function fn func fun def where let var const".split())

_MODS = (r"(?:(?:public|private|protected|internal|static|final|abstract|"
         r"virtual|override|async|sealed|readonly|extern|unsafe|partial|"
         r"constexpr|inline|explicit|friend|export|default|declare|open|"
         r"data|suspend|operator|noexcept|new|out|ref|native|synchronized|"
         r"strictfp|transient|volatile|mutating|nonmutating|convenience|"
         r"required|lateinit|tailrec|inner|actual|expect)\s+)*")

_DENY_STMT = (r"(?!\s*(?:if|for|foreach|while|switch|return|else|do|case|"
              r"catch|using|new|throw|sizeof|typeof|nameof|delete|lock|"
              r"fixed|goto|await|yield|assert|synchronized)\b)")


def _p(kind, pattern, style="func", container=False, flags=re.M):
    return Pattern(kind, re.compile(pattern, flags), style, container)


# ---- C ----
_C_FUNC = _p("function",
             r"^[ \t]*" + _DENY_STMT +
             r"[A-Za-z_][\w \t\*\&]*?[\w\*\&][ \t]+\**(?P<name>[A-Za-z_]\w*)"
             r"[ \t]*(?=\()")
_C_FUNC2 = _p("function",                       # return type on its own line
              r"^[A-Za-z_][\w \t\*\&]*\n\**(?P<name>[A-Za-z_]\w*)[ \t]*(?=\()")
_C_TYPE = _p("struct",
             r"^[ \t]*(?:typedef[ \t]+)?(?P<kw>struct|enum|union)[ \t]+"
             r"(?P<name>\w+)?[ \t]*(?=\{)", style="block")

C_SPEC = LangSpec(
    name="c", extensions=(".c", ".h"),
    line_comments=("//", "#"),                  # '#' masks preprocessor lines
    decl_patterns=(_C_FUNC, _C_FUNC2, _C_TYPE),
    import_patterns=(re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]',
                                re.M),),
    call_deny=_COMMON_DENY,
)

# ---- C++ ----
_CPP_NAME = r"(?P<name>~?\w+(?:::~?\w+)*)"
CPP_SPEC = LangSpec(
    name="cpp", extensions=(".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx",
                        ".inl", ".ipp", ".tpp", ".cu", ".cuh"),
    line_comments=("//", "#"),
    decl_patterns=(
        _p("class", r"^[ \t]*(?:template[ \t]*<[^\n]*>[ \t\n]*)?"
                    r"(?P<kw>class|struct)[ \t]+(?P<name>\w+)[^;{(\n]*(?=\{|:)",
           style="block", container=True),
        _p("namespace", r"^[ \t]*namespace[ \t]+(?P<name>[\w:]+)[ \t]*(?=\{)",
           style="block", container=True),
        _p("enum", r"^[ \t]*enum(?:[ \t]+class)?[ \t]+(?P<name>\w+)[^;{\n]*(?=\{)",
           style="block"),
        _p("function", r"^[ \t]*" + _DENY_STMT + _MODS +
           r"[A-Za-z_][\w \t\*\&:<>,]*?[\w\*\&>][ \t]+\**" + _CPP_NAME +
           r"[ \t]*(?=[<(])"),
        _p("function", r"^" + _CPP_NAME + r"[ \t]*(?=\()"),   # ctor at col 0
    ),
    member_patterns=(
        _p("method", r"^[ \t]*" + _DENY_STMT + _MODS +
           r"(?:[A-Za-z_][\w \t\*\&:<>,]*?[\w\*\&>][ \t]+\**)?(?P<name>~?\w+)"
           r"[ \t]*(?=[<(])"),
        _p("class", r"^[ \t]*(?P<kw>class|struct)[ \t]+(?P<name>\w+)"
                    r"[^;{(\n]*(?=\{|:)", style="block", container=True),
    ),
    import_patterns=(re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]',
                                re.M),),
    call_deny=_COMMON_DENY,
)

# ---- C# ----
CSHARP_SPEC = LangSpec(
    name="csharp", extensions=(".cs",),
    decl_patterns=(
        _p("namespace", r"^[ \t]*namespace[ \t]+(?P<name>[\w.]+)[ \t]*(?=\{|;)",
           style="block_or_stmt", container=True),
        _p("class", r"^[ \t]*" + _MODS +
           r"(?P<kw>class|interface|struct|record(?:[ \t]+(?:class|struct))?)"
           r"[ \t]+(?P<name>\w+)[^;{(\n]*(?=[{(:;<\n])",
           style="block", container=True),
        _p("enum", r"^[ \t]*" + _MODS + r"enum[ \t]+(?P<name>\w+)", style="block"),
    ),
    member_patterns=(
        _p("method", r"^[ \t]*" + _DENY_STMT + _MODS +
           r"[A-Za-z_][\w<>,.\[\]\? \t]*?[\w>\?\]][ \t]+(?P<name>\w+)[ \t]*(?=[<(])"),
        _p("class", r"^[ \t]*" + _MODS +
           r"(?P<kw>class|interface|struct|record)[ \t]+(?P<name>\w+)"
           r"[^;{(\n]*(?=[{(:;<\n])", style="block", container=True),
    ),
    import_patterns=(re.compile(r"^[ \t]*using[ \t]+(?:static[ \t]+)?"
                                r"([\w.]+)[ \t]*;", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- Java ----
JAVA_SPEC = LangSpec(
    name="java", extensions=(".java",),
    decl_patterns=(
        _p("class", r"^[ \t]*" + _MODS +
           r"(?P<kw>class|interface|enum|record)[ \t]+(?P<name>\w+)"
           r"[^;{(\n]*(?=[{(<])", style="block", container=True),
    ),
    member_patterns=(
        _p("method", r"^[ \t]*" + _DENY_STMT + _MODS +
           r"(?:<[^>\n]*>[ \t]+)?[A-Za-z_][\w<>,.\[\] \t]*?[\w>\]][ \t]+"
           r"(?P<name>\w+)[ \t]*(?=\()"),
        _p("class", r"^[ \t]*" + _MODS +
           r"(?P<kw>class|interface|enum|record)[ \t]+(?P<name>\w+)"
           r"[^;{(\n]*(?=[{(<])", style="block", container=True),
    ),
    import_patterns=(re.compile(r"^[ \t]*import[ \t]+(?:static[ \t]+)?"
                                r"([\w.*]+)[ \t]*;", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- JavaScript / TypeScript (and JSX / TSX) ----
_JS_DENY = _COMMON_DENY
_TS_COMMON = (
    _p("function", r"^[ \t]*(?:export[ \t]+(?:default[ \t]+)?)?"
       r"(?:declare[ \t]+)?(?:async[ \t]+)?function[ \t]*\*?[ \t]*"
       r"(?P<name>[\w$]+)?[ \t]*(?=[<(])"),
    _p("class", r"^[ \t]*(?:export[ \t]+(?:default[ \t]+)?)?"
       r"(?:declare[ \t]+)?(?:abstract[ \t]+)?class[ \t]+(?P<name>[\w$]+)"
       r"[^{{\n]*(?=\{)", style="block", container=True),
    _p("function", r"^[ \t]*(?:export[ \t]+(?:default[ \t]+)?)?"
       r"(?:const|let|var)[ \t]+(?P<name>[\w$]+)[^=\n;]*=[ \t]*"
       r"(?:async[ \t]*)?(?=\()", style="arrow"),
    _p("function", r"^[ \t]*(?:export[ \t]+(?:default[ \t]+)?)?"
       r"(?:const|let|var)[ \t]+(?P<name>[\w$]+)[^=\n;]*=[ \t]*"
       r"(?:async[ \t]+)?function[ \t]*\*?[ \t]*(?=\()", style="func"),
    _p("function", r"^[ \t]*(?:export[ \t]+(?:default[ \t]+)?)?"
      r"(?:const|let|var)[ \t]+(?P<name>[\w$]+)[^=\n;]*=[ \t]*"
      r"(?:async[ \t]+)?[\w$]+[ \t]*=>", style="arrow_done"),
    _p("interface", r"^[ \t]*(?:export[ \t]+)?(?:declare[ \t]+)?interface[ \t]+"
       r"(?P<name>[\w$]+)[^{\n]*(?=\{)", style="block"),
    _p("enum", r"^[ \t]*(?:export[ \t]+)?(?:declare[ \t]+)?(?:const[ \t]+)?"
       r"enum[ \t]+(?P<name>[\w$]+)[ \t]*(?=\{)", style="block"),
    _p("type", r"^[ \t]*(?:export[ \t]+)?type[ \t]+(?P<name>[\w$]+)"
       r"[^=\n]*=", style="stmt"),
)
_TS_MEMBERS = (
    _p("method", r"^[ \t]*(?!\s*(?:if|for|while|switch|catch|return|new|"
       r"typeof|do|else|try|function)\b)"
       r"(?:(?:public|private|protected|static|readonly|abstract|override|"
       r"async)[ \t]+)*(?:get[ \t]+|set[ \t]+)?\*?[ \t]*"
       r"(?P<name>#?[\w$]+)[ \t]*(?=[<(])"),
)
JS_SPEC = LangSpec(
    name="javascript", extensions=(".js", ".mjs", ".cjs", ".jsx"),
    template_delim="`",
    decl_patterns=_TS_COMMON, member_patterns=_TS_MEMBERS,
    import_patterns=(
        re.compile(r"""^[ \t]*import\b[^\n]*?from[ \t]+['"]([^'"]+)['"]""", re.M),
        re.compile(r"""^[ \t]*import[ \t]+['"]([^'"]+)['"]""", re.M),
        re.compile(r"""\brequire\([ \t]*['"]([^'"]+)['"]""", re.M),
    ),
    call_deny=_JS_DENY,
)
TS_SPEC = LangSpec(
    name="typescript", extensions=(".ts", ".tsx", ".mts", ".cts"),
    template_delim="`",
    decl_patterns=_TS_COMMON, member_patterns=_TS_MEMBERS,
    import_patterns=JS_SPEC.import_patterns,
    call_deny=_JS_DENY,
)

# ---- Go ----
GO_SPEC = LangSpec(
    name="go", extensions=(".go",),
    string_delims=('"',), template_delim="`",
    decl_patterns=(
        _p("function", r"^func[ \t]+(?:\((?P<recv>[^)]*)\)[ \t]+)?"
           r"(?P<name>\w+)[ \t]*(?=[<(])"),
        _p("struct", r"^type[ \t]+(?P<name>\w+)[ \t]+struct[ \t]*(?=\{)",
           style="block"),
        _p("interface", r"^type[ \t]+(?P<name>\w+)[ \t]+interface[ \t]*(?=\{)",
           style="block"),
    ),
    import_patterns=(),          # handled specially (import blocks)
    call_deny=_COMMON_DENY,
)

# ---- Rust ----
RUST_SPEC = LangSpec(
    name="rust", extensions=(".rs",),
    string_delims=('"',),
    decl_patterns=(
        _p("function", r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?"
           r"(?:const[ \t]+|async[ \t]+|unsafe[ \t]+|extern[ \t]+\"[^\"]*\"[ \t]+)*"
           r"fn[ \t]+(?P<name>\w+)[ \t]*(?=[<(])"),
        _p("impl", r"^impl(?:<[^>\n]*>)?[ \t]+(?P<name>[\w:<>, &']+?)[ \t]*(?=\{)",
           style="block", container=True),
        _p("struct", r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?struct[ \t]+"
           r"(?P<name>\w+)", style="block_or_stmt"),
        _p("enum", r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?enum[ \t]+(?P<name>\w+)",
           style="block"),
        _p("trait", r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:unsafe[ \t]+)?"
           r"trait[ \t]+(?P<name>\w+)", style="block", container=True),
        _p("module", r"^[ \t]*(?:pub[ \t]+)?mod[ \t]+(?P<name>\w+)[ \t]*(?=\{)",
           style="block", container=True),
    ),
    member_patterns=(
        _p("method", r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?"
           r"(?:const[ \t]+|async[ \t]+|unsafe[ \t]+|default[ \t]+)*"
           r"fn[ \t]+(?P<name>\w+)[ \t]*(?=[<(])"),
    ),
    import_patterns=(re.compile(r"^[ \t]*use[ \t]+([\w:{}, *]+?)[ \t]*;", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- PHP / Swift / Kotlin ----
PHP_SPEC = LangSpec(
    name="php", extensions=(".php",),
    line_comments=("//", "#"),
    decl_patterns=(
        _p("function", r"^[ \t]*function[ \t]+(?P<name>\w+)[ \t]*(?=\()"),
        _p("class", r"^[ \t]*(?:(?:abstract|final)[ \t]+)?"
           r"(?P<kw>class|interface|trait)[ \t]+(?P<name>\w+)[^{\n]*(?=\{)",
           style="block", container=True),
    ),
    member_patterns=(
        _p("method", r"^[ \t]*(?:(?:public|private|protected|static|final|"
           r"abstract)[ \t]+)*function[ \t]+(?P<name>\w+)[ \t]*(?=\()"),
    ),
    import_patterns=(re.compile(r"^[ \t]*use[ \t]+([\w\\]+)", re.M),),
    call_deny=_COMMON_DENY,
)
SWIFT_SPEC = LangSpec(
    name="swift", extensions=(".swift",),
    decl_patterns=(
        _p("function", r"^[ \t]*" + _MODS + r"func[ \t]+(?P<name>\w+)[ \t]*(?=[<(])"),
        _p("class", r"^[ \t]*" + _MODS +
           r"(?P<kw>class|struct|enum|protocol|extension|actor)[ \t]+"
           r"(?P<name>[\w.]+)[^{\n]*(?=\{)", style="block", container=True),
    ),
    member_patterns=(
        _p("method", r"^[ \t]*" + _MODS +
           r"(?:func[ \t]+(?P<name>\w+)|(?P<init>init))[ \t]*(?=[<(])"),
    ),
    import_patterns=(re.compile(r"^[ \t]*import[ \t]+([\w.]+)", re.M),),
    call_deny=_COMMON_DENY,
)
KOTLIN_SPEC = LangSpec(
    name="kotlin", extensions=(".kt", ".kts"),
    decl_patterns=(
        _p("function", r"^[ \t]*" + _MODS + r"fun[ \t]+(?:<[^>\n]*>[ \t]+)?"
           r"(?:[\w.]+\.)?(?P<name>\w+)[ \t]*(?=\()"),
        _p("class", r"^[ \t]*" + _MODS +
           r"(?P<kw>class|object|interface)[ \t]+(?P<name>\w+)",
           style="block_or_stmt", container=True),
    ),
    member_patterns=(
        _p("method", r"^[ \t]*" + _MODS + r"fun[ \t]+(?P<name>\w+)[ \t]*(?=\()"),
    ),
    import_patterns=(re.compile(r"^[ \t]*import[ \t]+([\w.*]+)", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- Scala ----
SCALA_SPEC = LangSpec(
    name="scala", extensions=(".scala", ".sc"),
    decl_patterns=(
        _p("class", r"^[ \t]*(?:case[ \t]+)?(?P<kw>class|object|trait)[ \t]+"
           r"(?P<name>\w+)", style="block_or_stmt", container=True),
        _p("function", r"^[ \t]*" + _MODS + r"def[ \t]+(?P<name>[\w$]+)",
           style="block_or_stmt"),
    ),
    member_patterns=(
        _p("method", r"^[ \t]+" + _MODS + r"def[ \t]+(?P<name>[\w$]+)",
           style="block_or_stmt"),
    ),
    import_patterns=(re.compile(r"^import[ \t]+([\w.{}, ]+)", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- Dart ----
DART_SPEC = LangSpec(
    name="dart", extensions=(".dart",),
    decl_patterns=(
        _p("class", r"^(?:abstract[ \t]+)?(?P<kw>class|mixin|enum|extension)"
           r"[ \t]+(?P<name>\w+)", style="block", container=True),
        _p("function", r"^" + _DENY_STMT +
           r"(?:[\w<>,\[\]? ]+[ \t]+)?(?P<name>\w+)[ \t]*(?=\()"),
    ),
    member_patterns=(
        _p("method", r"^[ \t]+" + _DENY_STMT +
           r"(?:static[ \t]+|final[ \t]+|const[ \t]+)*"
           r"(?:[\w<>,\[\]? ]+[ \t]+)?(?P<name>\w+)[ \t]*(?=\()"),
    ),
    import_patterns=(re.compile(r"^import[ \t]+'([^']+)'", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- Objective-C (.mm always; .m via content sniff in universal.py) ----
OBJC_SPEC = LangSpec(
    name="objc", extensions=(".mm",),
    decl_patterns=(
        _p("class", r"^@(?P<kw>interface|implementation|protocol)[ \t]+"
           r"(?P<name>\w+)", style="block_or_stmt", container=True),
        _C_FUNC,
    ),
    member_patterns=(
        _p("method", r"^[-+][ \t]*\([^)]*\)[ \t]*(?P<name>\w+)",
           style="block_or_stmt"),
    ),
    line_comments=("//", "#"),
    import_patterns=(re.compile(r'^#[ \t]*import[ \t]*[<"]([^>"]+)[>"]', re.M),),
    call_deny=_COMMON_DENY,
)

# ---- Zig ----
ZIG_SPEC = LangSpec(
    name="zig", extensions=(".zig",),
    string_delims=('"',),
    decl_patterns=(
        _p("function", r"^[ \t]*(?:pub[ \t]+)?(?:export[ \t]+|inline[ \t]+)*"
           r"fn[ \t]+(?P<name>\w+)[ \t]*(?=\()"),
        _p("struct", r"^[ \t]*(?:pub[ \t]+)?const[ \t]+(?P<name>\w+)[ \t]*=[ \t]*"
           r"(?:packed[ \t]+|extern[ \t]+)?(?P<kw>struct|enum|union)[ \t]*(?=[({])",
           style="block"),
    ),
    line_comments=("//",), block_comments=(),
    import_patterns=(re.compile(r'@import\("([^"]+)"\)'),),
    call_deny=_COMMON_DENY,
)

# ---- Groovy / Gradle ----
GROOVY_SPEC = LangSpec(
    name="groovy", extensions=(".groovy", ".gradle"),
    decl_patterns=(
        _p("class", r"^[ \t]*" + _MODS + r"(?P<kw>class|interface|trait|enum)"
           r"[ \t]+(?P<name>\w+)", style="block", container=True),
        _p("function", r"^[ \t]*" + _MODS +
           r"def[ \t]+(?P<name>\w+)[ \t]*(?=\()"),
    ),
    member_patterns=(
        _p("method", r"^[ \t]+" + _MODS + _DENY_STMT +
           r"(?:def[ \t]+|[\w<>\[\]]+[ \t]+)(?P<name>\w+)[ \t]*(?=\()"),
    ),
    import_patterns=(re.compile(r"^import[ \t]+([\w.*]+)", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- Perl ----
PERL_SPEC = LangSpec(
    name="perl", extensions=(".pl", ".pm"),
    line_comments=("#",), block_comments=(),
    decl_patterns=(
        _p("function", r"^[ \t]*sub[ \t]+(?P<name>\w+)", style="block_or_stmt"),
        _p("class", r"^[ \t]*package[ \t]+(?P<name>[\w:]+)", style="stmt"),
    ),
    import_patterns=(re.compile(r"^use[ \t]+([\w:]+)", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- PowerShell ----
POWERSHELL_SPEC = LangSpec(
    name="powershell", extensions=(".ps1", ".psm1"),
    line_comments=("#",), block_comments=(("<#", "#>"),),
    decl_patterns=(
        _p("function", r"^[ \t]*(?:function|filter|workflow)[ \t]+"
           r"(?P<name>[\w-]+)", style="block_or_stmt"),
        _p("class", r"^[ \t]*class[ \t]+(?P<name>\w+)", style="block",
           container=True),
    ),
    member_patterns=(
        _p("method", r"^[ \t]+(?:\[[\w\[\]]+\][ \t]*)?(?P<name>[\w-]+)"
           r"[ \t]*(?=\()"),
    ),
    call_deny=_COMMON_DENY,
)

# ---- R ----
R_SPEC = LangSpec(
    name="r", extensions=(".r", ".R"),
    line_comments=("#",), block_comments=(),
    decl_patterns=(
        _p("function", r"^[ \t]*(?P<name>[\w.]+)[ \t]*(?:<-|=)[ \t]*"
           r"function[ \t]*(?=\()"),
    ),
    import_patterns=(re.compile(r"^(?:library|require)\(([\w.]+)\)", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- Solidity ----
SOLIDITY_SPEC = LangSpec(
    name="solidity", extensions=(".sol",),
    decl_patterns=(
        _p("class", r"^[ \t]*(?:abstract[ \t]+)?(?P<kw>contract|interface|"
           r"library)[ \t]+(?P<name>\w+)", style="block", container=True),
    ),
    member_patterns=(
        _p("method", r"^[ \t]+(?P<kw>function|constructor|modifier|event)"
           r"[ \t]*(?P<name>\w*)[ \t]*(?=\()", style="block_or_stmt"),
    ),
    import_patterns=(re.compile(r'^import[ \t]+"([^"]+)"', re.M),),
    call_deny=_COMMON_DENY,
)

# ---- D ----
D_SPEC = LangSpec(
    name="d", extensions=(".d", ".di"),
    decl_patterns=(
        _p("class", r"^[ \t]*(?P<kw>class|struct|interface|union)[ \t]+"
           r"(?P<name>\w+)", style="block", container=True),
        _C_FUNC,
    ),
    member_patterns=(_p("method", r"^[ \t]+" + _DENY_STMT +
                        r"[\w \t\*\[\]!]*?[ \t](?P<name>\w+)[ \t]*(?=\()"),),
    import_patterns=(re.compile(r"^import[ \t]+([\w.]+)", re.M),),
    call_deny=_COMMON_DENY,
)

# ---- Protobuf ----
PROTO_SPEC = LangSpec(
    name="proto", extensions=(".proto",),
    decl_patterns=(
        _p("struct", r"^[ \t]*(?P<kw>message|enum|service)[ \t]+(?P<name>\w+)"
           r"[ \t]*(?=\{)", style="block", container=True),
    ),
    member_patterns=(
        _p("method", r"^[ \t]+rpc[ \t]+(?P<name>\w+)", style="block_or_stmt"),
        _p("struct", r"^[ \t]+(?P<kw>message|enum)[ \t]+(?P<name>\w+)"
           r"[ \t]*(?=\{)", style="block"),
    ),
    import_patterns=(re.compile(r'^import[ \t]+"([^"]+)"', re.M),),
    call_deny=_COMMON_DENY,
)

# ---- GraphQL ----
GRAPHQL_SPEC = LangSpec(
    name="graphql", extensions=(".graphql", ".gql"),
    line_comments=("#",), block_comments=(),
    decl_patterns=(
        _p("struct", r"^[ \t]*(?:extend[ \t]+)?(?P<kw>type|interface|enum|"
           r"input|union|scalar|schema|directive)[ \t]+(?P<name>@?\w*)",
           style="block_or_stmt"),
    ),
    call_deny=_COMMON_DENY,
)

# ---- Terraform / HCL ----
TERRAFORM_SPEC = LangSpec(
    name="terraform", extensions=(".tf", ".hcl", ".tfvars"),
    line_comments=("#", "//"),
    decl_patterns=(
        _p("struct", r'^[ \t]*(?P<kw>resource|data|module|provider|variable|'
           r'output|locals|terraform)[ \t]*"?(?P<name>[\w.-]*)"?'
           r'(?:[ \t]+"[\w.-]+")?[ \t]*(?=\{)', style="block"),
    ),
    call_deny=_COMMON_DENY,
)

BRACE_SPECS = (C_SPEC, CPP_SPEC, CSHARP_SPEC, JAVA_SPEC, JS_SPEC, TS_SPEC,
               GO_SPEC, RUST_SPEC, PHP_SPEC, SWIFT_SPEC, KOTLIN_SPEC,
               SCALA_SPEC, DART_SPEC, OBJC_SPEC, ZIG_SPEC, GROOVY_SPEC,
               PERL_SPEC, POWERSHELL_SPEC, R_SPEC, SOLIDITY_SPEC, D_SPEC,
               PROTO_SPEC, GRAPHQL_SPEC, TERRAFORM_SPEC)

_CALL_RX = re.compile(r"\b([A-Za-z_$][\w$]*)\s*\(")
_ATTR_RX = re.compile(r"^[ \t]*(@[\w.$]+|\[[^\]]*\]|#\[[^\]]*\]|"
                      r"template[ \t]*<[^>\n]*>)[ \t]*$")


# --------------------------------------------------------------------------
# the brace-language engine
# --------------------------------------------------------------------------

class HeuristicParser:
    """Parse one brace-delimited language according to a ``LangSpec``."""

    def __init__(self, spec: LangSpec) -> None:
        self.spec = spec

    # -- helpers ----------------------------------------------------------
    def _line_of(self, pos: int) -> int:
        return bisect.bisect_right(self._starts, pos)

    def _next_sig(self, masked: str, i: int, end: int) -> int:
        while i < end and masked[i] in " \t\n":
            i += 1
        return i if i < end else -1

    def _skip_generics(self, masked: str, i: int, end: int) -> int:
        j = self._next_sig(masked, i, end)
        if j != -1 and masked[j] == "<":
            depth = 0
            while j < end:
                if masked[j] == "<":
                    depth += 1
                elif masked[j] == ">":
                    depth -= 1
                    if depth == 0:
                        return j + 1
                elif masked[j] in ";{":
                    return i
                j += 1
            return i
        return i

    def _find_body(self, masked: str, i: int, end: int):
        """From ``i``, find the body: ('block', open, close) or ('stmt', semi)."""
        guard = i + 4000
        while i < end and i < guard:
            ch = masked[i]
            if ch == "{":
                close = _match(masked, i, "{", "}")
                if close == -1:
                    return None
                nxt = self._next_sig(masked, close + 1, end)
                # C++ member-init-list braces: `Foo() : a{x}, b{y} { … }`
                if nxt != -1 and masked[nxt] in ",{" and masked[nxt - 1] != "{":
                    if masked[nxt] == ",":
                        i = close + 1
                        continue
                    return ("block", nxt, _match(masked, nxt, "{", "}"))
                return ("block", i, close)
            if ch == ";":
                return ("stmt", i)
            if ch == "}":
                return None
            if ch == "(":
                j = _match(masked, i, "(", ")")
                if j == -1:
                    return None
                i = j + 1
                continue
            if ch == "[":
                j = _match(masked, i, "[", "]")
                i = (j + 1) if j != -1 else i + 1
                continue
            i += 1
        return None

    def _doc_and_decorators(self, decl_line: int):
        """Walk raw lines above the declaration for doc comments / attributes."""
        lines = self._raw_lines
        doc: list[str] = []
        decos: list[str] = []
        i = decl_line - 2                       # 0-based line above decl
        while i >= 0:
            s = lines[i].strip()
            if not s:
                break
            if _ATTR_RX.match(lines[i]):
                decos.insert(0, s)
                i -= 1
                continue
            if s.endswith("*/"):
                block = []
                while i >= 0:
                    t = lines[i].strip()
                    block.insert(0, t)
                    if t.startswith("/*"):
                        break
                    i -= 1
                doc = block + doc
                i -= 1
                continue
            if s.startswith(("///", "//!", "//", "*")) or \
                    ("#" in self.spec.line_comments and s.startswith("#") and
                     not s.startswith("#!")):
                doc.insert(0, s)
                i -= 1
                continue
            break
        cleaned = []
        for ln in doc:
            ln = re.sub(r"^/\*+<?|\*+/$", "", ln).strip()
            ln = re.sub(r"^(///?!?|\*|#+|<summary>|</summary>)\s?", "", ln).strip()
            ln = re.sub(r"^<summary>|</summary>$", "", ln).strip()
            if ln:
                cleaned.append(ln)
        text = "\n".join(cleaned)[:600] or None
        return text, decos

    def _calls(self, masked_body: str, own: str) -> list[str]:
        out: list[str] = []
        for m in _CALL_RX.finditer(masked_body):
            name = m.group(1)
            if name in self.spec.call_deny or name == own or name in out:
                continue
            out.append(name)
            if len(out) >= 40:
                break
        return out

    def _imports(self, raw: str) -> list[str]:
        if self.spec.name == "go":
            out = re.findall(r'^import[ \t]+"([^"]+)"', raw, re.M)
            for block in re.findall(r"^import[ \t]*\(([^)]*)\)", raw, re.M | re.S):
                out += re.findall(r'"([^"]+)"', block)
            return out[:40]
        out: list[str] = []
        for rx in self.spec.import_patterns:
            out += [m.group(1) for m in rx.finditer(raw)]
        return out[:40]

    # -- main -------------------------------------------------------------
    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        spec = self.spec
        self._raw = source
        self._raw_lines = source.splitlines()
        self._starts = [0]
        for k, ch in enumerate(source):
            if ch == "\n":
                self._starts.append(k + 1)
        masked = mask_source(source, spec.line_comments, spec.block_comments,
                             spec.string_delims, spec.template_delim)
        self._masked = masked
        self._depth = _depths(masked)
        imports = self._imports(source)
        chunks: list[CodeChunk] = []
        self._scan(0, len(masked), 0, None, spec.decl_patterns, rel_path,
                   chunks, imports, parent_kind=None)
        chunks.sort(key=lambda c: c.start_line)
        return chunks

    def _scan(self, start, end, base_depth, parent, patterns, rel, chunks,
              imports, parent_kind=None) -> None:
        masked = self._masked
        claimed: list[tuple[int, int]] = []
        found: list[tuple[int, int, Pattern, re.Match]] = []
        for pat in patterns:
            for m in pat.rx.finditer(masked, start, end):
                p = m.start()
                while p < end and masked[p] in " \t":
                    p += 1
                if self._depth[p] != base_depth:
                    continue
                found.append((p, m.end(), pat, m))
        found.sort()
        for p, mend, pat, m in found:
            if any(a <= p < b for a, b in claimed):
                continue
            span = self._materialise(p, mend, pat, m, end, base_depth,
                                     parent, rel, chunks, imports, parent_kind)
            if span:
                claimed.append(span)

    def _materialise(self, p, mend, pat, m, end, base_depth, parent, rel,
                     chunks, imports, parent_kind=None):
        masked = self._masked
        spec = self.spec
        style = pat.style
        body = None

        if style in ("func", "arrow"):
            i = self._skip_generics(masked, mend, end)
            i = self._next_sig(masked, i, end)
            if i == -1 or masked[i] != "(":
                return None
            close = _match(masked, i, "(", ")")
            if close == -1:
                return None
            after = close + 1
            if style == "arrow":
                arrow = masked.find("=>", after, min(end, after + 300))
                stop = masked.find(";", after, min(end, after + 300))
                if arrow == -1 or (stop != -1 and stop < arrow):
                    return None
                after = arrow + 2
                nxt = self._next_sig(masked, after, end)
                if nxt != -1 and masked[nxt] == "(":
                    pc = _match(masked, nxt, "(", ")")
                    semi = self._next_sig(masked, pc + 1, end)
                    e = semi if (semi != -1 and masked[semi] == ";") else pc
                    body = ("stmt", e)
                elif nxt != -1 and masked[nxt] == "{":
                    body = ("block", nxt, _match(masked, nxt, "{", "}"))
                else:
                    body = self._stmt_end(after, end, base_depth)
            else:
                body = self._find_body(masked, after, end)
                if body and body[0] == "stmt":       # prototype — skip
                    return None
        elif style == "arrow_done":
            body = self._body_after_arrow(mend, end, base_depth)
        elif style in ("block", "block_or_stmt"):
            body = self._find_body(masked, mend, end)
            if body is None:
                return None
            if body[0] == "stmt":
                if pat.kind == "namespace":          # C# file-scoped namespace
                    body = ("region", body[1] + 1, end - 1)
                elif style == "block":
                    return None
        elif style == "stmt":
            body = self._stmt_end(mend, end, base_depth)
        if body is None:
            return None

        span_end = body[2] if body[0] in ("block", "region") else body[1]
        start_line = self._line_of(p)
        end_line = self._line_of(span_end)
        gd = m.groupdict()
        name = gd.get("name")
        if not name and gd.get("init"):
            name = "init"
        if not name:
            name = f"{pat.kind}@L{start_line}"
        name = name.strip()
        if gd.get("recv"):                           # Go method receiver
            rt = re.findall(r"[A-Za-z_]\w*", gd["recv"])
            if rt:
                parent = rt[-1]
        class_like = parent_kind in ("class", "struct", "impl", "trait",
                                     "interface", "object", "extension",
                                     "record", "protocol", "actor")
        kind = pat.kind
        if kind in ("function", "method"):
            kind = "method" if (parent and (class_like or gd.get("recv"))) \
                else "function"
        qual = f"{parent}.{name}" if parent else name

        sig_end = masked.find("\n", p)
        sig_end = sig_end if sig_end != -1 else span_end
        header_stop = sig_end if body[0] == "region" else min(sig_end, body[1])
        signature = re.sub(r"\s+", " ", self._raw[p:max(p, header_stop)]).strip()
        signature = (signature[:158] + "…") if len(signature) > 160 else signature
        doc, decos = self._doc_and_decorators(start_line)
        body_masked = masked[p:span_end + 1]

        chunks.append(CodeChunk(
            chunk_id=f"{rel}::{qual}", kind=kind, name=name, qualname=qual,
            file=rel, start_line=start_line, end_line=end_line,
            signature=signature or name, docstring=doc,
            source="\n".join(self._raw_lines[start_line - 1:end_line]),
            language=spec.name, parent=parent, decorators=decos,
            calls=self._calls(body_masked, name), imports=list(imports),
        ))
        if pat.container and body[0] in ("block", "region"):
            if pat.kind in ("namespace", "module"):
                inner = spec.decl_patterns
            else:
                inner = spec.member_patterns or spec.decl_patterns
            if body[0] == "region":
                b0, inner_depth = body[1], base_depth
            else:
                b0, inner_depth = body[1] + 1, base_depth + 1
            self._scan(b0, span_end, inner_depth, qual, inner, rel, chunks,
                       imports, parent_kind=pat.kind)
        return (p, span_end + 1)

    def _stmt_end(self, i, end, base_depth):
        masked = self._masked
        j = i
        while j < end:
            if masked[j] == ";" and self._depth[j] == base_depth:
                return ("stmt", j)
            if masked[j] == "\n" and masked.find(";", i, j) == -1 and j - i > 400:
                break
            j += 1
        k = masked.find("\n", i)
        return ("stmt", k if k != -1 else end - 1)

    def _body_after_arrow(self, mend, end, base_depth):
        nxt = self._next_sig(self._masked, mend, end)
        if nxt != -1 and self._masked[nxt] == "{":
            return ("block", nxt, _match(self._masked, nxt, "{", "}"))
        return self._stmt_end(mend, end, base_depth)


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------

_SEMANTIC_TAGS = {"header", "nav", "main", "section", "article", "aside",
                  "footer", "form", "template", "dialog", "table"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
              "link", "meta", "param", "source", "track", "wbr"}


class _HtmlCollector(HTMLParser):
    def __init__(self, starts):
        super().__init__(convert_charrefs=True)
        self._starts = starts
        self.stack: list[dict] = []
        self.spans: list[dict] = []

    def _pos(self):
        line, col = self.getpos()
        return self._starts[line - 1] + col

    def handle_starttag(self, tag, attrs):
        if tag in _VOID_TAGS:
            return
        self.stack.append({"tag": tag, "attrs": dict(attrs),
                           "start": self._pos(), "line": self.getpos()[0]})

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                el = self.stack.pop(i)
                del self.stack[i:]
                el["end"] = self._pos() + len(tag) + 3
                aid = el["attrs"].get("id")
                if aid or tag in _SEMANTIC_TAGS or tag in ("script", "style"):
                    self.spans.append(el)
                return


class HtmlChunker:
    extensions = (".html", ".htm")
    name = "html"

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        starts = [0]
        for k, ch in enumerate(source):
            if ch == "\n":
                starts.append(k + 1)
        col = _HtmlCollector(starts)
        try:
            col.feed(source)
        except Exception:
            return []
        chunks: list[CodeChunk] = []
        seen: set[str] = set()
        for el in sorted(col.spans, key=lambda e: e["start"]):
            tag, attrs = el["tag"], el["attrs"]
            aid = attrs.get("id")
            name = f"{tag}#{aid}" if aid else (
                attrs.get("src") or f"{tag}@L{el['line']}")
            if name in seen:
                name = f"{name}@L{el['line']}"
            seen.add(name)
            kind = tag if tag in ("script", "style", "form", "template") \
                else "element"
            body = source[el["start"]:el.get("end", len(source))]
            head = body.split(">", 1)[0] + ">"
            end_line = el["line"] + body.count("\n")
            calls = []
            if tag == "script":
                calls = [c for c in dict.fromkeys(
                    m.group(1) for m in _CALL_RX.finditer(body))
                    if c not in _COMMON_DENY][:40]
            chunks.append(CodeChunk(
                chunk_id=f"{rel_path}::{name}", kind=kind, name=name,
                qualname=name, file=rel_path, start_line=el["line"],
                end_line=end_line,
                signature=re.sub(r"\s+", " ", head)[:160],
                docstring=attrs.get("title") or attrs.get("aria-label"),
                source=body, language="html", calls=calls,
            ))
        return chunks


# --------------------------------------------------------------------------
# CSS
# --------------------------------------------------------------------------

class CssChunker:
    extensions = (".css", ".scss", ".less")
    name = "css"

    def parse_file(self, rel_path: str, source: str) -> list[CodeChunk]:
        masked = mask_source(source, (), (("/*", "*/"),), ('"', "'"))
        starts = [0]
        for k, ch in enumerate(source):
            if ch == "\n":
                starts.append(k + 1)
        chunks: list[CodeChunk] = []
        self._scan(source, masked, 0, len(masked), starts, None,
                   rel_path, chunks)
        return chunks

    def _scan(self, raw, masked, start, end, starts, parent, rel, chunks):
        i = start
        sel_start = start
        while i < end:
            ch = masked[i]
            if ch == "{":
                close = _match(masked, i, "{", "}")
                if close == -1:
                    break
                sel = re.sub(r"\s+", " ", raw[sel_start:i]).strip()
                if sel:
                    name = sel[:60]
                    kind = "at_rule" if sel.startswith("@") else "rule"
                    lead = len(raw[sel_start:i]) - len(raw[sel_start:i].lstrip())
                    line = bisect.bisect_right(starts, sel_start + lead)
                    qual = f"{parent} > {name}" if parent else name
                    chunks.append(CodeChunk(
                        chunk_id=f"{rel}::{qual}", kind=kind, name=name,
                        qualname=qual, file=rel,
                        start_line=line,
                        end_line=bisect.bisect_right(starts, close),
                        signature=f"{name} {{ … }}", docstring=None,
                        source=raw[sel_start + lead:close + 1],
                        language="css", parent=parent,
                        calls=list(dict.fromkeys(
                            re.findall(r"([-a-zA-Z]{3,})\s*:",
                                       masked[i:close])))[:30],
                    ))
                    if kind == "at_rule" and any(
                            sel.startswith(p) for p in
                            ("@media", "@supports", "@layer", "@container")):
                        self._scan(raw, masked, i + 1, close, starts, name,
                                   rel, chunks)
                i = close + 1
                sel_start = i
            elif ch == ";":
                sel_start = i + 1
                i += 1
            else:
                i += 1


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

_EXT_SPEC: dict[str, LangSpec] = {}
for _s in BRACE_SPECS:
    for _e in _s.extensions:
        _EXT_SPEC[_e] = _s

_parsers: dict[str, object] = {}


def heuristic_parser_for(ext: str):
    """Return a cached parser instance for the extension, or ``None``."""
    if ext in _parsers:
        return _parsers[ext]
    inst = None
    if ext in _EXT_SPEC:
        inst = HeuristicParser(_EXT_SPEC[ext])
    elif ext in HtmlChunker.extensions:
        inst = HtmlChunker()
    elif ext in CssChunker.extensions:
        inst = CssChunker()
    _parsers[ext] = inst
    return inst


# extensions with a *dedicated* parser; universal.py adds end-block
# languages, data formats, and a generic fallback covering everything else
SUPPORTED_EXTENSIONS = tuple(sorted(
    {".py"} | set(_EXT_SPEC) | set(HtmlChunker.extensions)
    | set(CssChunker.extensions)))
