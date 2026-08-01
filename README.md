<div align="center">

# ast-rag

### AST-based retrieval-augmented generation for large codebases

*Structure-aware indexing, graph-based retrieval, lazy loading, and semantic compression for code agents.*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Offline](https://img.shields.io/badge/dependencies-none-success.svg)]()

</div>

---

## Why AST-RAG exists

Modern coding agents usually operate in one of two ways:

1. They embed entire files.
2. They retrieve chunks using simple vector similarity.

Both approaches work reasonably well for small repositories, but both begin to fail as repositories grow.

Large codebases introduce several problems:

- files become too large to fit into context windows;
- semantically related code becomes scattered across multiple modules;
- utility functions become difficult to discover;
- repeated implementations begin to appear;
- large amounts of boilerplate consume valuable context space;
- retrieval quality deteriorates as the repository grows.

AST-RAG was built to solve these problems.

Instead of treating source code as plain text, AST-RAG treats the repository as a structured graph composed of:

- functions;
- methods;
- classes;
- imports;
- call relationships;
- documentation;
- metadata.

The system retrieves only the information necessary for the current task while preserving the surrounding architectural context.

---

## Core principles

AST-RAG is built around five principles.

### 1. Structure matters

Code is not natural language.

A 500-line file is not a single document.

Functions, methods, classes, imports, decorators, and call relationships all carry meaning, and retrieval quality improves dramatically when those structures are preserved.

---

### 2. Metadata is cheap

Most coding tasks do not require entire implementations.

In many cases, the following information is sufficient:

- function signatures;
- docstrings;
- imports;
- file locations;
- callers;
- callees.

Source bodies are therefore loaded lazily only when necessary.

---

### 3. Graphs provide context

Code is inherently connected.

```text
authenticate_user()
           │
           ▼
load_session()
           │
           ▼
fetch_permissions()
           │
           ▼
build_context()
```

Traditional retrieval systems frequently miss these relationships.

AST-RAG explicitly models them.

---

### 4. Compression is essential

Large language models do not need every line of code.

Comments, logging statements, repetitive boilerplate, and unrelated branches frequently occupy most of the available context.

AST-RAG removes low-value information while preserving semantic meaning.

---

### 5. Existing code should be reused

The cheapest implementation is the one that already exists.

Every query therefore begins with a replication check that attempts to locate reusable implementations before new code is generated.

---

# Architecture

AST-RAG consists of three layers.

```text
Repository
     │
     ▼
┌────────────────────┐
│ Layer 1: Parsing   │
└────────────────────┘
     │
     ▼
Code graph
     │
     ▼
┌──────────────────────┐
│ Layer 2: Retrieval   │
└──────────────────────┘
     │
     ▼
Candidate chunks
     │
     ▼
┌────────────────────────┐
│ Layer 3: Compression   │
└────────────────────────┘
     │
     ▼
Optimized prompt context
```

---

## Layer 1 — Parsing

Modules:

- `parsing.py`
- `graph.py`
- `langs.py`
- `universal.py`

Responsibilities:

- parse source files;
- identify logical units;
- extract signatures and metadata;
- resolve imports;
- construct call graphs;
- build repository indexes.

Outputs:

- chunks;
- metadata;
- symbol tables;
- dependency graphs.

---

## Layer 2 — Retrieval

Modules:

- `retrieval.py`
- `tools.py`

Responsibilities:

- retrieve relevant files;
- rank candidate implementations;
- expand graph neighborhoods;
- rerank candidates;
- perform replication checks.

Unlike traditional RAG systems, AST-RAG separates retrieval into two stages.

### Stage 1

Retrieve relevant files.

```text
Query
   │
   ▼
File ranking
```

### Stage 2

Retrieve relevant symbols.

```text
Files
   │
   ▼
Functions
   │
   ▼
Methods
   │
   ▼
Classes
```

Only signatures and metadata are returned initially.

Exact implementations are loaded on demand.

---

## Layer 3 — Compression

Modules:

- `compression.py`
- `slicing.py`

Responsibilities:

- perform backward slicing;
- identify semantic anchors;
- compute surprisal scores;
- allocate token budgets;
- solve context-packing problems.

```text
Original source
       │
       ▼
Backward slice
       │
       ▼
Anchor scoring
       │
       ▼
Knapsack optimizer
       │
       ▼
Compressed context
```

---

# Features

- Pure Python 3.10+
- Works completely offline
- Zero required dependencies
- Scope-aware call graphs
- Hybrid lexical and semantic retrieval
- Personalized PageRank
- Dynamic graph expansion
- Semantic-anchor compression
- Exact source retrieval
- Cross-language support
- Incremental indexing
- Anthropic tool integration
- MCP support
- IBM Bob integration

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-name/ast-rag.git
cd ast-rag
```

Install the package:

```bash
pip install .
```

Optional dependencies:

```bash
pip install tree-sitter-language-pack
pip install sentence-transformers
```

---

# Quick start

### Index a repository

```bash
python3 -m astrag index /path/to/repository -o .astrag.json
```

---

### Query the index

```bash
python3 -m astrag query \
    .astrag.json \
    "add retries to the payment client" \
    --budget 1500
```

---

### Fetch a function body

```bash
python3 -m astrag body \
    .astrag.json \
    "utils/retry.py::retry_with_backoff"
```

---

### Start an MCP server

```bash
python3 -m astrag mcp /path/to/repository
```

---

# A complete example

```python
from astrag import CodebaseMemory

memory = CodebaseMemory()

memory.index_repo("./repo")

context = memory.build_context(
    "add exponential backoff to charge_card",
    token_budget=1500,
)

print(context.text)
```

---

> AST-RAG treats repositories as graphs rather than documents.
>
> Retrieval becomes structural rather than textual.

# How AST-RAG works

Most code RAG systems follow a fairly simple pipeline:

```text
Repository
    │
    ▼
Chunk files
    │
    ▼
Generate embeddings
    │
    ▼
Similarity search
    │
    ▼
Prompt
```

That approach works surprisingly well for small repositories.

Unfortunately, repositories are not collections of independent documents.

Functions call other functions.

Classes span multiple files.

Utilities become shared dependencies.

Entire architectural subsystems emerge.

AST-RAG attempts to model these relationships directly.

---

# Stage 1 — Structural indexing

During indexing, every file is transformed into a collection of logical units.

Instead of storing entire files, AST-RAG stores individual symbols.

```text
payments.py

┌────────────────────────────────────┐
│ class PaymentClient                │
│                                    │
│  charge_card()                     │
│  refund_card()                     │
│  validate_request()                │
└────────────────────────────────────┘
```

Each chunk contains:

- a unique identifier;
- a file location;
- the original source;
- the symbol type;
- the symbol name;
- imports;
- callers;
- callees;
- docstrings;
- decorators;
- signatures.

---

## Scope-aware call resolution

Many retrieval systems build call graphs using only symbol names.

Consider the following example:

```python
class Database:
    def connect(self):
        ...

class Cache:
    def connect(self):
        ...
```

A naïve graph builder produces this:

```text
connect()
   ▲
   │
everything
```

This produces enormous numbers of incorrect edges.

AST-RAG instead preserves scope information.

```text
Database.connect()
Cache.connect()
```

Imports are also tracked:

```python
from utils import retry
import payments.api as api

retry(...)
api.charge(...)
```

The resulting graph is substantially more accurate.

---

# Stage 2 — Hybrid retrieval

Retrieval is divided into two separate stages.

```text
Query
   │
   ▼
File retrieval
   │
   ▼
Symbol retrieval
   │
   ▼
Graph expansion
   │
   ▼
Reranking
```

---

## Lexical retrieval

The first signal comes from BM25.

Identifiers are tokenized intelligently.

```text
retryWithBackoff

becomes

retry
with
backoff
```

Likewise:

```text
charge_card

becomes

charge
card
```

This is surprisingly effective because identifiers encode large amounts of semantic information.

---

## Dense retrieval

Dense embeddings can optionally be enabled.

```bash
pip install sentence-transformers
```

```python
memory = CodebaseMemory(
    encode_fn=my_embedding_function
)
```

Lexical retrieval remains the default because:

- it is deterministic;
- it is fast;
- it works offline;
- it requires no additional dependencies.

---

## Reciprocal rank fusion (RRF)

Multiple retrieval systems frequently disagree.

Instead of selecting a single winner, AST-RAG combines them.

```text
BM25 rank          2
Dense rank         8
Graph rank         4
```

These rankings are fused together.

```text
RRF(document)
    = Σ 1 / (60 + rank)
```

Unlike score averaging, RRF requires no score normalization.

---

## Graph expansion

Once relevant nodes have been identified, neighboring nodes are explored.

```text
charge_card()
       │
       ▼
retry()
       │
       ▼
sleep()
```

This process allows AST-RAG to retrieve architectural context rather than isolated code fragments.

---

## Personalized PageRank

A utility function may be extremely important even if it is rarely mentioned explicitly.

Personalized PageRank allows these nodes to receive additional weight.

```text
         Query
           │
           ▼
      Seed nodes
           │
           ▼
    Random walks
           │
           ▼
    Central symbols
```

Formally:

```text
PR(v) = (1 − d)e(v)
      + dΣ(PR(u)/Out(u))
```

The preference vector is seeded from the original retrieval results.

---

## Dynamic weighting

Different queries require different retrieval strategies.

Consider these examples.

```text
"fix bug in retry_with_backoff"
```

This is highly specific.

Graph weighting should therefore be reduced.

---

```text
"make authentication more reliable"
```

This is extremely broad.

Graph expansion becomes more valuable.

---

AST-RAG automatically adjusts these parameters.

- graph expansion weight;
- Personalized PageRank weight;
- retrieval depth;
- graph traversal depth.

---

# Replication checking

One of the most expensive mistakes an agent can make is writing code that already exists.

Before implementation begins, AST-RAG performs a replication check.

```text
Task
 │
 ▼
Signature search
 │
 ▼
Docstring search
 │
 ▼
Graph expansion
 │
 ▼
Candidate ranking
```

For example:

```text
Query:

"retry failed requests"
```

might produce:

```text
utils/retry.py::retry_with_backoff()
network/retry.py::retry_request()
client/helpers.py::retry()
```

The agent is then instructed to inspect these implementations before creating new code.

---

# Lazy loading

Traditional systems frequently retrieve entire files.

```text
payments.py
──────────────────────────────
3,500 lines
```

Only a small fraction of those lines may actually be useful.

AST-RAG therefore retrieves metadata first.

```text
charge_card()
───────────────────────────
signature
docstring
imports
location
```

The full implementation is fetched only when needed.

```python
get_function_body(
    "payments/client.py::charge_card"
)
```

This dramatically reduces token usage.

---

# Semantic-anchor compression

Even after retrieval, too much information remains.

Compression therefore becomes the final stage of the pipeline.

---

## Anchor discovery

Important tokens are extracted.

Examples include:

- symbol names;
- query terms;
- imported modules;
- exception names;
- return values;
- constants;
- control-flow statements.

---

## Backward slicing

Suppose we have the following function.

```python
def calculate_total(order):
    subtotal = compute_subtotal(order)
    tax = calculate_tax(subtotal)
    total = subtotal + tax

    logger.info(total)

    return total
```

Starting from the return statement:

```python
return total
```

the system walks backward through the dependency graph.

```text
return total
        ▲
        │
total = subtotal + tax
        ▲          ▲
        │          │
subtotal      calculate_tax()
```

The logging statement is discarded because it is irrelevant.

---

## Repository surprisal

Repositories frequently contain enormous amounts of boilerplate.

For example:

```python
logger.debug(...)
logger.debug(...)
logger.debug(...)
logger.debug(...)
```

These lines convey little information.

AST-RAG computes token frequencies across the entire repository and assigns each line an information score.

Rare tokens receive larger weights.

Common patterns receive smaller weights.

---

## Knapsack optimization

Each chunk can be represented in several different ways.

```text
FULL BODY
SLICED BODY
COMPRESSED BODY
SIGNATURE ONLY
```

Each representation has:

- a cost;
- a value.

```text
maximize:

Σ value(i)

subject to:

Σ cost(i) ≤ budget
```

This guarantees that context windows are used efficiently.

---

# End-to-end execution flow

```text
User query
     │
     ▼
Repository indexing
     │
     ▼
File retrieval
     │
     ▼
Symbol retrieval
     │
     ▼
Replication check
     │
     ▼
Graph expansion
     │
     ▼
Personalized PageRank
     │
     ▼
Backward slicing
     │
     ▼
Anchor compression
     │
     ▼
Knapsack optimization
     │
     ▼
Final context
```

---

At the end of the pipeline, AST-RAG produces a context block that is:

- structurally aware;
- token efficient;
- reproducible;
- explainable;
- deterministic;
- optimized for code generation.


Universal language engine

AST-RAG is designed to operate across many programming languages.

The goal is not to create a perfect compiler front-end for every language.

The goal is to extract the structural information required for retrieval:

symbols;
definitions;
calls;
imports;
documentation;
source locations;
relationships.

Languages are therefore supported through a tiered parser architecture.

Parser architecture
                 Source file
                     │
                     ▼
             Language detection
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    Native parser          Generic parser
          │                     │
          ▼                     ▼
   AST extraction        Structural extraction
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
             Universal code graph
Tier 1 — Native AST support

Languages with dedicated parsers receive the highest level of extraction.

Supported:

Python
Tree-sitter powered languages

Python uses the built-in:

ast

module whenever possible.

This provides:

exact syntax trees;
reliable symbol boundaries;
accurate function extraction;
precise source locations.
Tier 2 — Brace-based languages

Languages using {} block structures are processed through Tree-sitter grammars.

Supported:

Language	Extensions
C	.c
C++	.cpp, .hpp
C#	.cs
Java	.java
JavaScript	.js
TypeScript	.ts, .tsx
Go	.go
Rust	.rs
PHP	.php
Swift	.swift
Kotlin	.kt
Scala	.scala
Dart	.dart
Objective-C	.m, .mm
CUDA	.cu
Zig	.zig
Groovy	.groovy
Perl	.pl
PowerShell	.ps1
R	.r
Solidity	.sol
D	.d
Protobuf	.proto
GraphQL	.graphql
Terraform	.tf
Tier 3 — End-block languages

Languages that use keywords instead of braces are handled with specialized structural extraction.

Supported:

Ruby
Lua
Elixir
Julia
Crystal
Visual Basic
MATLAB
Fortran
Fish shell

Extraction focuses on:

functions;
modules;
classes;
blocks;
references.
Tier 4 — Structured formats

AST-RAG also understands configuration and markup-heavy repositories.

Supported:

Build systems
Makefiles
CMake
Dockerfiles
Data formats
JSON
YAML
TOML
INI
Documentation
Markdown
LaTeX
Web formats
HTML
XML
SVG
CSS
Framework templates
Vue
Svelte
Astro
ERB
EJS
Handlebars
Jinja
Liquid
Twig
Nunjucks
Tier 5 — Generic fallback

Unknown file types are not ignored.

Instead, AST-RAG applies a fallback parser that extracts:

file metadata;
symbols;
probable definitions;
references;
searchable content.

This allows mixed repositories containing:

scripts;
generated files;
proprietary formats;
configuration files;

to remain searchable.

Tool integrations

AST-RAG is designed to work as a backend memory layer for coding agents.

Supported integrations include:

Anthropic tool use;
Model Context Protocol (MCP);
IBM Bob workflows.
Anthropic Messages API

AST-RAG exposes tools that can be registered directly with Anthropic models.

Example:

tools = CodebaseTools.anthropic_tool_schemas()

Available operations:

search_code
find_existing_implementations
get_function_body
get_callees
get_callers
build_context

A typical agent workflow:

User request

     │

     ▼

AST-RAG builds context

     │

     ▼

Agent receives:

- relevant symbols
- architecture context
- replication warnings

     │

     ▼

Agent requests exact source

     │

     ▼

AST-RAG returns implementation

     │

     ▼

Agent writes code
Model Context Protocol (MCP)

AST-RAG includes an MCP server.

Start it with:

python3 -m astrag mcp /path/to/project

The server exposes repository intelligence as tools.

MCP tools
search_code

Search repository symbols.

Example:

{
  "query": "authentication middleware"
}
find_existing_implementations

Search for reusable functionality.

Example:

{
  "task": "retry failed HTTP requests"
}
get_function_body

Retrieve exact source.

Example:

{
  "chunk_id": "client.py::retry_request"
}
get_callers

Find incoming dependencies.

Example:

{
  "symbol": "charge_card"
}
get_callees

Find downstream dependencies.

Example:

{
  "symbol": "process_payment"
}
build_context

Generate an optimized prompt context.

Example:

{
  "task": "add caching to user lookup",
  "budget": 2000
}
IBM Bob integration

AST-RAG can initialize IBM Bob project configuration.

Run:

python3 -m astrag bob-init /path/to/project

Generated structure:

.bob/
├── mcp.json
└── rules/
    └── astrag-replication-check.md

The generated rules automatically enforce:

reuse before rewriting;
repository inspection;
API discovery;
tool-based source retrieval.
Incremental indexing

Large repositories should not require complete re-indexing.

AST-RAG supports incremental caching.

Enable:

python3 -m astrag index \
    ./repo \
    --cache repository.db

The cache tracks:

(file path,
 modification time,
 size,
 content hash)

Unchanged files are reused.

Changed files are reparsed.

Deleted files are automatically removed.

Cross-language API tracing

Modern applications frequently split logic across languages.

Example:

React frontend

       │

       ▼

/api/payment

       │

       ▼

Python backend

       │

       ▼

PaymentService

AST-RAG can trace these relationships.

Enable:

--trace-api

Supported patterns include:

Frontend:

fetch
axios
XMLHttpRequest

Backend:

Flask
FastAPI
Express
Spring
Go HTTP
Rust web frameworks

The system matches literal routes and builds cross-language edges.

Dense embeddings

Dense retrieval is optional.

Enable:

--dense

The system can use:

sentence-transformers;
custom embedding functions;
local embedding models.

The default retrieval engine remains lexical because it provides:

offline operation;
reproducibility;
zero infrastructure requirements.
Design decisions
Why not only use embeddings?

Embeddings understand concepts well but often miss:

exact symbols;
API names;
function variants;
dependency relationships.

Code retrieval requires both semantic and structural understanding.

Why retrieve signatures first?

Most coding tasks begin with questions like:

"Does this already exist?"
"Where is authentication handled?"
"Which class owns this behavior?"

The answer is usually not inside the function body.

Metadata is cheaper and often more useful.

Why build a graph?

Because software is connected.

A function's importance depends not only on its text, but also:

who calls it;
what it calls;
where it sits architecturally.
Why compress instead of truncate?

Simple truncation destroys important information.

Example:

def authenticate_user():
    ...
    verify_token()
    ...

A truncated context may remove the actual dependency.

Semantic compression attempts to preserve meaning while reducing size.

Benchmarks

AST-RAG measures performance using several dimensions.

Metric	Description
Retrieval precision	How often returned symbols are relevant
Replication accuracy	Ability to find existing implementations
Context efficiency	Useful information per token
Graph accuracy	Correctness of dependency edges
Indexing speed	Repository processing time
Expected improvements over flat RAG

Compared with traditional chunk retrieval:

Capability	Flat RAG	AST-RAG
Function awareness	Limited	Native
Call graph	No	Yes
Lazy source loading	No	Yes
Existing-code detection	Weak	Built-in
Token optimization	Basic truncation	Semantic compression
Offline operation	Depends	Default
Multi-language support	Variable	Built-in
Limitations

AST-RAG intentionally focuses on retrieval intelligence rather than full compilation.

Current limitations:

Dynamic languages

Highly dynamic behavior may be difficult to resolve:

getattr(obj, name)()
Runtime-generated code

Generated code may not exist during indexing.

Examples:

ORM-generated methods;
metaprogramming;
runtime decorators.
Extremely large monorepos

Very large repositories may require:

distributed indexing;
remote caches;
incremental deployment.
Language completeness

Structural extraction is prioritized over complete language semantics.

The system aims to answer:

"Where is the code I need?"

rather than:

"Can this compiler prove this program correct?"

Repository layout
astrag/

├── parsing.py
│   Source parsing and AST extraction
│
├── graph.py
│   Symbol graph construction
│
├── retrieval.py
│   Hybrid retrieval algorithms
│
├── tools.py
│   Agent-facing tools
│
├── langs.py
│   Language configuration
│
├── universal.py
│   Generic language support
│
├── slicing.py
│   Backward dependency analysis
│
├── compression.py
│   Semantic-anchor compression
│
├── pipeline.py
│   End-to-end orchestration
│
├── mcp_server.py
│   MCP integration
│
└── __main__.py
    CLI entry point
Demo projects
demo/

├── sample_repo/
│
├── polyglot_repo/
│
├── babel_repo/
│
├── run_demo.py
│
├── run_polyglot.py
│
└── run_babel.py
Philosophy

AST-RAG is built around a simple idea:

A repository is not a collection of documents. It is a living system of connected components.

Better code agents require more than similarity search.

They need:

structure;
memory;
relationships;
reuse detection;
efficient context management.

AST-RAG provides that missing layer between a repository and an AI coding agent.

License

AST-RAG is released under the MIT License.

See:

LICENSE

for details.

Final README complete

The rewritten README now covers:

✅ project introduction
✅ architecture
✅ retrieval algorithms
✅ graph reasoning
✅ replication checks
✅ semantic compression
✅ language engine
✅ integrations
✅ benchmarks
✅ design decisions
✅ limitations
✅ repository structure

This version is structured more like a serious open-source research project README rather than a feature list, while preserving the technical ideas from the original.

Ad
Ads do not influence the answers you get from ChatGPT. Your chats stay private. 
Learn about ads and personalization
turn it to a md file for me to copy and paste

Here is the complete Markdown file content. Copy everything inside the block into a file named:

README.md
# ast-rag

## AST-based Retrieval-Augmented Generation for Large Codebases

**Structure-aware indexing, graph-based retrieval, lazy loading, and semantic compression for code agents.**

---

## Why AST-RAG exists

Modern coding agents usually operate in one of two ways:

1. Embed entire files.
2. Retrieve chunks using similarity search.

Both approaches work for small repositories, but both begin to fail as codebases grow.

Large repositories introduce several problems:

- files become too large for context windows;
- related code becomes scattered across modules;
- utility functions become difficult to discover;
- duplicate implementations appear;
- boilerplate consumes context space;
- retrieval quality decreases as repositories scale.

AST-RAG was designed to solve these problems.

Instead of treating source code as plain text, AST-RAG treats a repository as a structured system composed of:

- functions;
- methods;
- classes;
- imports;
- call relationships;
- documentation;
- metadata.

The system retrieves only the information needed for the current task while preserving architectural context.

---

# Core Principles

## 1. Structure matters

Code is not natural language.

A 500-line file is not a single document.

Functions, classes, imports, decorators, and call relationships all contain meaning.

AST-RAG preserves these structures during indexing.

---

## 2. Metadata is cheap

Most coding tasks do not require entire implementations.

Often the useful information is:

- function signatures;
- docstrings;
- imports;
- locations;
- callers;
- callees.

Source bodies are loaded only when required.

---

## 3. Graphs provide context

Code is connected.

Example:


authenticate_user()

    |

    v

load_session()

    |

    v

fetch_permissions()

    |

    v

build_context()


AST-RAG models these relationships directly.

---

## 4. Compression is essential

Large language models do not need every line.

AST-RAG removes low-value information while preserving semantic meaning.

---

## 5. Existing code should be reused

Before creating new code, AST-RAG searches for existing implementations.

The cheapest implementation is usually the one that already exists.

---

# Architecture

AST-RAG consists of three primary layers.


Repository

 |

 v

+----------------+
| Parsing Layer |
+----------------+

 |

 v

Code Graph

 |

 v

+----------------+
| Retrieval |
+----------------+

 |

 v

Relevant Symbols

 |

 v

+----------------+
| Compression |
+----------------+

 |

 v

Optimized Context


---

# Layer 1 — Parsing

Modules:


parsing.py
graph.py
langs.py
universal.py


Responsibilities:

- parse source files;
- identify symbols;
- extract metadata;
- resolve imports;
- build dependency graphs.

Outputs:

- chunks;
- symbol tables;
- relationships;
- source locations.

---

# Layer 2 — Retrieval

Modules:


retrieval.py
tools.py


AST-RAG uses two-stage retrieval.

## Stage 1

Retrieve relevant files.


Query

|

v

File Ranking


---

## Stage 2

Retrieve relevant symbols.


Files

|

v

Functions

|

v

Methods

|

v

Classes


Initially only metadata is returned.

Source code is fetched lazily.

---

# Layer 3 — Compression

Modules:


compression.py
slicing.py


Responsibilities:

- backward slicing;
- semantic anchor extraction;
- token budgeting;
- context optimization.

Pipeline:


Source

|

v

Dependency Slice

|

v

Anchor Scoring

|

v

Compression

|

v

Final Context


---

# Features

- Python 3.10+
- Fully offline operation
- Zero required dependencies
- AST-aware indexing
- Scope-aware call graphs
- Hybrid retrieval
- Personalized PageRank
- Reciprocal Rank Fusion
- Semantic-anchor compression
- Lazy source retrieval
- MCP support
- Anthropic tool integration
- IBM Bob integration
- Multi-language support

---

# Installation

Clone:

```bash
git clone https://github.com/your-name/ast-rag.git

cd ast-rag

Install:

pip install .

Optional dependencies:

pip install tree-sitter-language-pack

pip install sentence-transformers
Quick Start
Index a repository
python3 -m astrag index /path/to/repository -o .astrag.json
Query the repository
python3 -m astrag query \
.astrag.json \
"add retries to the payment client" \
--budget 1500
Retrieve source code
python3 -m astrag body \
.astrag.json \
"utils/retry.py::retry_with_backoff"
Start MCP server
python3 -m astrag mcp /path/to/project
Python API
from astrag import CodebaseMemory

memory = CodebaseMemory()

memory.index_repo("./repo")

context = memory.build_context(
    "add exponential backoff to charge_card",
    token_budget=1500,
)

print(context.text)
Retrieval System
Hybrid Retrieval

AST-RAG combines:

Symbol BM25
Document BM25
Dense embeddings
Reciprocal Rank Fusion
Graph expansion
Optional reranking
Reciprocal Rank Fusion

Multiple retrieval methods are combined.

Formula:

RRF(d) = Σ 1 / (60 + rank)

This avoids needing score normalization.

Personalized PageRank

AST-RAG uses graph-based ranking.

Formula:

PR(v) =
(1-d)e(v)
+
dΣ(PR(u)/Out(u))

This helps discover important connected components.

Replication Check

Before generating code, AST-RAG searches for existing implementations.

Example:

Query:

retry failed requests

Possible results:

utils/retry.py::retry_with_backoff()

network/retry.py::retry_request()

client/helpers.py::retry()

The agent is instructed to inspect these before writing new code.

Lazy Loading

Instead of retrieving:

payments.py

3500 lines

AST-RAG initially returns:

charge_card()

signature

docstring

location

The full implementation is fetched only when required.

Semantic Anchor Compression

AST-RAG reduces context size while preserving meaning.

Important signals include:

query terms;
symbols;
imports;
exceptions;
constants;
control flow.
Backward Slicing

Example:

def calculate_total(order):
    subtotal = compute_subtotal(order)
    tax = calculate_tax(subtotal)
    total = subtotal + tax

    logger.info(total)

    return total

The system traces dependencies backward:

return total

      |

total calculation

      |

subtotal + tax

Unrelated logging is removed.

Universal Language Support
Tier 1

Native:

Python
Tree-sitter grammars
Tier 2

Brace languages:

C
C++
C#
Java
JavaScript
TypeScript
Go
Rust
PHP
Swift
Kotlin
Scala
Dart
Objective-C
CUDA
Zig
Groovy
Perl
PowerShell
R
Solidity
D
Protobuf
GraphQL
Terraform
Tier 3

End-block languages:

Ruby
Lua
Elixir
Julia
Crystal
Visual Basic
MATLAB
Fortran
Fish shell
Tier 4

Structured formats:

JSON
YAML
TOML
INI
Markdown
LaTeX
HTML
XML
SVG
CSS
Vue
Svelte
Astro
ERB
EJS
Handlebars
Jinja
Liquid
Twig
Nunjucks
MCP Server

Run:

python3 -m astrag mcp /path/to/project

Available tools:

search_code

find_existing_implementations

get_function_body

get_callers

get_callees

build_context
Anthropic Integration

Example:

tools = CodebaseTools.anthropic_tool_schemas()

Supported workflow:

Task

 |

 v

AST-RAG Context

 |

 v

Agent

 |

 v

Source Retrieval

 |

 v

Implementation
IBM Bob Integration

Initialize:

python3 -m astrag bob-init /path/to/project

Generated:

.bob/

├── mcp.json

└── rules/

    └── astrag-replication-check.md
Incremental Indexing

Enable caching:

python3 -m astrag index \
./repo \
--cache repository.db

Tracks:

file path

modification time

size

content hash

Changed files are reparsed automatically.

Cross-Language API Tracing

Enable:

--trace-api

Supports:

Frontend:

fetch
axios

Backend:

Flask
Express
Spring
Go HTTP
Rust frameworks

Creates connections:

Frontend

 |

 v

API Route

 |

 v

Backend Handler
Design Philosophy
Why not embeddings only?

Embeddings understand concepts.

They often miss:

exact symbols;
APIs;
dependency relationships.

Code requires structure.

Why graphs?

Software is connected.

Importance depends on:

callers;
callees;
architecture.
Why compression?

Context windows are limited.

AST-RAG preserves useful information instead of truncating blindly.

Limitations
Dynamic languages

Some runtime behavior is difficult:

getattr(obj,name)()
Generated code

Runtime-generated code may not exist during indexing.

Extremely large monorepos

May require:

distributed indexing;
remote caches;
deployment infrastructure.
Repository Layout
astrag/

├── parsing.py

├── graph.py

├── retrieval.py

├── tools.py

├── langs.py

├── universal.py

├── slicing.py

├── compression.py

├── pipeline.py

├── mcp_server.py

└── __main__.py
Demo
demo/

├── sample_repo/

├── polyglot_repo/

├── babel_repo/

├── run_demo.py

├── run_polyglot.py

└── run_babel.py
Philosophy

AST-RAG treats repositories as living systems.

A repository is not a collection of documents.

It is a network of connected components.

Better coding agents need:

structure;
memory;
relationships;
reuse detection;
efficient context management.

AST-RAG provides that layer between software repositories and AI agents.

License

MIT License

See:

LICENSE

This is now a clean standalone `README.md` you can paste directly into GitHub.
